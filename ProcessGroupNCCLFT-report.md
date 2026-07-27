# ProcessGroupNCCLFT 架構分析報告

## 1. 整體架構概覽

`ProcessGroupNCCLFT` 是以原生 `ProcessGroupNCCL` 為基礎，加入容錯（Fault Tolerance）控制面的 PyTorch 分散式後端。它在不修改正常 NCCL 熱路徑的前提下，透過三個平行機制實現在 NIC 故障後的無縫切換：

```
┌─────────────────────────────────────────────────────────────┐
│                   Python 訓練迴圈                            │
│          allreduce() -> allgather() -> optimizer             │
└───────────────────┬─────────────────────────────────────────┘
                    │ collective() 呼叫
┌───────────────────▼─────────────────────────────────────────┐
│               主執行緒 (Main Thread)                         │
│  ┌──────────────────────────────────────────────────┐       │
│  │  collective() 熱路徑                              │       │
│  │  ├─ [正常]  fn() on original ncclComm            │       │
│  │  └─ [降級]  execute_shadow_allreduce()            │       │
│  │             ├─ Faulty GPU: NVLink send/recv       │       │
│  │             ├─ Proxy GPU:  recv+add+AllReduce     │       │
│  │             └─ Healthy GPU: AllReduce on proxy    │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│               Watchdog 執行緒 (pt_nccl_watchdg)             │
│  workMetaList_ 輪詢 -> 偵測 work.exception()                │
│  [FT enabled] -> 清除例外、標記 local_hardware_fault_dev_   │
│  [FT disabled] -> 原始 abort + throw                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│               側車執行緒 (pt_nccl_ft_side)                   │
│  輪詢 local_hardware_fault_dev_ -> 寫 NCCL_FT_EVENT         │
│  監聽 NCCL_FT_EVENT -> 寫 ACK key -> 等 COMMIT -> 解鎖主執行緒│
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 關鍵資料成員

| 成員 | 類型 | 用途 |
|---|---|---|
| `ft_disabled_` | `bool` | 由 `NCCL_FT_DISABLE=1` 環境變數控制；啟用後 FT 完全旁路 |
| `is_degraded_` | `bool` | 降級模式旗標；`true` 時所有 AllReduce 走 Shadow Ping-Pong |
| `local_hardware_fault_dev_` | `atomic<int>` | Watchdog→側車 的單向信號；-1 為正常，>=0 為故障 local device index |
| `do_not_cross_op_` | `atomic<uint64_t>` | 側車->主執行緒的「警戒線」OP 序號 |
| `final_commit_op_` | `atomic<uint64_t>` | 全網 quorum 確認的提交序號；主執行緒在此解除忙等 |
| `failed_dev_index_` | `atomic<int>` | 全網公認的故障 local device index |
| `local_nvlink_comm_` | `shared_ptr<NCCLFTComm>` | 節點內 NVLink-only communicator；由 ncclCommSplit 建立，永不重掃 NIC |
| `proxy_global_comm_` | `shared_ptr<NCCLFTComm>` | 排除故障裝置後的縮減跨節點 communicator；由 ncclCommSplit 建立 |
| `proxy_comm_ready_` | `atomic<bool>` | 保護 `proxy_global_comm_` 初始化競態；`false` 時降級路徑退回 native |
| `proxy_failed_local_dev_` | `int` | 已確認故障的 local device index（供 execute_shadow_allreduce 用） |
| `proxy_comm_rank_/size_` | `int` | 此 rank 在 `proxy_global_comm_` 中的新 rank/size |
| `current_shadow_reduce_op_` | `ReduceOp` | allreduce_impl 設定的實際 ReduceOp；供降級路徑使用 |

---

## 3. 建構子初始化流程

```
ProcessGroupNCCLFT::ProcessGroupNCCLFT()
|
|-- 1. 初始化 Backend(rank, size)
|-- 2. 讀取環境變數（BLOCKING_WAIT、ASYNC_ERROR_HANDLING 等）
|-- 3. 建立 HeartbeatMonitor、Watchdog 物件（尚未 start）
|
|-- 4. 讀取 NCCL_FT_DISABLE 環境變數
|    |-- [NCCL_FT_DISABLE=1]
|    |    └─ is_degraded_ = false，輸出 WARNING，不啟動 FT
|    └─ [正常]
|        |-- 將 this 插入 g_ft_pg_instances（全域 set）
|        |-- start_ft_negotiator_thread()  ->  pt_nccl_ft_side 側車執行緒
|        └─ local_nvlink_comm_ 延遲到第一次 collective() 時初始化
|
|-- 5. (ENABLE_NCCL_ERROR_CHECKING) watchdog_->start()
|-- 6. init()（正常 NCCL communicator 懶初始化）
└─ 7. 記錄環境設定 LOG
```

---

## 4. Watchdog 運作流程

### 4.1 正常路徑（無故障）

```
Watchdog::runLoop()  每 100ms 輪詢一次
|
└─ for each work in workMetaList_:
    |-- work.checkAndSetException()      查詢 ncclCommGetAsyncError
    |-- work.checkTimeout()              超時偵測
    |-- work.isStarted() / isCompleted() 更新 pgStatus_
    └─ work 完成 -> 從 workMetaList_ 移除
```

### 4.2 故障路徑（有 NCCL 例外）

```
work.exception() != nullptr
|
|-- [ft_disabled_ == true]  （原始路徑）
|   |-- LOG(ERROR) failure detected
|   |-- broadcastDumpSignal()
|   |-- sleep(getDumpTimeout() * 4)
|   |-- work.abort() + pg_->abortComms()
|   └─ work.handleException()  -> 拋出例外，終止訓練
|
└─ [ft_disabled_ == false]  （FT 容錯路徑）
    |
    |-- device_idx = work.device_.index() % localDeviceCount_
    |-- LOG(WARNING) Recoverable NIC fault on device X
    |
    |-- pg_->error_ = SUCCESS           解除後續 op 的阻塞
    |-- local_hardware_fault_dev_.store(device_idx)  信號給側車執行緒
    |-- work.setException(nullptr)      清除毒化，wait() 不再拋出
    |-- workMetaList_.erase(it)         移除死亡 work，避免無限輪詢
    |-- setLastWorkListUpdateTime(now)
    └─ continue                         繼續處理下一個 work
```

---

## 5. 網卡出錯時的完整流程

### Phase 1：偵測（< 1ms）

```
NCCL 底層偵測 NIC 故障
  -> ncclCommGetAsyncError() 回傳非 ncclSuccess
  -> (可選) ncclFaultCallback 觸發 nccl_ft_global_fault_callback(dev_idx)
       -> trigger_fault_proposal(dev_idx)
       -> local_hardware_fault_dev_.store(dev_idx)   原子寫入，< 1µs

OR

Watchdog 輪詢 work.checkAndSetException()
  -> work.exception_ 被設置
  -> FT 容錯路徑: local_hardware_fault_dev_.store(device_idx)
```

### Phase 2：2PC TCPStore 協商（~50-200ms，在側車執行緒）

```
ft_negotiator_thread_（pt_nccl_ft_side）

Phase 1 - PROPOSE:
  local_hardware_fault_dev_ != -1
  |-- target_op = seqCollective_ + 10
  |-- proposal = "PROPOSE:<target_op>:<node_id>:<dev_idx>"
  |-- globalStore_->set("NCCL_FT_EVENT", vec)
  |-- local_hardware_fault_dev_.store(-1)          復位信號
  └─ LOG 側車代發提案

所有 rank 的側車執行緒讀到 NCCL_FT_EVENT:
  |-- do_not_cross_op_.store(target_op)   設定主執行緒警戒線
  |-- failed_dev_index_.store(f_dev)
  |-- globalStore_->set("NCCL_FT_ACK_<rank>_<target_op>", "1")

Phase 2 - COMMIT (rank 0 側車):
  輪詢直到所有 rank 的 ACK key 都存在
  |-- globalStore_->set("NCCL_FT_COMMIT_<target_op>", "1")

所有 rank 的側車執行緒讀到 COMMIT key:
  |-- final_commit_op_.store(target_op)   解鎖主執行緒
  └─ 休眠防抖
```

### Phase 3：主執行緒煞車對齊

```
collective() 呼叫
|
|-- current_op = seqCollective_
|-- boundary = do_not_cross_op_.load()
|
└─ boundary > 0 && current_op == boundary  撞上警戒線
    |-- LOG "[NCCL-FT-TRACE] 主執行緒抵達警戒線"
    |-- while (final_commit_op_ == 0) yield()   忙等 quorum 確認
    └─ final_commit_op_ == current_op  確認
        |-- rebuild_shadow_ping_pong_topology()
        |-- is_degraded_ = true
        |-- do_not_cross_op_.store(0)
        |-- final_commit_op_.store(0)
        └─ (rank==0) globalStore_->deleteKey("NCCL_FT_EVENT")
```

### Phase 4：重建 proxy 拓撲

```
proxy_comm_ready_.store(false)          防止使用半初始化 comm
proxy_global_comm_.reset()              銷毀舊 comm（RAII）

計算 rank 映射：
  proxy_comm_size_ = size_ - (size_ / localDeviceCount_)
  遍歷所有 rank，跳過 local_rank == failed_dev，重新編號

取得 global comm（devNCCLCommMap_[key]）:

NCCLFTComm::split(globalComm, color, rank_, config):
  healthy rank: color=1  -> 加入 proxy_global_comm_ 子群
  faulty rank:  color=NCCL_SPLIT_NOCOLOR -> 排除

if !is_faulty:
    proxy_global_comm_ = NCCLFTComm::split(...)
else:
    proxy_global_comm_.reset()  (不參與跨節點通訊)

proxy_comm_ready_.store(true)           通知 execute_shadow_allreduce 可用
```

### Phase 5：降級後的 AllReduce

```
allreduce_impl():
    current_shadow_reduce_op_ = opts.reduceOp   記錄實際 op

collective() degraded path:
    execute_shadow_allreduce(inputs[0], outputs[0],
                             ncclStream, current_shadow_reduce_op_)

execute_shadow_allreduce(reduceOp):
    local_rank = rank_ % localDeviceCount_
    proxy_local_rank = (proxy_failed_local_dev_ + 1) % localDeviceCount_
    
    AVG workaround:
        use_avg_workaround = (reduceOp == ReduceOp::AVG)
        nccl_proxy_op = use_avg_workaround ? ncclSum : getNcclReduceOp(reduceOp, ...)

Faulty GPU  (local_rank == proxy_failed_local_dev_)
    ncclGroupStart()
        ncclSend(input -> proxy, NVLink)          Step 1
        ncclRecv(output <- proxy, NVLink)         Step 4
    ncclGroupEnd()

Proxy GPU  (local_rank == proxy_failed_local_dev_ + 1)
    proxy_buf = at::empty_like(input)
    ncclGroupStart()
        ncclRecv(proxy_buf <- faulty, NVLink)     Step 1
    ncclGroupEnd()
    output.copy_(input)
    output.add_(proxy_buf)                         Step 2: 預聚合
    ncclAllReduce(output, nccl_proxy_op,           Step 3: 跨節點
                  proxy_global_comm_)
    if use_avg_workaround:
        output.div_(size_)                         AVG 修正
    ncclGroupStart()
        ncclSend(output -> faulty, NVLink)         Step 4
    ncclGroupEnd()

Healthy GPU  (其他 rank)
    ncclAllReduce(input->output, nccl_proxy_op,
                  proxy_global_comm_)
    if use_avg_workaround:
        output.div_(size_)                         AVG 修正
```

---

## 6. 已解決的限制（本輪修正）

### ✅ 已修正：proxy_global_comm_ 使用 ncclCommInitRank（NIC 重掃崩潰）

**原問題：** `rebuild_shadow_ping_pong_topology()` 呼叫 `ncclCommInitRank`，
重新觸發 NIC 拓撲掃描，在每張 GPU 有專屬 NIC 的環境下崩潰。

**修正：** 改用 `NCCLFTComm::split`，`color=1`（健康 rank）和
`NCCL_SPLIT_NOCOLOR`（故障 rank），繼承已驗證的拓撲，不重掃 NIC。

### ✅ 已修正：ReduceOp 硬編碼為 SUM

**原問題：** 降級路徑中 `execute_shadow_allreduce` 呼叫時使用
`getNcclReduceOp(ReduceOp::SUM, ...)` 而非實際的 op，DDP 使用 AVG 時結果差
`world_size` 倍。

**修正：** `allreduce_impl` 在呼叫 `collective()` 前設定
`current_shadow_reduce_op_ = opts.reduceOp`，降級路徑讀取並傳入
`execute_shadow_allreduce`。

### ✅ 已修正：AVG 在 proxy_global_comm_ 的錯誤除數

**原問題：** `ncclAvg` 讓 NCCL 除以 `comm_size`，但 `proxy_global_comm_` 的
size 是 `size_ - nodes`，不是原始的 `size_`，導致 AVG 結果錯誤。

**修正：** 當 `reduceOp == AVG` 時，使用 `ncclSum` 進行跨節點通訊，然後在 CUDA
stream 上手動 `output.div_(size_)` 應用正確的分母。

### ✅ 已修正：2PC 不完整（final_commit_op_ 單方設定）

**原問題：** 側車執行緒在讀到 `NCCL_FT_EVENT` 後立即設定 `final_commit_op_`，
沒等待其他 rank 確認，不同 rank 可能在不同 OP 序號煞車導致 hang。

**修正：** 完整兩階段提交：每個 rank 寫 ACK key → rank 0 等所有 ACK 再寫
COMMIT key → 所有 rank 讀 COMMIT 後才設定 `final_commit_op_`。

---

## 7. 仍存在的限制

### P0 — 影響 ZeRO/FSDP 訓練

| 項目 | 原因 | 修正方向 |
|---|---|---|
| **AllGather 降級支援** | ZeRO/FSDP 使用 `_allgather_base`；NIC 壞後第一個 AllGather 就會再次失敗 | 實作 AllGather 的 Shadow Ping-Pong |
| **ReduceScatter 降級支援** | ZeRO/FSDP 使用 `_reduce_scatter_base`；同上 | 實作 ReduceScatter 的 Shadow Ping-Pong |

### P1 — 提升正確性與穩健性

| 項目 | 原因 | 修正方向 |
|---|---|---|
| **錯誤類型區分** | 嚴重 CUDA/NCCL 錯誤被靜默恢復，產生損壞的模型 | 在 Watchdog FT 路徑加入 NCCL 錯誤碼檢查，只對 `ncclRemoteError`、`ncclSystemError` 走容錯 |
| **多故障點支援** | 第二次故障後第一個故障 GPU 行為未定義 | 將 `proxy_failed_local_dev_` 改為 `std::unordered_set<int>` |

### P2 — 完整性補充

| 項目 | 說明 |
|---|---|
| **Broadcast 降級支援** | Broadcast 語意相對簡單，proxy 代為收發即可 |
| **TCPStore key 清理** | `NCCL_FT_ACK_<rank>_<op>` 和 `NCCL_FT_COMMIT_<op>` 累積，需定期清理 |
| **`ncclFaultCallback` 端到端驗證** | 確認客製化 NCCL 的 `ncclCommRegisterFaultCallback` 確實回調至 `nccl_ft_global_fault_callback` |

### P3 — 未來優化（超出 Prototype 範圍）

| 項目 | 說明 |
|---|---|
| **代理負載均衡** | 目前 proxy 固定為 `(failed+1) % localDeviceCount_`，可改為依頻寬動態選擇 |
| **Tensor 分片均攤** | 將 faulty GPU 的 tensor 切分給多個健康 GPU 代傳，充分利用所有健康 NIC 頻寬 |
