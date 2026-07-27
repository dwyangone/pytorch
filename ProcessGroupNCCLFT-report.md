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
│  監聽 NCCL_FT_EVENT -> 設定 do_not_cross_op_ / failed_dev   │
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
| `local_nvlink_comm_` | `ncclComm_t` | 節點內 NVLink-only communicator；初始化一次、永不依賴跨節點 NIC |
| `proxy_global_comm_` | `ncclComm_t` | 排除故障裝置後的縮減跨節點 communicator |
| `proxy_comm_ready_` | `atomic<bool>` | 保護 `proxy_global_comm_` 初始化競態；`false` 時降級路徑退回 native |
| `proxy_failed_local_dev_` | `int` | 已確認故障的 local device index（供 execute_shadow_allreduce 用） |
| `proxy_comm_rank_/size_` | `int` | 此 rank 在 `proxy_global_comm_` 中的新 rank/size |

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
|        └─ initLocalNvlinkComm()  ->  建立 local_nvlink_comm_
|
|-- 5. (ENABLE_NCCL_ERROR_CHECKING) watchdog_->start()
|-- 6. init()（正常 NCCL communicator 懶初始化）
└─ 7. 記錄環境設定 LOG
```

**重點**：FT 控制面（側車執行緒 + NVLink comm）在 Watchdog 啟動之前、正常 NCCL init 之前就已就緒，確保當第一個 collective 執行時所有 FT 基礎設施都已準備好。

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

### Phase 2：TCPStore 協商（~50ms，在側車執行緒）

```
ft_negotiator_thread_（pt_nccl_ft_side）

任務 1：代發提案
  local_hardware_fault_dev_ != -1
  |-- target_op = seqCollective_ + 10
  |-- proposal = "PROPOSE:<target_op>:<node_id>:<dev_idx>"
  |-- globalStore_->set("NCCL_FT_EVENT", vec)   承擔 TCPStore 延遲
  └─ local_hardware_fault_dev_.store(-1)         復位信號

任務 2：監聽全網提案（所有 rank 的側車執行緒都在做）
  globalStore_->check({"NCCL_FT_EVENT"}) == true
  |-- 解析 "PROPOSE:<target_op>:<node>:<dev>"
  |-- do_not_cross_op_.store(target_op)    設定主執行緒警戒線
  |-- failed_dev_index_.store(f_dev)
  |-- final_commit_op_.store(target_op)    quorum 確認
  └─ sleep(200ms)                          防抖動
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
if proxy_global_comm_ != nullptr:
    ncclCommDestroy(proxy_global_comm_)  銷毀舊 comm，防資源洩漏

計算 rank 映射：
  proxy_comm_size_ = size_ - (size_ / localDeviceCount_)
  遍歷所有 rank，跳過 local_rank == failed_dev，重新編號

UID 交換（key 含 seqCollective_ 後綴，防止不同 rebuild 輪衝突）：
  rank==0: ncclGetUniqueId() -> globalStore_.set("NCCL_FT_PROXY_UID_<seq>")
  rank!=0: globalStore_.get("NCCL_FT_PROXY_UID_<seq>")

if !is_faulty:
    ncclCommInitRank(&proxy_global_comm_, proxy_comm_size_, uid, proxy_comm_rank_)
else:
    proxy_global_comm_ = nullptr   故障 GPU 不參與跨節點通訊

proxy_comm_ready_.store(true)           通知 execute_shadow_allreduce 可用
```

### Phase 5：降級後的 AllReduce

```
local_rank = rank_ % localDeviceCount_
proxy_local_rank = (proxy_failed_local_dev_ + 1) % localDeviceCount_

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
  output.add_(proxy_buf)                      Step 2: 預聚合
  ncclAllReduce(output, proxy_global_comm_)   Step 3: 跨節點
  ncclGroupStart()
    ncclSend(output -> faulty, NVLink)        Step 4
  ncclGroupEnd()

Healthy GPU  (其他 rank)
  ncclAllReduce(input->output, proxy_global_comm_)
```

---

## 6. 目前實作的限制

### 限制 1：只支援 AllReduce 的容錯降級

`is_degraded_` 分支只有 `opType == ALLREDUCE` 才走 Shadow Ping-Pong；AllGather、ReduceScatter、Broadcast 等 fall back 到原始 comm，若 NIC 已壞則再次失敗或 hang。影響所有使用 ZeRO stage 2/3 或 FSDP 的訓練。

### 限制 2：Reduce Op 硬編碼為 SUM

`collective()` 的降級分支呼叫 `getNcclReduceOp(ReduceOp::SUM, ...)` 而非從原始 lambda 取得實際 op。若訓練使用 `ReduceOp::AVG`（新版 PyTorch DDP 的預設值），結果數值差 `world_size` 倍，**會產生靜默的數值錯誤**。

### 限制 3：只支援單一故障點

`proxy_failed_local_dev_` 只記錄一個故障裝置。第二次故障後，第一次故障的 GPU 不再被識別為 `is_faulty`，會嘗試使用已壞掉的 NIC 參與 `proxy_global_comm_` 的 AllReduce，行為未定義（hang 或再次引發錯誤）。

### 限制 4：2PC 協商不完整

`final_commit_op_` 由偵測到 `NCCL_FT_EVENT` 的側車執行緒**單方面設定**，沒有等待其他 rank 的確認回應。若某 rank 因網路延遲較晚讀到 `NCCL_FT_EVENT`，不同 rank 可能在不同 OP 序號煞車，造成訓練迴圈 hang。

### 限制 5：所有 NCCL 錯誤一律視為可恢復

Watchdog FT 路徑對所有 NCCL 錯誤都視為 NIC 故障。GPU 記憶體錯誤、CUDA context 損壞、`ncclInternalError` 等嚴重錯誤會被錯誤地觸發容錯恢復，讓訓練在損壞狀態下繼續，產生靜默的錯誤模型輸出。

### 限制 6：`initLocalNvlinkComm` 可能與 ncclGroupStart 衝突

`initLocalNvlinkComm()` 在建構子裡呼叫，若此時其他 PG 正在進行 `ncclGroupStart()` 的群組操作，NCCL 不允許在群組操作期間建立新 communicator，可能失敗或 hang。

### 限制 7：TCPStore key 無版本化（local NVLink comm）

`initLocalNvlinkComm` 使用固定 key `"NCCL_FT_LOCAL_UID_NODE_<N>"`。若 PG 被銷毀後重建（elastic training），舊 key 仍存在於 TCPStore，非 rank-0 的 rank 可能讀到舊 UID，導致 `ncclCommInitRank` 建立錯誤配對的 communicator。

---

## 7. 尚未完成的項目

### P0 — 必須修正才能讓 Prototype 產生正確結果

| 項目 | 原因 | 修正方向 |
|---|---|---|
| **修正 ReduceOp 傳遞** | 硬編碼 SUM 導致使用 AVG 的 DDP 訓練數值錯誤（差 world_size 倍） | 在 `allreduce_impl` 層面新增降級路徑直接傳入 `opts.reduceOp`，或將 `execute_shadow_allreduce` 的簽名改為接受 `AllreduceOptions` |
| **AllGather 降級支援** | ZeRO/FSDP 使用 `_allgather_base`；NIC 壞後第一個 AllGather 就會再次失敗 | 實作 AllGather 的 Shadow Ping-Pong：faulty GPU 透過 NVLink 把 input 送給 proxy，proxy 代為 AllGather，再把對應 slice 傳回 |
| **ReduceScatter 降級支援** | ZeRO/FSDP 使用 `_reduce_scatter_base`；同上 | 實作 ReduceScatter 的 Shadow Ping-Pong：proxy 接收、參與 ReduceScatter、分發 output slice 回 faulty GPU |

### P1 — 提升正確性與穩健性

| 項目 | 原因 | 修正方向 |
|---|---|---|
| **真正的 2PC 確認** | 目前缺 phase-2；不同 rank 可能在不同 OP 煞車導致 hang | 在 TCPStore 加入 `"NCCL_FT_ACK_<rank>"` key；側車執行緒在設定 `final_commit_op_` 前先等待所有 rank 的 ACK |
| **錯誤類型區分** | 嚴重 CUDA/NCCL 錯誤被靜默恢復，產生損壞的模型 | 在 Watchdog FT 路徑加入 NCCL 錯誤碼檢查，只對 `ncclRemoteError`、`ncclSystemError` 走容錯；`ncclInternalError` 等保留原始 abort |
| **多故障點支援** | 第二次故障後第一個故障 GPU 行為未定義 | 將 `proxy_failed_local_dev_` 改為 `std::unordered_set<int> failed_local_devs_`；rank mapping 排除所有故障 device；`execute_shadow_allreduce` 的 role 判斷改為 set 查找 |

### P2 — 完整性補充

| 項目 | 說明 |
|---|---|
| **Broadcast 降級支援** | Broadcast 語意相對簡單，proxy 代為收發即可 |
| **`ncclFaultCallback` 端到端驗證** | 確認客製化 NCCL 的 `ncclCommRegisterFaultCallback` 確實回調至 `nccl_ft_global_fault_callback` |
| **`local_nvlink_comm_` key 版本化** | 將 key 改為 `"NCCL_FT_LOCAL_UID_NODE_<N>_<pg_uid_>"` 避免重建時衝突 |
| **`initLocalNvlinkComm` 時機保護** | 加入 `ncclActiveGroupCounter_ == 0` 斷言或移至 `init()` 之後執行 |
| **TCPStore key 清理機制** | 多次 rebuild 累積的 `NCCL_FT_PROXY_UID_<seq>` key 需要定期清理 |

### P3 — 未來優化（超出 Prototype 範圍）

| 項目 | 說明 |
|---|---|
| **代理負載均衡** | 目前 proxy 固定為 `(failed+1) % localDeviceCount_`，可改為依頻寬動態選擇 |
| **Tensor 分片均攤** | 將 faulty GPU 的 tensor 切分給多個健康 GPU 代傳，充分利用所有健康 NIC 頻寬 |
