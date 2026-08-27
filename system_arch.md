# ProcessGroupNCCLFT — System Architecture

> 依據 `ProcessGroupNCCLFT.cpp` / `.hpp` 目前實際程式碼撰寫，2026-08（第六次修訂）。

---

## 一、元件概覽

`ProcessGroupNCCLFT` 是基於 `ProcessGroupNCCL` 複製並擴充的自訂 NCCL 容錯 backend，
在 c10d 中以名稱 `nccl_ft` 登記。設計目標：NIC 硬體故障時，訓練**不重啟、不拋出
Python 例外**，梯度數學結果與無故障版本相同。

### 硬體假設

| 維度 | 值 |
|------|----|
| 伺服器數量 | 2 台（node 0 = rank 0-7，node 1 = rank 8-15） |
| 每台 GPU 數量 | 8 張（localDeviceCount_ = 8） |
| NIC 拓撲 | 每張 GPU 對應一張 NUMA-local NIC（mlx5_0 ～ mlx5_7） |
| 節點內互連 | NVLink |
| 啟動方式 | `torchrun --nproc_per_node=8 --nnodes=2` |

---

## 二、執行緒模型（三執行緒）

### 2.1 主執行緒（訓練主迴圈）

執行 DDP backward → 呼叫 `allreduce()` → `allreduce_impl()` → `collective()`，
然後呼叫 `work->wait()` 等待完成。

`wait()` 的 FT 攔截邏輯（行 915-973）採用 **Future 機制**，一次性判斷是否需要 FT 等待：

**進入 FT 安全等待區的條件（二選一）：**
```cpp
bool is_ft_managed = (pg_->in_flight_shadow_bufs_.count(this->seq_) > 0);
bool is_ft_exception = false;
if (exception()) {
    try { std::rethrow_exception(exception()); }
    catch (const ::c10::NCCLFaultToleranceError&) { is_ft_exception = true; }
    catch (...) {}
}
if (is_ft_exception || (is_ft_managed && pg_->is_degraded_.load(acquire))) {
    // ← FT 安全等待區
}
```

**FT 安全等待區（行 936-972）：**
1. `future_->wait()` — 不耗 CPU 地等待側車的 `recover_and_replay` 呼叫 `markCompleted`
2. `setException(nullptr)` — 抹除例外，保護 Python
3. 從 `in_flight_shadow_bufs_` 取出 ShadowContext
4. `ctx.replayed_end_event->block(currentStream)` — 等待 replay 完成
5. `stashed_for_allocator_safety_->unstash()` — 釋放扣留的 Tensor 參照
6. **Inline GC**：`original_input/output = Tensor()`，`work_ptr.reset()`，歸還到 `free_shadow_bufs_`
7. `return true` — 對 DDP 完全透明

**設計要點：**
- `future_->wait()` 是阻塞呼叫，直到側車的 `recover_and_replay_inflight_ops()` 對應 Future 呼叫 `markCompleted()` 才解除
- 側車自行執行重播（不依賴主執行緒），所以主執行緒等 Future 必然會被喚醒
- 無 spin-loop，無 final_commit_op_ 輪詢，完全依賴 Future 機制

### 2.2 Watchdog 執行緒（`pt_nccl_watchdg`）

持續掃描 `workMetaList_`，對每個 work 執行三條路徑：

**路徑 A：FT early-abort**
```cpp
if (!ft_disabled_ && !work.exception() && commit_signal > 0)
    work.setException(NCCLFaultToleranceError, "FT early-abort: Comm dead.")
```
- 當 2PC 已達成共識（`final_commit_op_ > 0`），對尚無 exception 的飛行中任務主動注射 FT 例外
- 讓後續 wait() 知道需要走 FT 安全路徑

**路徑 B：FT clearing path**
```cpp
if (!ft_disabled_) {
    pg_->error_ = ErrorType::SUCCESS       // 清除 COMM_ERROR
    work.ncclEndEvent_->record(ncclStream) // 確保 block() 能解鎖
    // 若 stash 非空：push_back 到 shelvesToUnstash_（保底 GC）
    erase from workMetaList_
}
```
注意：此路徑**不做 shadow buffer GC**（保留給 wait() 使用），stash 保底送給 shelvesToUnstash_

**路徑 C：正常完成 GC**
```cpp
if (work.isCompleted() && work.opType_ == OpType::ALLREDUCE && !ft_disabled_)
```
- 從 `in_flight_shadow_bufs_` 找對應 seq
- 清空 `original_input` / `original_output` GPU 參照
- 歸還 ShadowContext 到 `free_shadow_bufs_`（free pool）

### 2.3 Side-car 執行緒（`pt_nccl_ft_side`）

透過 TCPStore 執行 2PC 協商，由 `start_ft_negotiator_thread()` 啟動。

**2PC 完成後側車直接啟動重播（新設計）：**
```cpp
// 設定 final_commit_op_ 通知所有人
this->final_commit_op_.store(cur_round + 1, std::memory_order_release);
// 側車自己呼叫重播中心，不依賴主執行緒！
this->recover_and_replay_inflight_ops();
// recover 內部自行設 final_commit_op_ = 0
```
側車完成後直接進入下一輪監控，不等待主執行緒。

---

## 三、NCCL 與 ProcessGroupNCCLFT 的互動

### 3.1 標準 collective 路徑（正常情況）

```
Python DDP
  └─ allreduce(tensors, opts)              [行 6053]
       └─ allreduce_impl(tensor, opts)     [行 5975]
            ├─ current_shadow_reduce_op_ = opts.reduceOp   [記錄 op，行 5981]
            └─ collective(tensor, tensor, fn, shadow_pre, post_noop, ALLREDUCE)
                 ├─ future_ 初始化（pre() 前）              [行 5117-5124]
                 ├─ [shadow_pre lambda]    D2H checkpoint   [行 5996-6025]
                 ├─ collective() 正常分支 [行 5165-5175]
                 │    └─ C10D_NCCL_FT_CHECK_TIMEOUT(ncclAllReduce, ...)
                 └─ work->ncclEndEvent_->record(ncclStream) [行 5208-5210]
```

**NCCL API 呼叫鏈（正常）：**
1. `ncclAllReduce(input, output, numel, dataType, reduceOp, comm, stream)` — 非同步排入 ncclStream
2. `ncclEndEvent_->record(ncclStream)` — 排入 EndEvent，DDP 的 `wait()` 透過 `block()` 等此 event

### 3.2 shadow_pre lambda（D2H Checkpoint，行 5996-6025）

每個 AllReduce 呼叫**在 ncclAllReduce 啟動前**執行 shadow_pre：

1. 呼叫 `get_or_allocate_shadow_context(tensor)` — 從 free pool 取或新建 ShadowContext
2. 設定 `ctx.original_input = ctx.original_output = tensor`（in-place AllReduce）
3. 設定 `ctx.reduce_op = opts.reduceOp`
4. 設定 `ctx.work_ptr = work`（綁定 Work 指標，供 Inline GC 使用）
5. 鎖內插入 `in_flight_shadow_bufs_[seq] = ctx`
6. 在 shadow_copy_stream_ 上：
   - `ctx.compute_event->record(compute_stream)`，block shadow_copy_stream_ 等 compute 完成
   - `ctx.buffer.narrow(0, 0, tensor.numel()).copy_(tensor.flatten(), non_blocking=true)` — D2H copy
   - record `ctx.copy_event`（精準標記此 seq 的 D2H 完成時間點）
7. `shadow_seq_.store(current_seq, release)` — 原子寫入，供 trigger_fault_proposal 快照用

**關鍵差異（vs 舊文件）：** `shadow_seq_` 已改為 `std::atomic<uint64_t>`，用 store(release) 寫入；`compute_done` event 已提升為 `ShadowContext::compute_event`，不再在 lambda 內每次 create。

### 3.3 NCCL Fault Callback 路徑（NIC 故障時）

```
[NCCL IB transport layer]
  ncclIbResiliencyHandleDeviceFailure()   → return ncclRemoteError (NCCL fork)
  ncclIbResiliencyProbeHandleCompletionEvent() → return ncclRemoteError (NCCL fork)
       ↓
  nccl_ft_trigger_fault(dev_idx)         → 呼叫已註冊的 callback
       ↓
  nccl_ft_global_fault_callback(dev_idx) → 呼叫 pg->trigger_fault_proposal(dev_idx)
       ↓
  trigger_fault_proposal(dev_idx)         [行 4520]
    ├─ local_hardware_fault_mask_.fetch_or(1ULL << dev_idx, release)
    └─ shadow_seq_.load(acquire) → pending_shadow_seq_.CAS(UINT64_MAX, shadow_seq)
```

`trigger_fault_proposal` 必須在微秒內返回（NCCL progress thread 呼叫），只做 atomic write。

### 3.4 NCCL Comm 管理 API

| API | 用途 | 呼叫時機 |
|-----|------|---------|
| `ncclCommRegisterFaultCallback(comm, cb)` | 每 rank 各自在自己的 comm 上註冊 fault callback | 第一個 collective 的 lazy-init |
| `ncclGetUniqueId(&id)` | 生成新 rendezvous ID | rebuild 時 proxy_comm_rank_==0 的 rank 呼叫；initLocalNvlinkComm 時 local_rank==0 呼叫 |
| `NCCLFTComm::create(size, rank, id, device, config)` | 從頭建立全新 communicator（完整 topo discovery） | local_nvlink_comm_ 初始化 + proxy_global_comm_ rebuild |
| `ncclAllReduce(...)` | 跨節點 AllReduce | 正常路徑 + 降級路徑（proxy/healthy 角色） |
| `ncclSend / ncclRecv` | NVLink 點對點傳輸 | execute_shadow_allreduce（faulty↔proxy） |
| `globalComm->abort(reason)` | 強制中止 communicator | 2PC committed 後 side-car 呼叫（喚醒主執行緒）+ recover_and_replay_inflight_ops |

### 3.5 ncclRemoteError 傳播機制

AllReduce 是集體操作，一個 NIC 故障 → 所有 16 個 rank 的 comm 都會回傳 `ncclRemoteError`。
因此 Watchdog 看到 exception 時，`device_idx = work.device_.index()` 是**每個 rank 自己的 GPU 索引**，
不是真正故障的 NIC。這就是為何 `fault_mask` 只由 `trigger_fault_proposal`（NCCL callback，
接收真正的 `dev_idx`）寫入，Watchdog **不寫** fault_mask（Fix A）。

### 3.6 IntraNodeComm 快速路徑（降級時跳過）

`allreduce()`（行 6114）：
```cpp
if (opts.reduceOp == ReduceOp::SUM && !is_degraded_) {
    // 嘗試 IntraNodeComm fast-path（NVLink-only, 繞過 NCCL）
}
```
降級後強制走 `allreduce_impl`，確保 shadow buffer checkpoint 和 execute_shadow_allreduce 正確執行（Bug 6 fix）。

---

## 四、通訊器管理

### 4.1 三種通訊器

| 通訊器 | 類型 | 覆蓋範圍 | 用途 |
|--------|------|---------|------|
| `devNCCLCommMap_[key]`（global comm） | `NCCLFTComm` | 所有 16 ranks | 正常訓練 AllReduce |
| `local_nvlink_comm_` | `NCCLFTComm` | 同節點 8 ranks（本機） | NVLink 傳輸，faulty↔proxy ping-pong |
| `proxy_global_comm_` | `NCCLFTComm` | 健康的 ranks（排除 faulty） | 降級後跨節點 AllReduce |

另外 `ft_root_comm_` 是第一個 collective 建立的全局 comm 的快照，用於記錄 fault callback 已註冊的 comm。

### 4.2 通訊器建立流程

**global comm（`devNCCLCommMap_`）：**
- 由 `initNCCLComm()` 在第一個 collective 建立

**local_nvlink_comm_（行 4043-4160）：**
- Lazy init，在第一個 collective 執行（行 5159）；recovery 路徑由 `rebuild_shadow_ping_pong_topology()` 再次呼叫
- 使用 `NCCLFTComm::create` 完整從頭建立（**不用 ncclCommSplit**，因為 global comm 可能在第一個 collective 期間就死掉）
- 呼叫者傳入 `attempt`，lazy-init 傳 0，recovery 傳 `ft_round_+1`，確保每次呼叫使用獨立的 TCPStore key 集合，消除 lazy-init 殘留 key 干擾 recovery
- **三屏障設計（行 4058-4132）：**
  1. **Entry-arrival barrier**（Phase 0）：所有 local rank 先寫 `NCCL_FT_LOCAL_ARRIVE_NODE_<n>_PG_<uid>_ATT_<attempt>_LR_<lr>`，再 wait 全部 local rank 到達；確保 local_rank=0 不會提前生成 ID 並進入 bootstrap（120 秒超時）
  2. **ID 生成**（Phase 1）：local_rank=0 生成 `ncclUniqueId`，寫入 `NCCL_FT_LOCAL_COMM_ID_NODE_<n>_PG_<uid>_ATT_<attempt>`；其他 local rank wait 30 秒後讀取
  3. **All-ready barrier**（Phase 2）：每個 local rank 讀取 ID 後先寫 ready key（`NCCL_FT_LOCAL_READY_..._LR_<lr>`），再 wait 所有 ready key，確保全部 local rank 同時進入 `NCCLFTComm::create`（60 秒超時）
- all-ready barrier 通過後，local_rank=0 立即 `deleteKey(local_id_key)` 清除 Store 殘留

**proxy_global_comm_（行 4346-4543）：**
- 在 `rebuild_shadow_ping_pong_topology()` 中建立
- 只有健康 rank 才建立（faulty rank 設 `proxy_global_comm_ = nullptr`）
- 流程（**環境變數重置方案**）：
  1. **Data channel**：從 `faulty_devs` 計算健康 HCA 列表（`mlx5_1,mlx5_2,...`），`setenv("NCCL_IB_HCA", healthy_hcas, 1)` + `nccl_ft_reset_ib_cache()`，讓下次 `ncclIbInitDevices()` 重新掃描並只使用健康 NIC。vNic 索引從 0 重新編號（0-6 對應 mlx5_1-mlx5_7），NCCL 用 modulo 分配 GPU→NIC，資料仍能正確到達目標 rank
  2. **Control channel**：掃描 `getifaddrs()` 找第一個 UP/非 loopback/非 IB 的 Ethernet 介面（fallback 到 `lo`），`setenv("NCCL_SOCKET_IFNAME", chosen_if, 1)` + `nccl_ft_reset_bootstrap_net()` + `nccl_ft_reset_net_socket()`，確保 bootstrap TCP socket 和 Socket transport singleton 都不再走死亡 IB NIC
  3. Pre-rebuild global barrier（key: `NCCL_FT_REBUILD_SYNC_ATT_<attempt>_R<rank>`），確保所有 16 個 rank 同步到達
  4. proxy_comm_rank_ == 0 生成新 `ncclUniqueId`，寫入 TCPStore（key: `NCCL_FT_PROXY_ID_ATT_<rebuild_attempt_>`）
  5. All-ready barrier（key: `NCCL_FT_PROXY_READY_ATT_<attempt>_<rank>`），健康 rank 間同步
  6. `NCCLFTComm::create(proxy_comm_size_, proxy_comm_rank_, proxyId, device.index(), config)`
  7. **零元素 AllReduce 驗證**：建立後立即在 ncclStream 上執行 `ncclAllReduce(nullptr, nullptr, 0, ...)` + `cudaStreamSynchronize`，確認 IB transport 穩定後才標記 `proxy_comm_ready_=true`

### 4.3 proxy_comm 大小與排名計算（行 4194-4209）

```
proxy_comm_size_ = size_ - num_faulty_per_node * num_nodes
  (對稱降級：每個節點都去掉相同的 local rank 索引)

proxy_comm_rank_：重新對健康 rank 編號（跳過 faulty_devs）
```

### 4.4 comms 的 abort 時機

| 情況 | abort 呼叫者 | 原因 |
|------|-------------|------|
| 2PC committed | side-car thread | 強制喚醒卡在 ncclAllReduce 的主執行緒 |
| recover_and_replay_inflight_ops() 開始 | 側車（recover 在側車內執行） | 確保舊 global comm 徹底死亡再 rebuild |
| rebuild 開始前（舊 local/proxy comm） | `rebuild_shadow_ping_pong_topology()` 同步 GC | 用 `std::move` 取走舊 comm shared_ptr，立即呼叫 `abort()` 再讓 destructor 執行（**不再使用背景 GC 執行緒**）；原因：背景 GC 執行緒與 `ncclCommInitRankConfig` 之間的 race 會造成 `ncclInvalidUsage` |

---

## 五、Shadow Ping-Pong 容錯機制

### 5.1 ShadowContext 結構（hpp 行 1178-1187）

```cpp
struct ShadowContext {
    at::Tensor buffer;                                         // pinned CPU memory（D2H 備份）
    std::shared_ptr<at::cuda::CUDAEvent> copy_event;          // per-seq D2H 完成事件
    std::shared_ptr<at::cuda::CUDAEvent> replayed_end_event;  // replay kernel 完成事件
    std::shared_ptr<at::cuda::CUDAEvent> compute_event;       // compute → shadow_copy stream 同步
    c10::intrusive_ptr<WorkNCCLFT> work_ptr;                  // 綁定 Work（Inline GC 使用）
    at::Tensor original_input;                                 // GPU tensor 參照（in-place = output）
    at::Tensor original_output;                                // GPU tensor 參照
    c10d::ReduceOp reduce_op;                                  // replay 使用的 ReduceOp
};
```

**新增欄位（vs 舊文件）：**
- `compute_event`：替代原來在 shadow_pre lambda 內每次建立的 `compute_done` event，提升為 ShadowContext 成員複用
- `work_ptr`：綁定原生 Work，供 `get_or_allocate_shadow_context` 的 **Inline GC** 主動回收成功完成的 bucket

**Shadow Buffer Pool 管理（`free_shadow_bufs_` + `in_flight_shadow_bufs_`）：**

```
allreduce() 呼叫時：
  get_or_allocate_shadow_context(tensor)
    Phase 1 (鎖內):
      A. Inline GC：掃描 in_flight_shadow_bufs_，主動回收已成功的 bucket 到 free pool
      B. Best-Fit 搜尋：找 numel >= t.numel() 且差距最小的 buffer 複用
      C. Memory Backpressure：in-flight 總量 > 2GB 時 sleep 等待，防止 OOM
    Phase 2 (鎖外，若 Phase 1B 未命中):
      執行昂貴的 pin_memory() + cudaEventCreate
  → 插入 in_flight_shadow_bufs_[seq]

Inline GC（get_or_allocate_shadow_context 呼叫時主動）：
  work_ptr->finishedGPUExecutionInternal() && !work_ptr->exception()
  → 回收至 free_shadow_bufs_

Watchdog 正常 GC 路徑（isCompleted()）：
  in_flight_shadow_bufs_[seq]
  → ctx.original_input = Tensor(), ctx.original_output = Tensor()
  → free_shadow_bufs_.push_back(ctx)
  → in_flight_shadow_bufs_.erase(it)

wait() FT 後處理 GC（is_degraded_ && seq <= committed_ss）：
  相同的 GC 流程，由 wait() 執行
```

`get_or_allocate_shadow_context` 採用**三段式設計**（升級自舊版兩段式）：
- 鎖內 Inline GC → Best-Fit 搜尋 → Memory Backpressure（皆在鎖內）
- 鎖外：執行昂貴的 `pin_memory()` + `cudaEventCreate`

### 5.2 execute_shadow_allreduce 四角色邏輯（行 4326-4515）

根據 `local_rank = rank_ % localDeviceCount_` 與 `faulty_local_devs_` 決定角色。

**角色分配函式 `findProxy(faulty_dev, localDeviceCount, faulty_devs)`（行 4315-4324）：**
- 從 `(faulty_dev+1) % N` 開始走，找第一個不在 faulty set 中的 local rank
- 保證多 NIC 故障時仍能找到 proxy（要求 faulty_devs.size() < localDeviceCount_）

**FAULTY 角色（本地 NIC 故障）：**
```
Step 1: ncclSend(input → my_proxy, nvlink_comm)
Step 4: ncclRecv(result ← my_proxy, nvlink_comm)
（Steps 1+4 在同一個 ncclGroup 內）
```

**PROXY 角色（為一個或多個 faulty rank 代勞）：**
```
Step 1: ncclRecv 所有 ward 的 tensor（一個 ncclGroup）
Step 2: output = input + sum(ward_bufs)   (setCurrentCUDAStream + copy_ + add_)
Step 3: ncclAllReduce(output, proxy_global_comm_)
        若 reduceOp=AVG → 改用 ncclSum + div(size_)
Step 4: ncclSend 結果回所有 wards（一個 ncclGroup）
```

**HEALTHY 角色（普通健康 rank，不是任何人的 proxy）：**
```
ncclAllReduce(input, output, proxy_global_comm_)
若 reduceOp=AVG → div(size_) (AVG workaround)
```

**ReduceOp::AVG 特殊處理：**
proxy_global_comm_ 的 size 是 `proxy_comm_size_`（非原始 size_），
如果用 ncclAvg 除數會錯誤，改為 ncclSum + `output.div_(size_)`。

**ATen 算術串流保證（Bug 7 fix，行 4461-4468）：**
proxy 的 `output.copy_()` 和 `output.add_()` 必須在和 ncclRecv 相同的 stream 執行，
避免讀到未完成的 recv data。使用 `setCurrentCUDAStream(stream)` + restore。

### 5.3 2PC 協商流程（side-car thread）

#### Phase 1：PROPOSE（各 rank 寫自己的 key）

**Key 格式：** `NCCL_FT_PROPOSE_<rank>_<round>`

**Value 格式：** `PROPOSE:<round>:<node_id>:0x<fault_mask_hex>:<shadow_seq>`

- 有故障的 rank：`fault_mask` = 本機 NIC 故障 bitmask
- 無故障的 rank：`fault_mask` = `0x0`（佔位用，確保 coordinator 計數正確）
- `shadow_seq`：從 `pending_shadow_seq_` 讀取；若為 `UINT64_MAX` 表示尚無 checkpoint

**De-duplication：** 用 `last_proposed_round` 防止同一 round 寫兩次（callback 可能 fire 兩次）。

**any_proposed 快速路徑：** 每輪迴圈先掃描是否有任何 rank 已 propose，若無則 sleep 50ms 直接 continue，避免空轉。

#### Phase 2：COMMIT（rank 0 協調）

Coordinator（rank 0）等待所有 size_ 個 PROPOSE key 後：
- 聚合 `agg_fault_mask`（OR 所有 fault_mask）
- 計算 `agreed_ss`（min strategy，忽略 UINT64_MAX sentinel）
- 寫 `NCCL_FT_SS_AGREED_<round>` = agreed_ss
- 寫 `NCCL_FT_COMMIT_<round>` = "1"

#### Phase 3：Commit 後動作（所有 rank）

1. 讀取 `NCCL_FT_SS_AGREED_<round>` → `committed_shadow_seq_`（release）
2. 將 `agg_fault_mask` 解展為 `faulty_local_devs_`（OR 更新，支援逐步故障累加）
3. 強制 abort globalComm（喚醒卡死的主執行緒）
4. `final_commit_op_.store(cur_round + 1, release)` — 通知所有人
5. **直接呼叫 `recover_and_replay_inflight_ops()`**（側車自己執行，不依賴主執行緒）
6. `recover_and_replay` 末尾自行設 `final_commit_op_ = 0`，側車無需再等

---

## 六、恢復流程：recover_and_replay_inflight_ops()（行 5297-5395）

### 6.1 重入防護

```cpp
std::lock_guard<std::mutex> lock(recovery_mutex_); // 只有一個 thread 進入
uint64_t commit_signal = final_commit_op_.load(acquire);
if (commit_signal == 0) return;   // 已被其他 thread 恢復過了
```

`recovery_mutex_` 確保同一時間只有一個執行緒執行恢復（主執行緒或側車執行緒皆可）。

### 6.2 恢復步驟

```
Step 1: globalComm->abort("FT Recovery")   [確保舊 comm 死亡]

Step 2: rebuild_shadow_ping_pong_topology()
          └─ 使用最新的 faulty_local_devs_（2PC 已聚合所有節點的故障資訊）
             ├─ 同步 GC：std::move 舊 local/proxy comm → abort() → destructor
             ├─ [Data] setenv("NCCL_IB_HCA", "mlx5_1,...") + nccl_ft_reset_ib_cache()
             ├─ [Ctrl] setenv("NCCL_SOCKET_IFNAME", <enp*|lo>) + nccl_ft_reset_bootstrap_net()
             │          + nccl_ft_reset_net_socket()
             ├─ Pre-rebuild global barrier (NCCL_FT_REBUILD_SYNC_ATT_<att>_R<r>)
             ├─ initLocalNvlinkComm(attempt=rebuild_attempt_)
             ├─ 健康 rank：
             │    NCCLFTComm::create(proxy_comm_size_, proxy_comm_rank_, ...)
             │    零元素 AllReduce 驗證 (cudaStreamSynchronize 後檢查 async error)
             └─ proxy_comm_ready_ = true
        is_degraded_ = true

Step 3: 收集 seqs_to_replay
          agreed_ss = committed_shadow_seq_.load()
          from in_flight_shadow_bufs_: collect seq >= agreed_ss
          sort ascending

Step 4: 依序 H2D restore + replay（迴圈外宣告一個 restore_done CUDAEvent 複用）
          for seq in seqs_to_replay:
            ctx.copy_event->synchronize()                [等 D2H 備份完成]
            on shadow_copy_stream_:
              ctx.original_input.flatten().copy_(ctx.buffer.narrow(0,0,numel), true)
            restore_done.record(shadow_copy_stream_)
            restore_done.block(ncclStream)
            execute_shadow_allreduce(ctx.original_input, ctx.original_output,
                                     ncclStream, ctx.reduce_op)
            ctx.replayed_end_event->record(ncclStream)
            ctx.work_ptr->setException(nullptr)          [抹除例外]
            ctx.work_ptr->future_->markCompleted(...)    [補齊 Future，解鎖 DDP]

Step 5: ft_round_++         [Bug A fix：讓 side-car 知道可以進入下一輪]
        TCPStore cleanup:
          每個 rank（每個 attempt att=1..cur_attempt）:
            deleteKey("NCCL_FT_PROPOSE_<rank>_<round>")
            deleteKey("NCCL_FT_REBUILD_SYNC_ATT_<att>_R<rank>")
            deleteKey("NCCL_FT_LOCAL_ARRIVE_NODE_<node>_PG_<uid>_ATT_<att>_LR_<local_rank>")
            deleteKey("NCCL_FT_LOCAL_READY_NODE_<node>_PG_<uid>_ATT_<att>_LR_<local_rank>")
            deleteKey("NCCL_FT_LOCAL_COMM_ERR_NODE_<node>_PG_<uid>_ATT_<att>")   [非必寫，但 delete 無害]
            if !is_faulty_rank: deleteKey("NCCL_FT_PROXY_READY_ATT_<att>_<rank>")
          if cur_round==0（lazy-init attempt=0 的殘留）:
            deleteKey("NCCL_FT_LOCAL_ARRIVE_NODE_<node>_PG_<uid>_ATT_0_LR_<local_rank>")
            deleteKey("NCCL_FT_LOCAL_READY_NODE_<node>_PG_<uid>_ATT_0_LR_<local_rank>")
          if rank==0（每個 attempt att=1..cur_attempt）:
            deleteKey("NCCL_FT_COMMIT_<round>")
            deleteKey("NCCL_FT_SS_AGREED_<round>")
            deleteKey("NCCL_FT_PROXY_ID_ATT_<att>")

Step 6: final_commit_op_.store(0, release)
          [解除 wait() 的自旋，允許下一輪 2PC 開始]
```

**新增（vs 舊文件）：** Step 4 末尾直接呼叫 `work_ptr->setException(nullptr)` 和 `markCompleted`，在 recover 內部就解鎖 DDP 的 Future，不再依賴 wait() 攔截點 2 才做。

### 6.3 `agreed_ss` 的正確語義

`shadow_seq_` 在 `shadow_pre` 中設定（D2H copy 已排入但 ncclAllReduce 尚未確認）。
replay 邊界條件是 `seq >= agreed_ss`（含 agreed_ss 本身），
因為 agreed_ss 那個 bucket 的跨節點 AllReduce 狀態未知，必須重播。

---

## 七、collective() 中的執行路徑（二分支設計）

### 7.1 重要結構改變：Future 提前初始化

```cpp
// Future 在 pre() 呼叫前就初始化（行 5117-5124）
{
    c10::cuda::CUDAMultiStreamGuard sg(ncclStream);
    work->future_ = c10::make_intrusive<at::ivalue::Future>(...);
}
// 這確保 shadow_pre 將 work_ptr 暴露給重播中心時，Future 已存在，消滅 Race Condition
```

### 7.2 降級分支（行 5132-5163）

```cpp
if (C10_UNLIKELY(this->is_degraded_.load(acquire))) {
    if (opType == OpType::ALLREDUCE && proxy_comm_ready_.load(acquire)) {
        execute_shadow_allreduce(inputs[0], outputs[0], ncclStream, current_shadow_reduce_op_);
    } else {
        // 非 AllReduce 或 proxy comm 未就緒：fallback 到原生路徑（可能繼續失敗）
        fn(inputs[0], outputs[0], comm, ncclStream);
    }
    // Bug 10 fix：work->ncclComm_ 依角色選擇（行 5157-5163）
    work->ncclComm_ = is_faulty_rank ? local_nvlink_comm_ : proxy_global_comm_;
}
```

### 7.3 原生分支（行 5165-5203）

```cpp
else {
    try {
        C10D_NCCL_FT_CHECK_TIMEOUT(fn(...), ncclComm, ...);
        work->ncclComm_ = ncclComm;
    } catch (const NCCLFaultToleranceError& e) {
        if (!ft_disabled_) {
            work->ncclEndEvent_->record(ncclStream);
            work->setException(e);             // 設定例外
            work->numelIn_  = inputs[0].numel();  // 確保 debug 欄位完整
            work->numelOut_ = outputs[0].numel();
            workEnqueue(work);
            return work;              // 提早返回，等側車重播並 markCompleted
        }
        throw;
    }
}
```

**注意：** future_ 已在進入 try 前初始化。catch 區塊設定例外後提早返回；wait() 進入 FT 安全等待區，呼叫 `future_->wait()` 等側車的 `markCompleted`（不再有 spin-loop 或 A/B/C 分支）。

---

## 八、狀態變數對照表

| 變數 | 型別 | 語義 |
|------|------|------|
| `ft_disabled_` | `bool` | 環境變數 `NCCL_FT_DISABLE=1` 時為 true，完全繞過 FT 邏輯 |
| `is_degraded_` | `std::atomic<bool>` | 系統已降級，使用 Shadow Ping-Pong 路徑（改為 atomic） |
| `local_hardware_fault_mask_` | `std::atomic<uint64_t>` | bitmask，bit d 表示 NIC d 故障；NCCL callback 寫，side-car 讀 |
| `pending_shadow_seq_` | `std::atomic<uint64_t>` | 故障快照時的 shadow_seq，UINT64_MAX = 尚無 checkpoint |
| `shadow_seq_` | `std::atomic<uint64_t>` | 最新一次 D2H checkpoint 的 seq（shadow_pre 寫，atomic release） |
| `committed_shadow_seq_` | `std::atomic<uint64_t>` | 2PC agreed_ss（side-car 寫，主執行緒讀） |
| `final_commit_op_` | `std::atomic<uint64_t>` | 0 = 無 pending commit；cur_round+1 = 2PC 達成 |
| `ft_round_` | `std::atomic<uint64_t>` | 容錯回合計數，TCPStore key 的 namespace（recover 末尾遞增） |
| `faulty_local_devs_` | `std::unordered_set<int>` | 累積的故障 NIC local index（受 faulty_devs_mutex_ 保護） |
| `proxy_comm_ready_` | `std::atomic<bool>` | proxy_global_comm_ 已就緒，可執行降級 AllReduce |
| `proxy_comm_size_` | `int` | 降級後的 comm 大小 |
| `proxy_comm_rank_` | `int` | 本 rank 在降級 comm 中的排名（faulty rank = -1） |
| `current_shadow_reduce_op_` | `ReduceOp` | allreduce_impl 記錄，collective 降級分支使用 |
| `ft_root_comm_` | `std::shared_ptr<NCCLFTComm>` | 第一個 collective 建立的 global comm，記錄 fault callback 已在此 comm 上註冊 |
| `nvlink_init_attempted_` | `bool` | 防止 lazy-init 在 initLocalNvlinkComm 拋出後每次 collective 都重試 |
| `total_pinned_bytes_` | `std::atomic<size_t>` | Pinned Memory 總用量統計 |
| `total_pinned_buffers_` | `std::atomic<size_t>` | Pinned buffer 數量統計 |

---

## 九、WorkNCCLFT 生命週期

```
initWork()
  → pre() 呼叫前：future_ 已初始化（行 5117-5124）
  → pre()：shadow_pre D2H checkpoint，work_ptr 插入 in_flight_shadow_bufs_
  → enqueue to workMetaList_
       ↓
  Watchdog scan:
    isCompleted()? → 路徑 C：GC shadow buffer 到 free pool（正常路徑）
    exception (FT early-abort)? → 路徑 A：注射 FT 例外
    exception (FT clearing path)? → 路徑 B：erase + 保底 GC stash
       ↓
  側車（2PC 完成後直接呼叫）：
    recover_and_replay_inflight_ops()
      → rebuild topology → H2D restore → execute_shadow_allreduce
      → work_ptr->setException(nullptr)
      → work_ptr->future_->markCompleted(...)   ← 喚醒 wait()
       ↓
  DDP calls wait():
    FT 安全等待區（行 915-973）：
      is_ft_exception || (is_ft_managed && is_degraded_)
        → future_->wait()           ← 等側車 markCompleted
        → setException(nullptr)
        → replayed_end_event->block(currentStream)
        → unstash + Inline GC
        → return true（對 Python 完全透明）
    原生路徑（無 FT 例外）: synchronize() + handleException()
```

---

## 十、記憶體與串流管理

### 10.1 CUDA Streams

| Stream | 用途 |
|--------|------|
| `ncclStream_`（per-device） | NCCL collective 執行串流 |
| `shadow_copy_stream_` | D2H / H2D shadow buffer copy，與 ncclStream 並行 |
| 當前 compute stream | ATen 算術（backward pass gradient 計算） |

### 10.2 串流同步點

| 同步點 | 機制 |
|--------|------|
| compute → shadow_copy（shadow_pre） | `ctx.compute_event->record(compute_stream); ctx.compute_event->block(shadow_copy_stream_)` |
| shadow_copy → ncclStream（replay 時） | `restore_done.record(shadow_copy_stream_); restore_done.block(ncclStream)` |
| ncclStream → DDP compute（wait()） | `ncclEndEvent_->block(currentStream)` 或 `replayed_end_event->block(currentStream)` |

### 10.3 Pinned Memory（Shadow Buffer）

- `cudaHostAlloc`（`at::empty().pin_memory()`）= DMA 可達 pinned memory，保障 D2H/H2D 非同步傳輸
- Pool 設計避免頻繁 alloc/free：free pool → in-flight map → free pool（環狀複用）
- Best-Fit 策略：尋找大小最接近的 buffer 複用，減少記憶體浪費
- Memory Backpressure：in-flight 總量超過 2GB 時 sleep 2ms 等待，防止主執行緒暴走 OOM

---

## 十一、故障完整時序（目前實作）

```
T0: NIC mlx5_0 (dev_idx=0) 硬體故障
    └─ IB transport: ncclIbResiliencyHandleDeviceFailure()
                     → return ncclRemoteError

T1: NCCL progress thread
    └─ nccl_ft_trigger_fault(dev_idx=0)
       → nccl_ft_global_fault_callback(0)
       → pg->trigger_fault_proposal(0)
          ├─ local_hardware_fault_mask_.fetch_or(1, release)
          └─ shadow_seq_.load(acquire) → pending_shadow_seq_.CAS(UINT64_MAX → shadow_seq)

T2: Watchdog 掃描 workMetaList_
    └─ work.checkAndSetException() → 偵測 ncclRemoteError
       [FT Bug 1 fix] 不設 COMM_ERROR
       [early-abort] final_commit_op_ > 0 時 setException("FT early-abort")
                     → 進入 clearing path → erase + 保底 GC stash

T3: Side-car thread（pt_nccl_ft_side）
    └─ local_hardware_fault_mask_.exchange(0) = 0x1
       → 寫 "NCCL_FT_PROPOSE_<rank>_0" = "PROPOSE:0:<node>:0x1:<shadow_seq>"
       [等其他 rank 也寫完 PROPOSE]
       [rank 0] 計算 agreed_ss = min(所有 shadow_seq)
                寫 "NCCL_FT_SS_AGREED_0" = agreed_ss
                寫 "NCCL_FT_COMMIT_0" = "1"
       committed_shadow_seq_ = agreed_ss
       globalComm->abort("FT 2PC Committed")   ← 喚醒主執行緒
       final_commit_op_ = 1
       
       ↓ 側車直接執行重播（不等主執行緒）
       
       recover_and_replay_inflight_ops() [在 recovery_mutex_ 保護下]
          ├─ globalComm->abort()（若還未 abort）
          ├─ rebuild_attempt_++（確保 TCPStore key namespace 唯一）
          ├─ rebuild_shadow_ping_pong_topology()
          │    ├─ 同步 GC：std::move 舊 local/proxy comm → abort() → destructor
          │    ├─ [Data] setenv("NCCL_IB_HCA","mlx5_1,...") + nccl_ft_reset_ib_cache()
          │    ├─ [Ctrl] setenv("NCCL_SOCKET_IFNAME","enp*") + nccl_ft_reset_bootstrap_net()
          │    │         + nccl_ft_reset_net_socket()
          │    ├─ Pre-rebuild global barrier (NCCL_FT_REBUILD_SYNC_ATT_1_R<r>)
          │    ├─ initLocalNvlinkComm(attempt=1)
          │    ├─ proxy_global_comm_ = NCCLFTComm::create(14, new_rank, ...)
          │    └─ 零元素 AllReduce 驗證（若失敗 → 側車 catch → rebuild attempt=2）
          ├─ is_degraded_ = true
          ├─ seqs_to_replay = { seq | seq >= agreed_ss }
          ├─ for seq:
          │    ctx.copy_event.sync()
          │    H2D restore (flatten + copy_)
          │    execute_shadow_allreduce
          │      ├─ PROXY:   ncclGroupStart/End recv wards → AllReduce → ncclGroupStart/End send
          │      ├─ FAULTY:  ncclSend → ncclRecv (via NVLink through proxy)
          │      └─ HEALTHY: ncclAllReduce → cudaStreamSynchronize → ncclCommGetAsyncError
          │                  [P0 fix] 任一失敗即 throw → 側車 catch → 觸發 rebuild attempt=N+1
          │    record replayed_end_event
          │    work_ptr->setException(nullptr)
          │    work_ptr->future_->markCompleted(...)
          ├─ ft_round_++
          ├─ TCPStore cleanup (per-rank keys + per-attempt keys)
          ├─ pending_shadow_seq_.store(UINT64_MAX)  [為下一輪故障重置 checkpoint]
          ├─ final_commit_op_ = 0
          └─ 背景執行緒 sleep(3s) → nccl_ft_cleanup_stale_ib_contexts()
             [延遲清理 ibv_close_device，不阻塞恢復路徑]

T4: 主執行緒（wait()）
    └─ FT 安全等待區判斷（行 915-973）：
         is_ft_exception=true（comm abort 後 checkForNCCLErrors 偵測到 FaultToleranceError）
         → future_->wait()           ← 等側車已呼叫 markCompleted（T3 已完成）
         → setException(nullptr)     ← 抹除例外，保護 Python
         → replayed_end_event->block(currentStream)
         → stashed_for_allocator_safety_->unstash()
         → Inline GC: 回收 shadow buffer 到 free pool
         → return true (對 DDP 完全透明)

T5: 後續所有 AllReduce 走降級路徑
    └─ is_degraded_ = true → execute_shadow_allreduce(...)
```

---

## 十二、NCCL Custom Fork 修改說明

以下為 `nccl/` 目錄下所有 NCCL-FT 相關的原始碼修改，分為三類：

### 12.1 IB 傳輸層錯誤分類（`transport/net_ib/`）

| 修改位置 | 函式 | 修改內容 |
|----------|------|---------|
| `transport/net_ib/p2p_resiliency.cc` | `ncclIbResiliencyHandleDeviceFailure()` | 將原本的 `ncclSystemError` 改回傳 `ncclRemoteError`，讓 ProcessGroupNCCLFT 的 Watchdog 能辨識為可容錯的 NIC 硬體故障，而非致命系統錯誤 |
| `transport/net_ib/p2p_resiliency.cc` | `ncclIbResiliencyProbeHandleCompletionEvent()` | 探針 QP 偵測到完成錯誤時從 `ncclSuccess` 改為 `ncclRemoteError`，確保 fault callback 被觸發 |

### 12.3 Fault Callback 與動態環境變數重置（`init.cc` / `bootstrap.cc` / `net_socket.cc`）

| 函式 / 變數 | 檔案 | 說明 |
|-------------|------|------|
| `nccl_ft_is_disabled()` | `init.cc` | 檢查 `NCCL_FT_DISABLE` 環境變數，若為 1 則所有 FT 邏輯均跳過 |
| `ncclCommRegisterFaultCallback(comm, cb)` | `init.cc` | 每個 comm 可獨立註冊 fault callback；故障時 NCCL progress thread 呼叫 callback |
| `nccl_ft_trigger_fault(dev_idx)` | `init.cc` | 在 IB 傳輸層確認故障後呼叫，遍歷所有已註冊的 comm callback |
| `nccl_ft_reset_bootstrap_net()` | `bootstrap.cc` | 重置 `bootstrapNetInitDone=0`；在 `setenv("NCCL_SOCKET_IFNAME",...)` 後呼叫，讓下次 `ncclGetUniqueId()` 的 bootstrap socket 重讀介面名稱 |
| `nccl_ft_reset_net_socket()` | `net_socket.cc` | 重置 Socket transport singleton（`ncclNetIfs=-1`, `netRefCount=0`），釋放 pciPath 字串；與 `nccl_ft_reset_bootstrap_net()` 配對呼叫 |
| `nccl_ft_reset_ib_cache()` | `net_ib/init.cc` | 非阻塞式重置 IB 全域設備快取（`ncclNIbDevs=-1`），將舊 context 移入 stale 列表；在 `setenv("NCCL_IB_HCA",...)` 後呼叫，讓下次 `ncclIbInitDevices()` 重新掃描並只使用健康 NIC |
| `nccl_ft_cleanup_stale_ib_contexts()` | `net_ib/init.cc` | 阻塞式延遲清理：呼叫 `ibv_close_device()` 關閉被 `nccl_ft_reset_ib_cache()` 移入 stale 列表的 context。在 recover 完成後 3 秒背景執行 |

### 12.4 舊 NIC 排除方案歷史

**方案三（ABI 變更）：** 嘗試在 `ncclConfig_t` 新增 `bannedNicsMask` 欄位。已 revert 因破壞公開 ABI。

**方案二（Ban Mask + Topo Filter）：** 採用 global ban mask (`g_nccl_ft_banned_nics_mask`) 配合兩層 topo graph 過濾（`ncclTopoPopulateNics` 軟 ban + `ncclTopoGetLocalNetType` modulo filter）。已完全移除，原因：NCCL 的 topo graph 在 comm 建立時已確定，動態更新 ban mask 不能改變既存 QP 的路由，只能影響下一次 comm init。

**目前方案（環境變數重置，方案四）：** 完全移除 ban mask 機制。改用 `setenv` + 重置三個 singleton 強迫 NCCL 重新掃描：
- 所有涉及 ban mask 的程式碼均已移除，包含 `g_nccl_ft_banned_nics_mask`、`ncclCommBanNic()`、`ncclCommBanNicReset()`、`nccl_ft_is_nic_banned()`（`init.cc`），以及 topo graph 中的兩層過濾 hook（`topo.cc`）
- `nccl.h.in` 中對應的公開 API 宣告亦已移除

---

## 十三、多網卡故障支援狀態

### 同時多 NIC 故障

✅ 完整支援：
- `trigger_fault_proposal`：`fetch_or`（不是 CAS，所有 bit 都記錄）
- side-car：`exchange(0)` 一次取走整個 bitmask，PROPOSE message 帶完整 fault_mask
- 協調：`agg_fault_mask |= p_fmask`（OR 聚合所有 rank 的 mask）
- `findProxy`：走 round-robin，一定能找到非故障 proxy
- `my_wards`：proxy 可代理多個 faulty rank

### 逐步故障（Sequential Faults）

✅ 已修復（三個關鍵 bug）：

**Bug A（ft_round_ 從未遞增）：**
修復：`recover_and_replay_inflight_ops()` 末尾執行 `ft_round_++`，side-car 進入下一輪 2PC。

**Bug B（rollback_done_ 永遠不重置）：**
修復：改用 `final_commit_op_.load() == 0` 判斷是否已完成本輪恢復（原子變數，天然可重置）。

**Bug C（faulty_local_devs_ 在第二輪不更新）：**
修復：side-car Phase 1b 直接 OR 更新 `faulty_local_devs_`（累積），不覆蓋；
`rebuild_shadow_ping_pong_topology()` 每次 snapshot 最新的 `faulty_local_devs_`，包含歷次故障。

### 降級後繼續做 D2H shadow copy（設計決定）

`shadow_pre` 在 `ft_disabled_` 為 false 時**永遠執行**，包含降級後。
這是刻意設計：為第二次故障的 replay 預備 shadow buffer。不應移除。

---

## 十四、已知問題與狀態

### ✅ 已修復

| Bug | 說明 | 修復位置 |
|-----|------|---------|
| Bug 1 | collective() 降級路徑 execute_shadow_allreduce 被呼叫兩次 | 清理為乾淨二分支 |
| Bug 2 | catch NCCLFaultToleranceError 的 early return 未設 work->future_ | future_ 移到 pre() 前初始化 |
| Bug 3 | recover_and_replay seq 邊界（確認 `>=` 是正確的） | 保留 `>=`，~5447 |
| Bug 4 | get_or_allocate_shadow_context 在鎖內呼叫 pin_memory() | 三段式設計，~5012 |
| Bug 5 | 降級後 shadow_pre 仍執行 D2H copy | **設計決定**，不是 bug |
| Bug 6 | 降級模式下 intraNodeComm fast-path 繞過 FT 邏輯 | allreduce() 加 `!is_degraded_` 條件，~6281 |
| Bug 7 | proxy 的 ATen 算術未在正確 stream 執行 | setCurrentCUDAStream + restore，~4563 |
| Bug 10 | 降級模式下 work->ncclComm_ 指向錯誤 comm | collective() 依角色選 comm，~5317 |
| Bug 12 | 多 NIC 同時故障 CAS 丟失 bit | 改為 fetch_or bitmask，~4651 |
| Bug A | ft_round_ 從未遞增 | recover_and_replay 末尾 ft_round_++，~5511 |
| Bug B | rollback_done_ 永遠不重置 | 改用 final_commit_op_ == 0 判斷 |
| Bug C | faulty_local_devs_ 第二輪不更新 | side-car Phase 1b OR 累積更新 |
| shadow_seq_ 資料競爭 | shadow_seq_ 是普通 uint64_t，主執行緒寫、NCCL callback thread 讀 | 改為 `std::atomic<uint64_t>` |
| logPrefix() static Bug | function-local static 第一次呼叫後固定不變 | 移除 `static` 關鍵字 |
| 多維 Tensor copy_ Shape Mismatch | H2D 還原及 D2H copy 對多維 tensor 直接操作造成 shape 不符 | 兩處均加 `.flatten()` |
| 迴圈內 CUDAEvent 重複 create/destroy | recover_and_replay 每次迭代原本建立新 restore_done event | 迴圈外宣告 `restore_done`，迴圈內 `record()` 複用 |
| Watchdog 例外清除後無 GC 保底 | FT clearing path 清除 exception 後 stash 無處釋放 | 加入保底 GC：push 到 `shelvesToUnstash_` |
| compute_done event per-call create | shadow_pre 每次建立 CUDAEvent | 提升為 ShadowContext::compute_event 複用 |
| ncclCommSplit 在 RemoteError comm 上失敗 | local_nvlink_comm_ 的建立依賴 parent comm | 改用 NCCLFTComm::create（TCPStore rendezvous，完全獨立） |
| 側車依賴主執行緒才能啟動重播 | 主執行緒若卡死，重播永遠不啟動 | 側車在 2PC 完成後直接呼叫 recover_and_replay_inflight_ops() |
| **initLocalNvlinkComm key 衝突（lazy-init vs recovery 共用 ROUND_0 key）** | NIC 在第一個 collective 期間故障時，lazy-init 失敗的 rank 沒有寫入 `LOCAL_READY` key；recovery 路徑使用相同的 key，進入 bootstrap 時缺少某些 rank，永久卡住 | `initLocalNvlinkComm(attempt)` 使用呼叫者傳入的 attempt（lazy=0, recovery=ft_round+1）作為 key 後綴；加入 entry-arrival barrier 確保所有 local rank 同時進入 |
| **GC 執行緒不呼叫 abort()，造成 NCCL 資源洩漏** | `~NCCLFTComm()` 只印警告，不 ncclCommAbort/Destroy；舊版 GC 執行緒直接 reset() 等於靜默洩漏 | GC 執行緒在 sleep 後先呼叫 `abort()`（設 aborted_=true 以抑制警告），再 reset() |
| **`pending_shadow_seq_` 在 recovery 後未重置** | 第二次故障的 PROPOSE 帶著第一輪的 checkpoint seq，agreed_ss 錯誤，可能重播已完成的 op | `recover_and_replay_inflight_ops()` 末尾在 `final_commit_op_.store(0)` 前重置 `pending_shadow_seq_` 為 `UINT64_MAX` |
| **DEBUG-HANG / DEBUG-RUNAWAY log 殘留** | 暫時性調試輸出混入生產 log，干擾問題分析 | 移除所有 `[DEBUG-HANG]` 和 `[DEBUG-RUNAWAY]` log |
| **P0：HEALTHY rank rebuild deadlock（Pre-rebuild global barrier 死鎖）** | attempt=N 失敗後只有 PROXY rank 和直接收到 socket error 的 HEALTHY rank 觸發 rebuild attempt=N+1；其餘 HEALTHY rank 的 ncclAllReduce 非同步完成，side-car 不感知錯誤，不重入 rebuild，導致 Pre-rebuild global barrier 等不到所有 16 rank，永遠死鎖 | `execute_shadow_allreduce` HEALTHY 路徑在 ncclAllReduce enqueue 後加 `cudaStreamSynchronize` + `ncclCommGetAsyncError`；任一失敗即 throw，讓側車 catch 到並重入 rebuild |
| **P1：proxy_global_comm_ 仍使用 dead NIC 建立 QP** | 方案二的 topo filter 無法完全防止 dead NIC 被選中（topo graph 在 comm init 時已固定） | **已改以方案四（環境變數重置）取代**：`setenv("NCCL_IB_HCA", healthy_hcas)` + `nccl_ft_reset_ib_cache()` 讓下一次 `ncclIbInitDevices()` 重新掃描時根本不看到 dead NIC，從根本排除問題。`ncclTopoGetLocalNetType()` 目前不含任何 ban filter（已清除）。 |

### ❓ 待確認 / 待修復

| 項目 | 嚴重性 | 說明 |
|------|--------|------|
| `sscanf` 格式字串可移植性 | **低危** | 行 4752：`%lu`/`%lx` 在 Windows 上對應 32-bit `unsigned long` 而非 64-bit `uint64_t`。應改用 `SCNu64`/`SCNx64`（`<cinttypes>`）。 |
| 側車在 recovery_mutex_ 上潛在死結 | **低危** | 側車呼叫 recover_and_replay → 鎖 recovery_mutex_；主執行緒 wait() 不觸發 recover，無死結風險。但側車持鎖期間若觸發新 exception，後者需等待當前鎖釋放，多等一輪，可接受。 |

### 遺留效能注意事項

| 項目 | 影響 | 說明 |
|------|------|------|
| shadow_pre `setCurrentCUDAStream` / `getCurrentCUDAStream` 每次呼叫 | 極低 | 兩次 thread-local 讀寫，無 CUDA API 呼叫 |
| Inline GC 在 get_or_allocate 鎖內掃描 in_flight_shadow_bufs_ | O(n) 掃描 | n = 飛行中 bucket 數，通常 < 100，可接受 |
| Memory Backpressure sleep 2ms | 輕微 latency | 只在 in-flight > 2GB 時觸發，視為異常保護 |
| proxy step 2 每次 at::empty_like | 輕微 GPU allocator 呼叫 | 降級後每次 AllReduce 都配置 ward_bufs；可考慮預配置加入 ShadowContext pool |
