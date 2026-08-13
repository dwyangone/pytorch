# ProcessGroupNCCLFT — System Architecture

> 依據 `ProcessGroupNCCLFT.cpp` 目前實際程式碼撰寫，2026-08。

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

`wait()` 有兩個 FT 攔截點（行 907-952）：

**攔截點 1（行 907-909）**
```
if (!pg_->ft_disabled_ && pg_->final_commit_op_.load() > 0)
    pg_->recover_and_replay_inflight_ops();
```
條件成立時（2PC 已達成共識），主執行緒在此觸發全域重播中心。

**攔截點 2（行 914-952）**
```
if (!pg_->ft_disabled_ && pg_->is_degraded_ && seq >= committed_shadow_seq_)
```
條件成立時（已降級 + 此 seq 已被重播）：
1. 等待 `replayed_end_event` → block 目前 compute stream
2. 呼叫 `stashed_for_allocator_safety_->unstash()` 釋放扣留的 Tensor 參照
3. 清除可能被 Watchdog 設置的 exception（`setException(nullptr)`）
4. GC：將 ShadowContext 放回 free pool，從 `in_flight_shadow_bufs_` 移除
5. `return true` — 對 DDP 隱藏整個故障過程

### 2.2 Watchdog 執行緒（`pt_nccl_watchdg`）

持續掃描 `workMetaList_`，對每個 work 執行：

**路徑 A：FT early-abort（行 2503-2526）**
```
if (!ft_disabled_ && !work.exception() && final_commit_op_ > 0)
```
- 在 ncclStream 上 record `ncclEndEvent_`（解鎖可能等待的 DDP）
- 呼叫 `work.setException("FT early-abort: Comm dead.")`
- 不 erase（讓下一輪 `work.exception()` 分支自動處理）

**路徑 B：FT clearing path（行 2555-2641）**
```
if (!ft_disabled_ && work.exception())
```
- 清除 `pg_->error_` → `SUCCESS`
- record `ncclEndEvent_`（確保 block() 能解鎖）
- `work.setException(nullptr)`
- erase from workMetaList_
- **不做 GC**（shadow buffer 保留供 wait() 攔截點 2 使用）

**路徑 C：正常完成 GC（行 2792-2803）**
```
if (work.isCompleted() && work.opType_ == OpType::ALLREDUCE && !ft_disabled_)
```
- 從 `in_flight_shadow_bufs_` 找到對應 seq
- 清空 `original_input` / `original_output` GPU 參照
- 歸還 ShadowContext 到 `free_shadow_bufs_`（free pool）

### 2.3 Side-car 執行緒（`pt_nccl_ft_side`）

透過 TCPStore 執行 2PC 協商，名稱 `pt_nccl_ft_side`，由 `start_ft_negotiator_thread()` 啟動。

---

## 三、NCCL 與 ProcessGroupNCCLFT 的互動

### 3.1 標準 collective 路徑（正常情況）

```
Python DDP
  └─ allreduce(tensors, opts)              [行 5985]
       └─ allreduce_impl(tensor, opts)     [行 5908]
            ├─ current_shadow_reduce_op_ = opts.reduceOp   [記錄 op]
            └─ collective(tensor, tensor, fn, shadow_pre, post_noop, ALLREDUCE)
                 ├─ [shadow_pre lambda]    D2H checkpoint   [行 5929-5956]
                 ├─ collective() 正常分支 [行 5117-5151]
                 │    └─ C10D_NCCL_FT_CHECK_TIMEOUT(ncclAllReduce, ...)
                 └─ work->ncclEndEvent_->record(ncclStream) [行 5159]
```

**NCCL API 呼叫鏈（正常）：**
1. `ncclAllReduce(input, output, numel, dataType, reduceOp, comm, stream)` — 非同步排入 ncclStream
2. `ncclEndEvent_->record(ncclStream)` — 排入 EndEvent，DDP 的 `wait()` 透過 `block()` 等此 event

### 3.2 shadow_pre lambda（D2H Checkpoint，行 5929-5956）

每個 AllReduce 呼叫**在 ncclAllReduce 啟動前**執行 shadow_pre：

1. 呼叫 `get_or_allocate_shadow_context(tensor)` — 從 free pool 取或新建 ShadowContext
2. 設定 `ctx.original_input = ctx.original_output = tensor`（in-place AllReduce）
3. 設定 `ctx.reduce_op = opts.reduceOp`
4. 鎖內插入 `in_flight_shadow_bufs_[seq] = ctx`
5. 在 shadow_copy_stream_ 上：
   - record `compute_done` event，block shadow_copy_stream_ 等 compute 完成
   - `ctx.buffer.narrow(0, 0, tensor.numel()).copy_(tensor, non_blocking=true)` — D2H copy
   - record `ctx.copy_event`（精準標記此 seq 的 D2H 完成時間點）
6. `shadow_seq_ = current_seq`（供 trigger_fault_proposal 快照用）

`shadow_seq_` 的語義：**D2H 備份已排入、跨節點 AllReduce 尚未確認完成**的最後一個 seq。

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
  trigger_fault_proposal(dev_idx)         [行 4461]
    ├─ local_hardware_fault_mask_.fetch_or(1ULL << dev_idx)   [atomic bitmask]
    └─ pending_shadow_seq_.compare_exchange_strong(UINT64_MAX, shadow_seq_)
```

`trigger_fault_proposal` 必須在微秒內返回（NCCL progress thread 呼叫），只做 atomic write。

### 3.4 NCCL Comm 管理 API

| API | 用途 | 呼叫時機 |
|-----|------|---------|
| `ncclCommRegisterFaultCallback(comm, cb)` | 每 rank 各自在自己的 comm 上註冊 fault callback | 第一個 collective 的 lazy-init（行 4956-4993） |
| `ncclCommBanNic(dev_idx)` | 將指定 NIC 加入黑名單，下次 ncclCommInitRank 跳過 | `rebuild_shadow_ping_pong_topology()` 中，所有 rank 都呼叫 |
| `ncclGetUniqueId(&id)` | 生成新 rendezvous ID | rebuild 時 proxy_comm_rank_==0 的 rank 呼叫 |
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

`allreduce()`（行 6008）：
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

### 4.2 通訊器建立流程

**global comm（`devNCCLCommMap_`）：**
- 由 `initNCCLComm()` 在第一個 collective 建立
- 使用標準 `ncclCommInitRank` 路徑

**local_nvlink_comm_（行 4040-4089）：**
- Lazy init，在第一個 collective 後執行（行 4956-4992）
- local_rank == 0 生成 `ncclUniqueId`，寫入 TCPStore
- 其他 local rank 從 TCPStore 讀取
- `NCCLFTComm::create(localDeviceCount_, local_rank, localId, device.index(), config)`

**proxy_global_comm_（行 4196-4236）：**
- 在 `rebuild_shadow_ping_pong_topology()` 中建立
- 只有健康 rank 才建立（faulty rank 設 `proxy_global_comm_ = nullptr`）
- 流程：
  1. 呼叫 `ncclCommBanNic(d)` 排除所有 faulty NIC（所有 rank 都呼叫）
  2. proxy_comm_rank_ == 0 生成新 `ncclUniqueId`，寫入 TCPStore
  3. `NCCLFTComm::create(proxy_comm_size_, proxy_comm_rank_, proxyId, device.index(), config)`

### 4.3 proxy_comm 大小與排名計算（行 4134-4149）

```
proxy_comm_size_ = total_size - num_faulty_per_node * num_nodes
  (對稱降級：每個節點都去掉相同的 local rank 索引)

proxy_comm_rank_：重新對健康 rank 編號（跳過 faulty_devs）
```

### 4.4 comms 的 abort 時機

| 情況 | abort 呼叫者 | 原因 |
|------|-------------|------|
| 2PC committed | side-car thread（行 4818-4822） | 強制喚醒卡在 ncclAllReduce 的主執行緒 |
| recover_and_replay_inflight_ops() 開始 | 主執行緒（行 5216-5218） | 確保舊 comm 徹底死亡再 rebuild |

---

## 五、Shadow Ping-Pong 容錯機制

### 5.1 ShadowContext 結構

```cpp
struct ShadowContext {
    at::Tensor buffer;                                        // pinned CPU memory（D2H 備份）
    std::shared_ptr<at::cuda::CUDAEvent> copy_event;         // per-seq D2H 完成事件
    std::shared_ptr<at::cuda::CUDAEvent> replayed_end_event; // replay kernel 完成事件
    at::Tensor original_input;                               // GPU tensor 參照（in-place = output）
    at::Tensor original_output;                              // GPU tensor 參照
    c10d::ReduceOp reduce_op;                                // replay 使用的 ReduceOp
};
```

**Shadow Buffer Pool 管理（`free_shadow_bufs_` + `in_flight_shadow_bufs_`）：**

```
allreduce() 呼叫時：
  get_or_allocate_shadow_context(tensor)
    ├─ 先鎖內查 free_shadow_bufs_（O(n) 找 numel 足夠的）
    └─ 找不到 → 解鎖後 cudaHostAlloc + cudaEventCreate（昂貴，但不在鎖內）
  → 插入 in_flight_shadow_bufs_[seq]

成功完成後（Watchdog 正常 GC 路徑）：
  in_flight_shadow_bufs_[seq]
  → ctx.original_input = Tensor()   (斷開 GPU 參照)
  → ctx.original_output = Tensor()
  → free_shadow_bufs_.push_back(ctx)
  → in_flight_shadow_bufs_.erase(it)

replay 完成後（wait() 攔截點 2 GC）：
  相同的 GC 流程，由主執行緒在 wait() 中執行
```

`get_or_allocate_shadow_context` 採用**兩段式設計**（Bug 4 fix）：
- 鎖內：只查 free pool（輕量）
- 鎖外：執行昂貴的 `pin_memory()` + `cudaEventCreate`

### 5.2 execute_shadow_allreduce 四角色邏輯（行 4270-4457）

根據 `local_rank = rank_ % localDeviceCount_` 與 `faulty_local_devs_` 決定角色。

**角色分配函式 `findProxy(faulty_dev, localDeviceCount, faulty_devs)`（行 4259-4267）：**
- 從 `(faulty_dev+1) % N` 開始走，找第一個不在 faulty set 中的 local rank
- 保證多 NIC 故障時仍能找到 proxy（要求 faulty_devs.size() < localDeviceCount_）

**FAULTY 角色（本地 NIC 故障）：**
```
Step 1: ncclSend(input → my_proxy, nvlink_comm)
Step 4: ncclRecv(result ← my_proxy, nvlink_comm)
```

**PROXY 角色（為一個或多個 faulty rank 代勞）：**
```
Step 1: ncclRecv 所有 ward 的 tensor（一個 ncclGroup）
Step 2: output = input + sum(ward_bufs)   (pre-aggregate on stream)
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

**ATen 算術串流保證（Bug 7 fix，行 4401-4410）：**
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
4. `final_commit_op_.store(cur_round + 1, release)` — 通知主執行緒
5. 等待 `final_commit_op_` 被主執行緒重置為 0（確保不重複 relay）

**pending_shadow_seq_ reset：** 在 `final_commit_op_` 設定前，side-car 重置為 UINT64_MAX，供下一輪使用。

---

## 六、恢復流程：recover_and_replay_inflight_ops()（行 5201-5289）

### 6.1 重入防護

```cpp
uint64_t commit_signal = final_commit_op_.load(acquire);
if (commit_signal == 0) return;   // Bug B fix：用 atomic 變數取代 rollback_done_
```

使用 `recovery_mutex_` 確保同一時間只有一個執行緒進行恢復（防止同一 Process 多次觸發）。

### 6.2 恢復步驟

```
Step 1: globalComm->abort("FT Recovery")   [確保舊 comm 死亡]

Step 2: rebuild_shadow_ping_pong_topology()
          └─ 使用最新的 faulty_local_devs_（2PC 已聚合所有節點的故障資訊）
             ├─ 所有 rank：ncclCommBanNic(d) for d in faulty_devs
             └─ 健康 rank：NCCLFTComm::create(proxy_comm_size_, proxy_comm_rank_, ...)
        is_degraded_ = true

Step 3: 收集 seqs_to_replay
          agreed_ss = committed_shadow_seq_.load()
          from in_flight_shadow_bufs_: collect seq >= agreed_ss
          sort ascending

Step 4: 依序 H2D restore + replay
          for seq in seqs_to_replay:
            ctx.copy_event->synchronize()           [等 D2H 備份完成]
            on shadow_copy_stream_:
              ctx.original_input.copy_(ctx.buffer)  [H2D 還原梯度]
            record restore_done event
            restore_done.block(ncclStream)          [串流同步]
            execute_shadow_allreduce(ctx.original_input, ctx.original_output,
                                     ncclStream, ctx.reduce_op)
            ctx.replayed_end_event->record(ncclStream)

Step 5: ft_round_++          [Bug A fix：讓 side-car 知道可以進入下一輪]
        TCPStore cleanup:
          deleteKey("NCCL_FT_PROPOSE_<rank>_<round>")   [每個 rank 清自己的]
          if rank==0: deleteKey("NCCL_FT_COMMIT_<round>")
                      deleteKey("NCCL_FT_SS_AGREED_<round>")

Step 6: final_commit_op_.store(0, release)
          [解除 side-car 的等待，允許下一輪 2PC 開始]
```

### 6.3 `agreed_ss` 的正確語義

`shadow_seq_` 在 `shadow_pre` 中設定（D2H copy 已排入但 ncclAllReduce 尚未確認）。
replay 邊界條件是 `seq >= agreed_ss`（含 agreed_ss 本身），
因為 agreed_ss 那個 bucket 的跨節點 AllReduce 狀態未知，必須重播。

### 6.4 wait() 攔截點 2 的 GC

replay 完成後，shadow buffer **不在 recover 函式內** GC，而是在每個 work 的 `wait()` 被呼叫時（攔截點 2）由主執行緒負責回收。這保證了 GC 的時序正確性（replayed_end_event 已在 wait() 中 block 完成）。

---

## 七、collective() 中的執行路徑（二分支設計）

### 7.1 降級分支（行 5085-5114）

```cpp
if (C10_UNLIKELY(this->is_degraded_)) {
    if (opType == OpType::ALLREDUCE && proxy_comm_ready_) {
        execute_shadow_allreduce(inputs[0], outputs[0], ncclStream, current_shadow_reduce_op_);
    } else {
        // 非 AllReduce 或 proxy comm 未就緒：fallback 到原生路徑
        fn(inputs[0], outputs[0], comm, ncclStream);
    }
    // Bug 10 fix：work->ncclComm_ 指向實際使用的 comm
    work->ncclComm_ = is_faulty_rank ? local_nvlink_comm_ : proxy_global_comm_;
}
```

### 7.2 原生分支（行 5117-5151）

```cpp
else {
    try {
        C10D_NCCL_FT_CHECK_TIMEOUT(fn(...), ncclComm, ...);
    } catch (const NCCLFaultToleranceError& e) {
        if (!ft_disabled_) {
            // Bug 2 fix：補齊 work->future_，然後 workEnqueue + return work
            work->ncclEndEvent_->record(ncclStream);
            work->future_ = make_intrusive<Future>(...);
            work->future_->markCompleted(...);
            return work;  // 提早返回，等 side-car + wait() 處理
        }
        throw;  // FT disabled → 往上拋
    }
    work->ncclComm_ = ncclComm;
}
```

### 7.3 完整 collective() 流程圖

```
collective(inputs, outputs, fn, pre, post, opType, ...)
  │
  ├─ seqCollective_++
  ├─ initNCCLComm (if null)
  ├─ Lazy FT init (first collective only):
  │    ├─ ncclCommRegisterFaultCallback(ncclComm, nccl_ft_global_fault_callback)
  │    └─ initLocalNvlinkComm()
  │
  ├─ work = initWork(device, rank_, opType, ...)
  ├─ work->outputs_ = outputs
  ├─ stashed_for_allocator_safety_->stash(inputs, outputs)
  │
  ├─ pre(ncclStream, work)   ← shadow_pre：D2H checkpoint
  │
  ├─ [is_degraded_ = true]
  │    └─ execute_shadow_allreduce(...)   ← Shadow Ping-Pong
  │
  └─ [is_degraded_ = false]  ← 原生 NCCL
       ├─ C10D_NCCL_FT_CHECK_TIMEOUT(fn, ...)
       └─ [catch NCCLFaultToleranceError] → early return
  │
  ├─ post(ncclStream, work)
  ├─ ncclEndEvent_->record(ncclStream)
  ├─ work->future_ = markCompleted(outputs)
  └─ workEnqueue(work)
```

---

## 八、狀態變數對照表

| 變數 | 型別 | 語義 |
|------|------|------|
| `ft_disabled_` | `bool` | 環境變數 `NCCL_FT_DISABLE=1` 時為 true，完全繞過 FT 邏輯 |
| `is_degraded_` | `bool` | 系統已降級，使用 Shadow Ping-Pong 路徑 |
| `local_hardware_fault_mask_` | `std::atomic<uint64_t>` | bitmask，bit d 表示 NIC d 故障；NCCL callback 寫，side-car 讀 |
| `pending_shadow_seq_` | `std::atomic<uint64_t>` | 故障快照時的 shadow_seq，UINT64_MAX = 尚無 checkpoint |
| `shadow_seq_` | `uint64_t` | 最新一次 D2H checkpoint 的 seq（shadow_pre 寫） |
| `committed_shadow_seq_` | `std::atomic<uint64_t>` | 2PC agreed_ss（side-car 寫，主執行緒讀） |
| `final_commit_op_` | `std::atomic<uint64_t>` | 0 = 無 pending commit；cur_round+1 = 2PC 達成 |
| `ft_round_` | `uint64_t` | 容錯回合計數，TCPStore key 的 namespace（主執行緒寫） |
| `faulty_local_devs_` | `std::unordered_set<int>` | 累積的故障 NIC local index（受 faulty_devs_mutex_ 保護） |
| `proxy_comm_ready_` | `std::atomic<bool>` | proxy_global_comm_ 已就緒，可執行降級 AllReduce |
| `proxy_comm_size_` | `int` | 降級後的 comm 大小 |
| `proxy_comm_rank_` | `int` | 本 rank 在降級 comm 中的排名（faulty rank = -1） |
| `current_shadow_reduce_op_` | `ReduceOp` | allreduce_impl 記錄，collective 降級分支使用 |

---

## 九、WorkNCCLFT 生命週期

```
initWork()
  → enqueue to workMetaList_
       ↓
  Watchdog scan:
    isCompleted()?
      ├─ Yes → GC shadow buffer → erase
      ├─ exception (FT)? → FT clearing path → erase (no GC)
      └─ early-abort? → record EndEvent + setException
       ↓
  DDP calls wait():
    攔截點 1: final_commit_op_ > 0 → recover_and_replay_inflight_ops()
    攔截點 2: is_degraded_ + seq >= committed_ss → GC + return true
    原生路徑: synchronize() → handleException()
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
| compute → shadow_copy（shadow_pre） | `compute_done.record(compute_stream); compute_done.block(shadow_copy_stream_)` |
| shadow_copy → ncclStream（replay 時） | `restore_done.record(shadow_copy_stream_); restore_done.block(ncclStream)` |
| ncclStream → DDP compute（wait()） | `ncclEndEvent_->block(currentStream)` 或 `replayed_end_event->block(currentStream)` |

### 10.3 Pinned Memory（Shadow Buffer）

- `cudaHostAlloc`（`at::empty().pin_memory()`）= DMA 可達 pinned memory，保障 D2H/H2D 非同步傳輸
- Pool 設計避免頻繁 alloc/free：free pool → in-flight map → free pool（環狀複用）

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
          ├─ local_hardware_fault_mask_.fetch_or(1)
          └─ pending_shadow_seq_ = shadow_seq_（若為 UINT64_MAX → CAS 成功）

T2: Watchdog 掃描 workMetaList_
    └─ work.checkAndSetException() → 偵測 ncclRemoteError
       [FT Bug 1 fix] 不設 COMM_ERROR
       [early-abort] final_commit_op_ > 0 時 record EndEvent + setException

T3: Side-car thread（pt_nccl_ft_side）
    └─ local_hardware_fault_mask_.exchange(0) = 0x1
       → 寫 "NCCL_FT_PROPOSE_<rank>_0" = "PROPOSE:0:<node>:0x1:<shadow_seq>"
       [等其他 rank 也寫完 PROPOSE]
       [rank 0] 計算 agreed_ss = min(所有 shadow_seq)
                寫 "NCCL_FT_SS_AGREED_0" = agreed_ss
                寫 "NCCL_FT_COMMIT_0" = "1"
       committed_shadow_seq_ = agreed_ss
       globalComm->abort("FT 2PC Committed")  ← 喚醒主執行緒
       final_commit_op_ = 1                   ← 通知主執行緒

T4: 主執行緒（wait() 攔截點 1）
    └─ final_commit_op_ > 0 → recover_and_replay_inflight_ops()
          ├─ globalComm->abort()（若還未 abort）
          ├─ rebuild_shadow_ping_pong_topology()
          │    ├─ ncclCommBanNic(0)
          │    └─ proxy_global_comm_ = NCCLFTComm::create(14, new_rank, ...)
          ├─ is_degraded_ = true
          ├─ seqs_to_replay = { seq | seq >= agreed_ss }
          ├─ for seq: ctx.copy_event.sync(); H2D restore; execute_shadow_allreduce; record replayed_end_event
          ├─ ft_round_++
          ├─ TCPStore cleanup
          └─ final_commit_op_ = 0

T5: DDP 繼續呼叫 wait() (攔截點 2)
    └─ is_degraded_ && seq >= committed_ss
       → replayed_end_event->block(currentStream)
       → stashed->unstash()
       → setException(nullptr)
       → GC shadow buffer
       → return true

T6: 後續所有 AllReduce 走降級路徑
    └─ is_degraded_ = true → execute_shadow_allreduce(...)
```

---

## 十二、NCCL Custom Fork 對應

| NCCL 修改 | 位置 | 作用 |
|-----------|------|------|
| `ncclIbResiliencyHandleDeviceFailure()` | `transport/net_ib/p2p_resiliency.cc` | `ncclSystemError → ncclRemoteError`（讓 PG 可辨識 NIC 故障） |
| `ncclIbResiliencyProbeHandleCompletionEvent()` | 同上 | `ncclSuccess → ncclRemoteError` |
| `ncclTopoPopulateNics()` 移除 ban check | `graph/topo.cc` | 允許 `ncclCommBanNic` 在 reinit 時生效（topo discovery 尊重 ban list） |
| `ncclCommBanNicReset()` | `init.cc` | 重置 ban list（預留 API，目前暫未使用） |
| `nccl_ft_trigger_fault(dev_idx)` | `init.cc` | 呼叫已註冊的 fault callback |
| `ncclCommRegisterFaultCallback(comm, cb)` | `init.cc` | 每 comm 可獨立註冊一個 fault callback |

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
| Bug 1 | collective() 降級路徑 execute_shadow_allreduce 被呼叫兩次 | 清理為乾淨二分支，移除舊切換點 A/B |
| Bug 2 | catch NCCLFaultToleranceError 的 early return 未設 work->future_ | 行 5133-5139：補齊 future_ |
| Bug 3 | recover_and_replay seq 邊界（確認 `>=` 是正確的） | 行 5232：保留 `>=` |
| Bug 4 | get_or_allocate_shadow_context 在鎖內呼叫 pin_memory() | 兩段式設計（行 4865-4894） |
| Bug 5 | 降級後 shadow_pre 仍執行 D2H copy | **設計決定**，不是 bug，為二次故障 replay 預備 |
| Bug 6 | 降級模式下 intraNodeComm fast-path 繞過 FT 邏輯 | allreduce() 加 `!is_degraded_` 條件（行 6008） |
| Bug 7 | proxy 的 ATen 算術未在正確 stream 執行 | setCurrentCUDAStream + restore（行 4401-4410） |
| Bug 10 | 降級模式下 work->ncclComm_ 指向錯誤 comm | collective() 行 5108-5114：依角色選 comm |
| Bug 12 | 多 NIC 同時故障 CAS 丟失 bit | 改為 fetch_or bitmask（trigger_fault_proposal） |
| Bug A | ft_round_ 從未遞增 | recover_and_replay 末尾 ft_round_++ |
| Bug B | rollback_done_ 永遠不重置 | 改用 final_commit_op_ == 0 判斷 |
| Bug C | faulty_local_devs_ 第二輪不更新 | side-car Phase 1b OR 累積更新 |

### ❓ 待確認

| 項目 | 說明 |
|------|------|
| `ncclCommBanNic` 作用域 | Process-level global 還是 per-comm？影響是否需要在 reinit 前重複呼叫 |
| `NCCLFTComm::create` topo discovery | 呼叫後是否確實跳過被 ban 的 NIC？需要實驗確認 |

### 遺留效能注意事項

| 項目 | 影響 | 說明 |
|------|------|------|
| early-abort 的 ncclEndEvent_ 被 record 兩次 | 低 | 行 2517（early-abort）和 2633（FT clearing path）；多一次 CUDA API 呼叫，無害 |
| 降級後仍執行 D2H copy | 輕微額外記憶體頻寬 | 刻意設計，為二次故障準備 |
