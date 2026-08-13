# ProcessGroupNCCLFT — System Architecture

> 根據 `ProcessGroupNCCLFT.cpp` 實際程式碼整理（2026-08）

---

## 一、元件概覽

`ProcessGroupNCCLFT` 是以 `ProcessGroupNCCL` 為基礎擴充的自訂 NCCL 容錯 backend，在 PyTorch c10d 框架中以名稱 `nccl_ft` 註冊。主要功能是在發生單一 NIC 硬體故障時，透過「Shadow Ping-Pong」機制讓分散式訓練在**不重啟、不拋出 Python 例外**的情況下繼續執行。

### 硬體假設

```
Server 0: GPU 0–7，每張 GPU 有一張 NUMA-local NIC (mlx5_0 ~ mlx5_7)
Server 1: GPU 8–15，同樣的 NIC 配置
同 server 內所有 GPU 以 NVLink 互連
torchrun --nproc_per_node=8 --nnodes=2
```

---

## 二、執行緒模型（三執行緒）

```
┌──────────────────────────────────────────────────────────────┐
│  主執行緒 (main thread)                                       │
│  ・呼叫 allreduce / collective 等操作                         │
│  ・collective() 中有舊的 2PC barrier（行 5203-5246），但在     │
│    目前架構下已不是主要恢復路徑（詳見第六章）                 │
│  ・wait() 中包含兩個攔截點（行 907-938）：                    │
│    攔截點 1：偵測 final_commit_op_ > 0 → 呼叫                │
│              recover_and_replay_inflight_ops()                │
│    攔截點 2：is_degraded_ 時等待 replayed_end_event          │
│              然後 unstash / clearException / return true       │
└─────────────────────┬────────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────────┐
│  Watchdog (pt_nccl_watchdg)                                   │
│  ・每 100ms 掃描 workMetaList_                                │
│  ・偵測 exception 後（FT 模式）：                             │
│    - 清除 exception + 記錄 ncclEndEvent_                      │
│    - 不寫 fault_mask，不 rethrow（Fix A）                     │
│  ・early-abort（Fix 3）：若 final_commit_op_ > 0，            │
│    強制記錄 ncclEndEvent_，讓 wait() 可以繼續                 │
│  ・成功完成 AllReduce 後 GC：                                 │
│    in_flight_shadow_bufs_[seq] → 釋放 Tensor 參照             │
│    → 放回 free_shadow_bufs_ ring pool                        │
└─────────────────────┬────────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────────┐
│  Side-car negotiator (pt_nccl_ft_side)                        │
│  ・偵測 local_hardware_fault_mask_ 被設定（由 NCCL callback   │
│    trigger_fault_proposal() 寫入）                            │
│  ・每 rank 各自寫入 TCPStore per-rank PROPOSE key             │
│  ・等待全部 size_ 個 PROPOSE keys 就緒（Phase 1 collection）  │
│  ・rank 0 計算 agreed_shadow_seq（取 min）並寫 COMMIT key     │
│  ・所有 rank 讀到 COMMIT 後：                                  │
│    - 更新 committed_shadow_seq_ (release)                     │
│    - 強制 abort 全域 comm（喚醒卡在 ncclAllReduce 的主執行緒） │
│    - 設定 final_commit_op_ ≠ 0（喚醒 wait() 攔截點 1）        │
│  ・等主執行緒將 final_commit_op_ 重置為 0 後，進入下一輪      │
└──────────────────────────────────────────────────────────────┘
```

---

## 三、NCCL 與 ProcessGroupNCCLFT 的互動

### 3.1 標準 collective 路徑（正常情況）

```
DDP/FSDP
  │
  ▼ allreduce() → allreduce_impl()
     │
     ├── shadow_pre lambda (pre-hook，在 ncclAllReduce 之前)
     │     ・get_or_allocate_shadow_context(tensor)
     │       → 從 free_shadow_bufs_ 取或配置 pinned CPU ShadowContext
     │     ・ctx.original_input = tensor（供 replay 使用）
     │     ・ctx.reduce_op = opts.reduceOp（供 replay 使用）
     │     ・in_flight_shadow_bufs_[seq] = ctx
     │     ・compute_done.record → compute_done.block(shadow_copy_stream_)
     │     ・ctx.buffer.copy_(tensor, non_blocking=true)  ← D2H
     │     ・ctx.copy_event->record(shadow_copy_stream_)  ← per-seq 精準事件
     │
     ├── collective() ──→ ncclAllReduce(input, output, comm, ncclStream)
     │                       NCCL 底層執行 ring/tree all-reduce
     │
     ├── ncclEndEvent_->record(ncclStream)
     │
     └── workEnqueue() ──→ Watchdog 監控
                           └── 完成後 GC: in_flight_shadow_bufs_[seq] 放回 free pool
```

### 3.2 NCCL Fault Callback 路徑（NIC 故障時）

```
NIC 硬體故障
  │
  ▼ NCCL 底層 progress thread 偵測到 IB 事件
     │
     ├── 修改過的 NCCL (custom fork):
     │     ncclIbResiliencyHandleDeviceFailure()
     │     ncclIbResiliencyProbeHandleCompletionEvent()
     │     → ncclRemoteError 替代原本的 ncclSystemError
     │
     ├── nccl_ft_trigger_fault() (NCCL custom fork 新增)
     │     → 呼叫已註冊的 nccl_ft_global_fault_callback(dev_idx)
     │
     └── nccl_ft_global_fault_callback(dev_idx)   [C 語言橋樑函式]
           │
           ├── 鎖定 g_ft_pg_mutex
           └── 對所有 g_ft_pg_instances 呼叫 pg->trigger_fault_proposal(dev_idx)
                 │
                 └── local_hardware_fault_mask_.fetch_or(1<<dev_idx)
                     pending_shadow_seq_.compare_exchange (once)
```

**重要：`local_hardware_fault_mask_` 只由 NCCL callback 路徑（`trigger_fault_proposal`）寫入，Watchdog 不寫入（Fix A）。**

### 3.3 NCCL Callback 註冊

```cpp
// 在第一次 collective() 呼叫時（lazy init），每 rank 各自執行
if (!ft_disabled_ && local_nvlink_comm_ == nullptr && !nvlink_init_attempted_) {
    ncclCommRegisterFaultCallback(ncclComm->getNcclComm(),
                                  nccl_ft_global_fault_callback);
    initLocalNvlinkComm();  // try-catch：NIC 在首次 collective 時故障仍可繼續
}
// 每個 rank 在自己的 ncclComm_t 上獨立呼叫，不互相廣播（Fix 1）
```

### 3.4 NCCL NIC 黑名單 API

```cpp
// 在 rebuild_shadow_ping_pong_topology() 中呼叫（NCCL custom fork 新增）
ncclCommBanNic(dev_idx);   // 讓後續 ncclCommInitRank 跳過此 NIC
ncclCommBanNicReset();     // 恢復（目前實作中不在重建後重置，詳見待確認項目）
```

### 3.5 IntraNodeComm 快速路徑（降級時跳過）

```cpp
// allreduce() 中：降級模式下跳過 IntraNodeComm（Bug 6 fix）
if (opts.reduceOp == ReduceOp::SUM && !is_degraded_) {
    // 嘗試 IntraNodeComm fast-path（NVSwitch / PCIe 直連）
}
// is_degraded_ = true 時，直接走 allreduce_impl → collective
// 原因：IntraNodeComm 繞過了 shadow_pre hook 和 collective() 的容錯邏輯
```

---

## 四、通訊器管理

### 4.1 三種通訊器

| 通訊器 | 成員變數 | 用途 | 建立時機 |
|--------|---------|------|---------|
| **全域通訊器** | `devNCCLCommMap_[deviceKey]` | 正常訓練的所有 collective | 第一次 collective 時 lazy init |
| **本地 NVLink 通訊器** | `local_nvlink_comm_` | Shadow Ping-Pong 的 Step 1/4（faulty ↔ proxy NVLink 傳輸）| 第一次 collective 時 lazy init，由 TCPStore 協商 per-node ncclUniqueId |
| **代理全域通訊器** | `proxy_global_comm_` | 降級模式的跨節點 AllReduce（排除故障 rank）| `rebuild_shadow_ping_pong_topology()` 建立 |

### 4.2 通訊器建立流程

```
initNCCLComm()
  │
  ├── 1. 生成/交換 ncclUniqueId（透過 TCPStore broadcastUniqueNCCLID 或 allgatherUniqueNCCLIDs）
  ├── 2. NCCLFTComm::create() / NCCLFTComm::create_scalable() / NCCLFTComm::split()
  ├── 3. 建立 NCCL stream (ncclStreams_[deviceKey]) 和 CUDA event (ncclEvents_[deviceKey])
  ├── 4. 移入 devNCCLCommMap_ cache
  └── 5. 若啟用 TENSOR_REGISTER_ALLOCATOR_HOOK，將現有 CUDA segments 全部 register 到新 comm
```

### 4.3 降級通訊器重建（rebuild_shadow_ping_pong_topology）

```
1. 快照 faulty_local_devs_（已由 side-car 聚合 2PC 後的結果）
2. 計算 proxy_comm_size_ = size_ - num_faulty_per_node * num_nodes
3. 計算每個 rank 的 proxy_comm_rank_
4. 對所有 faulty_devs 呼叫 ncclCommBanNic(d)（全部節點都呼叫，對稱遮蔽）
5. 健康 rank (is_faulty == false)：
     a. TCPStore rendezvous 新的 ncclUniqueId (key: NCCL_FT_PROXY_ID_<round>)
     b. NCCLFTComm::create(proxy_comm_size_, proxy_comm_rank_, proxyId, device_idx, config)
     c. proxy_global_comm_ = 新建的 comm
6. 故障 rank (is_faulty == true)：proxy_global_comm_ = nullptr
7. proxy_comm_ready_.store(true)
```

---

## 五、Shadow Ping-Pong 容錯機制

### 5.1 ShadowContext 結構與 Shadow Buffer 管理

每次 AllReduce 的 `shadow_pre` hook 在 NCCL stream 執行之前，將梯度 tensor 非同步複製到 pinned CPU memory，並記錄完整的 replay 所需資訊：

```cpp
// 定義於 ProcessGroupNCCLFT.hpp（行 1181 附近）
struct ShadowContext {
    at::Tensor buffer;           // pinned CPU memory（D2H 備份）
    std::shared_ptr<at::cuda::CUDAEvent> copy_event;         // per-seq D2H 完成事件
    std::shared_ptr<at::cuda::CUDAEvent> replayed_end_event; // replay kernel 完成事件
    at::Tensor original_input;   // GPU tensor 參照（供 H2D restore 使用）
    at::Tensor original_output;  // GPU tensor 參照（AllReduce in-place 通常同一個）
    c10d::ReduceOp reduce_op;    // ReduceOp（確保 replay 使用正確操作，不預設為 SUM）
};
```

```
shadow_pre lambda (每次 AllReduce)：
  ├── get_or_allocate_shadow_context(tensor)
  │     └── 優先從 free_shadow_bufs_ (ring pool) 取，沒有才 pin_memory() 配置
  ├── ctx.original_input / ctx.original_output = tensor（in-place AllReduce 同一個）
  ├── ctx.reduce_op = opts.reduceOp
  ├── in_flight_shadow_bufs_[seq] = ctx
  ├── compute_done.record(compute_stream)
  ├── compute_done.block(shadow_copy_stream_)    （GPU-side fence）
  └── ctx.buffer.narrow(0,0,tensor.numel()).copy_(tensor, non_blocking=true)
      ctx.copy_event->record(shadow_copy_stream_) （per-seq 精準紀錄）

Watchdog 成功 GC 後：
  └── ctx.original_input = {}   （釋放 GPU VRAM 參照）
      ctx.original_output = {}
      free_shadow_bufs_.push_back(ctx)
      in_flight_shadow_bufs_.erase(seq)
```

### 5.2 Execute Shadow AllReduce（四角色）

故障後，`execute_shadow_allreduce()` 依各 rank 的角色執行：

```
角色判斷：
  local_rank = rank_ % localDeviceCount_
  is_faulty  = faulty_local_devs_.count(local_rank) > 0
  is_proxy   = 本 rank 是某個 faulty rank 的代理（findProxy 計算）
  healthy    = 既不 faulty 也不 proxy

┌─────────────────────────────────────────────────────────────┐
│ FAULTY rank：NIC 故障，透過 NVLink 借道 proxy               │
│   Step 1: ncclSend(input → proxy, via local_nvlink_comm_)   │
│   Step 4: ncclRecv(output ← proxy, via local_nvlink_comm_)  │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ PROXY rank：健康，代表 faulty rank 參與跨節點 AllReduce      │
│   Step 1: ncclRecv(ward_buf ← faulty, via local_nvlink_comm_)
│   Step 2: output = input + sum(ward_bufs)  （pre-aggregate）│
│   Step 3: ncclAllReduce(output, proxy_global_comm_)         │
│           （ReduceOp::AVG → ncclSum 再除以原始 size_）      │
│   Step 4: ncclSend(output → faulty, via local_nvlink_comm_) │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ HEALTHY rank：直接參與降級通訊                               │
│   ncclAllReduce(input, output, proxy_global_comm_)           │
│   （ReduceOp::AVG → 同樣手動除以原始 size_）                │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 2PC 協商流程（side-car thread）

```
TCPStore Key Namespace（Round R）：
  NCCL_FT_PROPOSE_<rank>_<R>   — 每個 rank 各自寫，格式：
                                  "PROPOSE:<round>:<node_id>:0x<mask_hex>:<shadow_seq>"
  NCCL_FT_SS_AGREED_<R>        — rank 0 寫 agreed_shadow_seq
  NCCL_FT_COMMIT_<R>           — rank 0 寫 "1" 觸發 commit

Phase 1（收集）：
  ├── 任一 rank 偵測到 fault_mask ≠ 0 → 寫入自己的 PROPOSE key
  ├── 其他 rank 偵測到有 PROPOSE key 存在 → 也寫入自己的 PROPOSE key（mask=0x0）
  └── 等待所有 size_ 個 PROPOSE keys 就緒
       ├── 聚合 agg_fault_mask（所有 p_fmask OR）
       └── 計算 agreed_ss = min(所有非 UINT64_MAX 的 shadow_seq)

Phase 2（決策）：
  ├── rank 0：寫 NCCL_FT_SS_AGREED_<R> = agreed_ss
  │           寫 NCCL_FT_COMMIT_<R> = "1"
  └── 所有 rank：輪詢直到 COMMIT key 出現
                  ├── committed_shadow_seq_.store(agreed_ss, release)
                  ├── globalComm->abort()（強制喚醒卡死的主執行緒）
                  └── final_commit_op_.store(cur_round + 1, release)

主執行緒感知（透過 wait() 攔截點 1）：
  └── WorkNCCLFT::wait() 偵測 final_commit_op_ ≠ 0
        → 呼叫 recover_and_replay_inflight_ops()
```

---

## 六、恢復流程：recover_and_replay_inflight_ops()

這是目前架構的**主要恢復入口**，在 `WorkNCCLFT::wait()` 的攔截點 1 觸發（行 907-909）。

### 6.1 函式流程（行 5440-5507）

```
recover_and_replay_inflight_ops()
  │
  ├── 0. recovery_mutex_ 防止重入
  ├── 1. 快速返回條件：is_degraded_ && rollback_done_ 表示已恢復過
  │
  ├── 2. 強制 abort 舊的全域 comm（確保 GPU kernel 完全停止）
  │
  ├── 3. rebuild_shadow_ping_pong_topology()
  │      → proxy_global_comm_ 就緒
  │      → is_degraded_ = true
  │
  ├── 4. 讀取 agreed_ss = committed_shadow_seq_.load()
  │
  ├── 5. 收集所有 seq >= agreed_ss 的 in-flight shadow buffers
  │      → 排序後依序重播
  │
  ├── 6. 對每個需要重播的 seq：
  │      a. ctx.copy_event->synchronize()  （確保 D2H 完成）
  │      b. H2D restore：ctx.buffer → ctx.original_input (via shadow_copy_stream_)
  │      c. restore_done.block(ncclStream)（GPU-side fence）
  │      d. execute_shadow_allreduce(ctx.original_input, ctx.original_output,
  │                                  ncclStream, ctx.reduce_op)
  │      e. ctx.replayed_end_event->record(ncclStream)
  │
  ├── 7. rollback_done_ = true
  └── 8. final_commit_op_.store(0)  （解鎖 side-car 進入下一輪）
```

### 6.2 wait() 攔截點 2（行 914-938）

```
WorkNCCLFT::wait() 攔截點 2：
  條件：!ft_disabled_ && is_degraded_ && seq_ >= committed_shadow_seq_

  ├── 從 in_flight_shadow_bufs_ 查找 this->seq_
  ├── 找到（is_replayed = true）：
  │     a. ctx.replayed_end_event->block(currentStream)  （等待 replay kernel 完成）
  │     b. stashed_for_allocator_safety_->unstash()       （釋放 Tensor 參照）
  │     c. setException(nullptr)                          （清除 Watchdog 可能設的 exception）
  │     d. return true  （透明成功，DDP 不感知錯誤）
  └── 找不到（seq 尚未 replay 或 GC 已清除）：
        fall-through → synchronize()（正常路徑）
```

---

## 七、collective() 中的執行路徑（四條，含舊邏輯）

`collective()` 內部目前**有兩套邏輯並存**，理解其順序非常重要：

### 7.1 前置分流（行 5138-5169）：最新路徑

這是目前**主要的執行路徑切換點**，位於 `pre()` 和 `fn()` 之間：

```cpp
pre(ncclStream, work);  // shadow_pre：D2H 快照
ncclComm_t comm = ncclComm->getNcclComm();

// 分流 1：降級模式
if (C10_UNLIKELY(this->is_degraded_)) {
    if (opType == OpType::ALLREDUCE) {
        execute_shadow_allreduce(inputs[0], outputs[0], ncclStream,
                                 current_shadow_reduce_op_);
    } else {
        // Non-AllReduce in degraded mode：fallback（可能失敗）
    }
    work->ncclComm_ = proxy_global_comm_ ? proxy_global_comm_ : local_nvlink_comm_;
} else {
    // 分流 2：正常訓練路徑
    try {
        fn(inputs[0], outputs[0], comm, ncclStream);
    } catch (const ::c10::NCCLFaultToleranceError& e) {
        if (!ft_disabled_) {
            // 吞掉同步錯誤，設定 work 欄位，return work
            // 注意：future_ 在這裡未設置（Bug 2，見待確認項目）
            work->ncclEndEvent_->record(ncclStream);
            work->ncclComm_ = ncclComm;
            return work;
        }
        throw;
    }
    work->ncclComm_ = ncclComm;
}
```

### 7.2 舊的 2PC barrier + 切換點 A/B（行 5203-5343）：仍然存在，但路徑已被前置分流攔截

**注意：** 由於 `is_degraded_ = true` 的情況在行 5138-5169 已被處理（`execute_shadow_allreduce` 完成後繼續往下），而 `recover_and_replay_inflight_ops()` 設定 `rollback_done_ = true`，這段舊邏輯的行為如下：

```
行 5203：if (!ft_disabled_ && C10_UNLIKELY(rollback_done_)) { ... }
  ├── rollback_done_ = true 時（由 recover_and_replay_inflight_ops 設定）
  │     → 若無 shadow_replay_pending_，重置 rollback_done_ = false
  │     → 目前架構下 shadow_replay_pending_ 通常為 false，所以此處立刻重置

行 5218：if (!ft_disabled_ && C10_UNLIKELY(shadow_replay_pending_) && ...) { ... }
  ├── shadow_replay_pending_ 由舊的 REBUILD path 設定，目前架構下不會被設定
  │     → 這個切換點 A（Replay 路徑）在目前架構下不會觸發

行 5250：if (C10_UNLIKELY(this->is_degraded_) && !ran_shadow_replay) { ... }
  ├── is_degraded_ = true 時，這裡也會呼叫 execute_shadow_allreduce
  │     → 與行 5142-5148 的前置分流形成**重複執行**（Bug 1，見待確認項目）
  │     → ran_shadow_replay 在前置分流中沒有設定，此處 !ran_shadow_replay = true
  └── 這是目前架構的已知問題
```

### 7.3 完整流程圖

```
collective()
  │
  ├─ 1. seqCollective_++ / 取或建立 ncclComm
  ├─ 2. [Lazy FT init - 首次執行]
  │      ├── ncclCommRegisterFaultCallback(comm, callback)
  │      └── initLocalNvlinkComm() [try-catch]
  │
  ├─ 3. pre(ncclStream, work)  ← shadow_pre lambda（D2H 快照）
  │
  ├─ 4. [前置分流，行 5138-5169]
  │      ├── is_degraded_ = true  → execute_shadow_allreduce()
  │      │     work->ncclComm_ = proxy_global_comm_ ?: local_nvlink_comm_
  │      └── is_degraded_ = false → fn() with try-catch(NCCLFaultToleranceError)
  │            → 吞掉例外，記錄 end event，return work（不執行後續邏輯）
  │
  ├─ 5. [舊邏輯，行 5203-5343，仍存在但已不是主路徑]
  │      ├── [切換點 A] shadow_replay_pending_（目前通常不觸發）
  │      ├── [切換點 B] is_degraded_ && !ran_shadow_replay（與步驟 4 重複！）
  │      └── [Normal else] 若步驟 4 已處理，此處 is_degraded_=false → 再執行 fn()
  │           → 正常訓練時此處才是真正執行 fn() 的地方
  │
  ├─ 6. post(ncclStream, work)
  ├─ 7. ncclEndEvent_->record(ncclStream)
  ├─ 8. [Bug 10 fix] work->ncclComm_ 指向實際使用的 comm
  └─ 9. workEnqueue(work) → Watchdog 監控
```

**⚠ 重要說明：** 步驟 4 的前置分流（行 5138-5169）目前是**提前 return** 還是 **fall-through** 到步驟 5，取決於：
- 正常路徑（`is_degraded_ = false`）：若 `fn()` 無 exception，fall-through 到步驟 5（舊邏輯 else 分支）執行原生 fn()。**兩段 fn() 都不執行是不對的**，實際上步驟 4 的 else 分支只有 try-catch wrapper，fn() 本身在步驟 5 的 else 分支執行。
- 降級路徑（`is_degraded_ = true`）：步驟 4 執行 `execute_shadow_allreduce`，fall-through 到步驟 5（切換點 B）**再次執行** `execute_shadow_allreduce`（Bug 1）。

---

## 八、狀態變數對照表

| 變數 | 類型 | Normal | Replay/Degraded | 說明 |
|------|------|--------|-----------------|------|
| `is_degraded_` | `bool` | false | true | 拓撲已降級，一旦設為 true 不回到 false（除非重啟） |
| `rollback_done_` | `atomic<bool>` | false | true（recover 後） | 防止 recover_and_replay 重入 |
| `shadow_replay_pending_` | `bool` | false | 通常 false | 舊 REBUILD path 設定，目前架構下通常不觸發 |
| `final_commit_op_` | `atomic<uint64_t>` | 0 | 非 0（side-car 設） | wait() 攔截點 1 的觸發訊號，recover 完成後重置為 0 |
| `proxy_comm_ready_` | `atomic<bool>` | false | true（rebuild 後） | rebuild_shadow_ping_pong_topology 完成後設為 true |
| `committed_shadow_seq_` | `atomic<uint64_t>` | UINT64_MAX | agreed_ss | 2PC 協商出的最後安全 checkpoint seq |

---

## 九、Bug 10 修正：work->ncclComm_ 的正確對應

執行路徑確定後，`work->ncclComm_` 必須指向**實際使用的 comm**，否則 Watchdog 的 `checkAndSetException()` 會查詢已死亡的 comm 造成誤報：

```
ran_shadow_replay || (is_degraded_ && ALLREDUCE)（行 5369-5394）：
  ├── is_faulty_rank == true  → work->ncclComm_ = local_nvlink_comm_
  ├── is_faulty_rank == false → work->ncclComm_ = proxy_global_comm_
  └── fallback（comm 為 null） → work->ncclComm_ = ncclComm  [warning]

其他情況（normal 或 non-AllReduce degraded）：
  └── work->ncclComm_ = ncclComm  [原始 comm]
```

---

## 十、WorkNCCLFT 生命週期

```
initWork() → 建立 WorkNCCLFT，分配 ncclStartEvent_ / ncclEndEvent_
    │
    ├── 加入 workMetaList_（workEnqueue）
    │
    ├── [主執行緒] wait()
    │     ├── 攔截點 1：final_commit_op_ > 0 → recover_and_replay_inflight_ops()
    │     ├── 攔截點 2：is_degraded_ && seq >= committed_shadow_seq_
    │     │     → wait replayed_end_event → unstash → clearException → return true
    │     └── 原生路徑：synchronize() → ncclEndEvent_->block(currentStream)
    │           stashed_for_allocator_safety_->unstash()
    │           若有 exception：handleException(SkipCleanUpFT) [不 rethrow]
    │
    └── [Watchdog]
          ├── FT enabled + exception：清除 exception，記錄 ncclEndEvent_，erase work
          ├── FT disabled + exception：正常錯誤處理（rethrow）
          ├── early-abort（final_commit_op_ > 0）：強制記錄 ncclEndEvent_
          └── 成功完成：
                FlightRecorder retire
                in_flight_shadow_bufs_[seq] GC → ctx.original_input/output 清空
                → free_shadow_bufs_.push_back(ctx)
                erase from workMetaList_
```

---

## 十一、記憶體與串流管理

| 串流 | 用途 |
|------|------|
| `ncclStreams_[deviceKey]` | NCCL collective 執行串流 |
| `shadow_copy_stream_` | Shadow buffer D2H（GPU→CPU）及 H2D（CPU→GPU）複製串流 |
| current CUDA stream | 使用者計算串流（AllReduce sync mode 時也是 NCCL 串流） |

| 事件 | 類型 | 用途 |
|------|------|------|
| `ncclStartEvent_` | per-Work | 記錄 collective 開始（timing 啟用時） |
| `ncclEndEvent_` | per-Work | 記錄 collective 結束；block() 用於 wait() 同步 |
| `ShadowContext::copy_event` | per-seq shared_ptr | 記錄 D2H shadow copy 完成（per-seq 精準） |
| `ShadowContext::replayed_end_event` | per-seq shared_ptr | 記錄 replay kernel 完成；wait() 攔截點 2 block 用 |

---

## 十二、環境變數與設定

| 環境變數 | 預設值 | 說明 |
|---------|--------|------|
| `NCCL_FT_DISABLE` | 0 | 設為 1 時完全停用容錯機制 |
| `TORCH_NCCLFT_BLOCKING_WAIT` | false | 阻塞模式（停用 Watchdog） |
| `TORCH_NCCLFT_ASYNC_ERROR_HANDLING` | 3 (SkipCleanUpFT) | 錯誤處理模式 |
| `TORCH_NCCLFT_HEARTBEAT_TIMEOUT_SEC` | 480 | Watchdog 心跳超時 |
| `TORCH_NCCLFT_TRACE_BUFFER_SIZE` | 2000 | Flight Recorder buffer 大小 |
| `TORCH_NCCLFT_ENABLE_MONITORING` | true | 啟用 HeartbeatMonitor |
| `TORCH_NCCLFT_DUMP_ON_TIMEOUT` | true | 超時時 dump Flight Recorder |
| `TORCH_NCCLFT_NAN_CHECK` | false | 啟用 NaN 檢查 |
| `TORCH_NCCLFT_CUDA_EVENT_CACHE` | true | 啟用 CUDA event cache |
| `TORCH_NCCLFT_USE_COMM_NONBLOCKING` | auto | 使用 NCCL non-blocking mode |

---

## 十三、故障完整時序（目前實作）

```
t=0:  NIC mlx5_0 硬體故障
t=1:  NCCL progress thread → ncclRemoteError → nccl_ft_global_fault_callback(dev_idx=0)
t=2:  trigger_fault_proposal() → local_hardware_fault_mask_ |= 1<<0
                                  pending_shadow_seq_ = shadow_seq_ (once)
t=3:  Side-car 偵測到 fault_mask ≠ 0，寫 PROPOSE key
t=4:  其他 rank 偵測到有 PROPOSE → 寫自己的 PROPOSE (mask=0x0)
t=5:  所有 PROPOSE 就緒，rank 0 計算 agreed_ss = min(shadow_seqs)
t=6:  rank 0 寫 COMMIT key
t=7:  Side-car：globalComm->abort()（喚醒主執行緒的 ncclAllReduce hang）
t=8:  Side-car：final_commit_op_.store(cur_round + 1)
t=9:  Watchdog early-abort：偵測 final_commit_op_ > 0，force-record ncclEndEvent_
t=10: 主執行緒 WorkNCCLFT::wait() 感知 final_commit_op_ ≠ 0
        → recover_and_replay_inflight_ops()
            → abort 舊 globalComm
            → rebuild_shadow_ping_pong_topology() → proxy_global_comm_ 就緒
            → 收集 seq >= agreed_ss 的所有 in-flight shadow bufs
            → 依序 H2D restore + execute_shadow_allreduce() + record replayed_end_event
            → rollback_done_ = true, final_commit_op_.store(0)
t=11: 同一 wait() call：攔截點 2
        → ctx.replayed_end_event->block(currentStream)（等待 replay kernel）
        → unstash, clearException, return true
t=12: 後續 AllReduce：collective() 前置分流 is_degraded_=true
        → execute_shadow_allreduce() 走降級路徑
        → 訓練繼續 ✓
```

---

## 十四、已知問題（待解決）

### Bug 1（嚴重）：collective() 中降級路徑的 execute_shadow_allreduce 被呼叫兩次

**位置：** 行 5142-5148（前置分流）和行 5250-5278（舊切換點 B）

**現象：** `is_degraded_ = true` 時，前置分流執行 `execute_shadow_allreduce` 後 fall-through，舊切換點 B 的條件 `is_degraded_ && !ran_shadow_replay` 也成立（`ran_shadow_replay` 在前置分流中未設定），導致同一個 AllReduce 被執行兩次。

**影響：** 梯度數值錯誤（double reduce），或 NCCL 序列混亂。

### Bug 2（中）：early return 路徑的 future_ 未設置

**位置：** 行 5157-5165（catch NCCLFaultToleranceError）

**現象：** 在前置分流的 catch 中，`return work` 前未設置 `work->future_`。若 DDP 呼叫 `getFuture()` 將崩潰。（舊切換點 B 的對應 catch，行 5307-5340，有完整設置。）

### Bug 3（中）：recover_and_replay_inflight_ops 的 seq 邊界

**位置：** 行 5465：`if (seq >= agreed_ss)`

**現象：** `agreed_ss` 是已成功完成的最後一個 seq。若 `seq == agreed_ss` 的那個 AllReduce 已被 Watchdog GC 清除（成功完成），`in_flight_shadow_bufs_` 中找不到它，replay 會跳過不影響。但若 `seq == agreed_ss` 的 AllReduce **尚未**被 GC，會被 replay 兩次（一次已成功，一次又 replay）。正確邏輯應為 `seq > agreed_ss`。

### Bug 4（低）：get_or_allocate_shadow_context 在鎖內呼叫 pin_memory()

**位置：** 行 4934, 4948

**現象：** `pin_memory()` = `cudaHostAlloc`（同步 CUDA call）在 `shadow_buf_mutex_` 鎖住狀態下執行，可能 stall Watchdog GC（低機率，ring pool 命中時不會發生）。
