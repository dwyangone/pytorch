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
│  ・故障後在 2PC barrier spin-wait，直到 final_commit_op_ ≠ 0 │
│  ・REBUILD path：呼叫 rebuild_shadow_ping_pong_topology()     │
│                   → 還原 shadow buffer → replay AllReduce    │
└─────────────────────┬────────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────────┐
│  Watchdog (pt_nccl_watchdg)                                   │
│  ・每 100ms 掃描 workMetaList_                                │
│  ・偵測 exception 後：                                        │
│    - FT 啟用：清除 exception + 記錄 ncclEndEvent_             │
│      （不寫 fault mask，不 rethrow）                          │
│    - FT 停用：正常錯誤處理路徑                                 │
│  ・early-abort：若 final_commit_op_ > 0，強制結束飛行中的 op  │
│  ・成功完成 AllReduce 後：釋放 in_flight_shadow_bufs_[seq]    │
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
│    - 設定 final_commit_op_ ≠ 0（喚醒主執行緒 barrier）        │
│  ・等主執行緒重置 final_commit_op_ = 0 後，進入下一輪         │
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
     ├── shadow_pre lambda (pre-hook, 在 ncclAllReduce 之前)
     │     • 從 free_shadow_bufs_ 取或配置 pinned CPU buffer
     │     • 在 shadow_copy_stream_ 上執行 GPU→CPU 非同步 D2H copy
     │     • 寫入 in_flight_shadow_bufs_[seq]，更新 shadow_seq_
     │
     ├── collective() ──→ ncclAllReduce(input, output, comm, ncclStream)
     │                       NCCL 底層執行 ring/tree all-reduce
     │
     ├── ncclEndEvent_->record(ncclStream)
     │
     └── workEnqueue() ──→ Watchdog 監控
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
// 在第一次 collective() 呼叫時（lazy init）
ncclCommRegisterFaultCallback(ncclComm->getNcclComm(),
                              nccl_ft_global_fault_callback);
// 每個 rank 各自在自己的 ncclComm_t 上獨立呼叫，不互相廣播
```

### 3.4 NCCL NIC 黑名單 API

```cpp
// 在 rebuild_shadow_ping_pong_topology() 中呼叫
// NCCL custom fork 新增的 API，作用域為 process-level global
ncclCommBanNic(dev_idx);  // 讓後續 ncclCommInitRank 跳過此 NIC
```

### 3.5 IntraNodeComm 快速路徑（降級時跳過）

```cpp
// allreduce() 中：降級模式下跳過 IntraNodeComm
if (opts.reduceOp == ReduceOp::SUM && !is_degraded_) {
    // 嘗試 IntraNodeComm fast-path（NVSwitch / PCIe 直連）
}
// is_degraded_ = true 時，直接走 allreduce_impl → collective
```

---

## 四、通訊器管理

### 4.1 三種通訊器

| 通訊器 | 成員變數 | 用途 | 建立時機 |
|--------|---------|------|---------|
| **全域通訊器** | `devNCCLCommMap_[deviceKey]` | 正常訓練的所有 collective | 第一次 collective 時 lazy init |
| **本地 NVLink 通訊器** | `local_nvlink_comm_` | Shadow Ping-Pong 的 Step 1/4（faulty ↔ proxy）| 第一次 collective 時 lazy init，由 TCPStore 協商 per-node ID |
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
     a. 再次呼叫 ncclCommBanNic(d)
     b. TCPStore rendezvous 新的 ncclUniqueId (key: NCCL_FT_PROXY_ID_<round>)
     c. NCCLFTComm::create(proxy_comm_size_, proxy_comm_rank_, proxyId, device_idx, config)
     d. proxy_global_comm_ = 新建的 comm
6. 故障 rank (is_faulty == true)：proxy_global_comm_ = nullptr
7. proxy_comm_ready_.store(true)
```

---

## 五、Shadow Ping-Pong 容錯機制

### 5.1 Shadow Buffer 管理

每次 AllReduce 的 `shadow_pre` hook 在 NCCL stream 執行之前，將梯度 tensor 非同步複製到 pinned CPU memory：

```
shadow_pre lambda (每次 AllReduce)：
  ├── get_or_allocate_shadow_buffer(tensor)
  │     └── 優先從 free_shadow_bufs_ (ring pool) 取，沒有才向 OS 申請 pin_memory
  ├── in_flight_shadow_bufs_[seq] = shadow_tensor  (by seq 索引)
  ├── compute_done.record(compute_stream)
  ├── compute_done.block(shadow_copy_stream_)       (GPU-side fence)
  └── shadow_tensor.copy_(tensor, non_blocking=true) on shadow_copy_stream_
      shadow_copy_event_.record(shadow_copy_stream_) (可查詢是否完成)
      shadow_seq_ = seq

Watchdog 成功回報後（gc）：
  └── in_flight_shadow_bufs_[seq] 移回 free_shadow_bufs_
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
│   Step 1: ncclSend(input → proxy, via nvlink_comm)          │
│   Step 4: ncclRecv(output ← proxy, via nvlink_comm)         │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ PROXY rank：健康，代表 faulty rank 參與跨節點 AllReduce      │
│   Step 1: ncclRecv(ward_buf ← faulty, via nvlink_comm)      │
│   Step 2: output = input + sum(ward_bufs) （pre-aggregate） │
│   Step 3: ncclAllReduce(output, proxy_global_comm_)         │
│           （ReduceOp::AVG 用 ncclSum 再除以原始 size_）     │
│   Step 4: ncclSend(output → faulty, via nvlink_comm)        │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ HEALTHY rank：直接參與降級通訊                               │
│   ncclAllReduce(input, output, proxy_global_comm_)           │
│   （ReduceOp::AVG 同樣手動除以原始 size_）                  │
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
                  ├── globalComm->abort() （強制喚醒卡死的主執行緒）
                  └── final_commit_op_.store(cur_round + 1, release)

主執行緒感知：
  └── collective() 的 2PC barrier 偵測 final_commit_op_ ≠ 0
        ├── DRAIN path：current_op > agreed_ss + 1
        │     → 跳過 fn()，返回空 Work
        └── REBUILD path：current_op == agreed_ss + 1
              → rebuild_shadow_ping_pong_topology()
              → shadow buffer restore (H2D)
              → execute_shadow_allreduce (replay)
```

---

## 六、正常 ↔ Replay ↔ Degraded 三態切換機制

這是 `ProcessGroupNCCLFT` 的核心設計，也是你說的「兩層防線」。`collective()` 內部的邏輯完美區隔了三種執行狀態，且每層防線的條件**互斥且完備**。

### 6.1 三種狀態定義

| 狀態 | 條件 | 使用的 comm |
|------|------|------------|
| **Normal（正常）** | `is_degraded_ == false` | `devNCCLCommMap_`（全域原始 comm） |
| **Replay（重播）** | `is_degraded_ == true` && `shadow_replay_pending_ == true` | `local_nvlink_comm_` / `proxy_global_comm_` |
| **Degraded（降級後續）** | `is_degraded_ == true` && `shadow_replay_pending_ == false` | `local_nvlink_comm_` / `proxy_global_comm_` |

### 6.2 第一層防線：2PC Commit-Aware Barrier（故障感知閘道）

**位置：** [`collective()` 行 5106–5279](torch/csrc/distributed/c10d/ProcessGroupNCCLFT.cpp:5106)

**觸發條件：** `!ft_disabled_ && !rollback_done_ && final_commit_op_ > 0`

這層防線在 `pre()` 和 `fn()` 執行之前運行，負責**攔截並分類**每一個到達的 op。它根據 `current_op`（已 bump 的 seqCollective_）相對於 `agreed_ss`（2PC 協商出的最後安全 checkpoint）的位置，產生三條分支：

```
第一層防線內部邏輯（guard: !rollback_done_）
│
├─ [agreed_ss == UINT64_MAX]  故障發生在任何 AllReduce 完成之前
│    → 拓撲重建 only（rebuild_shadow_ping_pong_topology）
│    → is_degraded_ = true, rollback_done_ = false（不需要 replay）
│    → final_commit_op_ = 0（解鎖 side-car）
│    → 直接 fall-through 到 fn()，以 degraded comm 執行（見第二層防線）
│
├─ [current_op > agreed_ss + 1]  DRAIN 路徑
│    → 這個 op 是 DDP 故障期間超發的「廢棄 op」，必須丟棄
│    → seqCollective_-- （退回 seq，維持一致性）
│    → ncclEndEvent_->record（讓 wait() 立刻返回）
│    → future_->markCompleted（讓 getFuture() 不崩潰）
│    → return work  ←── 提前返回，fn() 完全不執行
│
└─ [current_op == agreed_ss + 1]  REBUILD 路徑
     → ncclStream.synchronize()（確保舊 GPU kernel 完全停止）
     → rebuild_shadow_ping_pong_topology()
     → is_degraded_ = true
     → seqCollective_ = agreed_ss（對齊 seq，讓後續 replay 路徑的 -- 落在正確位置）
     → H2D shadow restore：in_flight_shadow_bufs_[agreed_ss] → inputs[0]
     → shadow_restore_event_.block(ncclStream)（GPU-side fence）
     → shadow_replay_pending_ = true
     → rollback_done_ = true  ←── 設定後，下一個 op 跳過本防線
     → final_commit_op_ = 0（解鎖 side-car）
     → fall-through，讓第二層防線的 Replay 路徑接手
```

**`rollback_done_` 的作用**：防止第二個及後續的 degraded op 再次進入 REBUILD 邏輯（因為拓撲只需重建一次）。它在 DRAIN 之後繼續允許第一層正常運作，但 REBUILD 執行後會鎖閉自己，直到 replay 完成後才由第二層防線重置。

### 6.3 第二層防線：執行路徑選擇器（切換點 A + 切換點 B）

**位置：** [`collective()` 行 5372–5499](torch/csrc/distributed/c10d/ProcessGroupNCCLFT.cpp:5372)

第二層防線在 `pre()` 之後運行，負責選擇**實際執行哪條 kernel 路徑**。它由兩個互斥的條件判斷構成：

#### 切換點 A：Replay 路徑（行 5374）

```cpp
// 條件：三個都要同時成立
if (!ft_disabled_ &&
    C10_UNLIKELY(shadow_replay_pending_) &&   // 第一層防線設定
    opType == OpType::ALLREDUCE &&             // 只有 AllReduce 能 replay
    proxy_comm_ready_.load(std::memory_order_acquire))  // topology 重建完成
{
    seqCollective_--;  // 退回 bump，replay 不是新 op
    execute_shadow_allreduce(inputs[0], outputs[0], ncclStream, ...);
    shadow_replay_pending_ = false;
    ran_shadow_replay = true;
    rollback_done_ = false;  // 重置，為下一個 fault round 準備
}
```

進入此路徑後，`ran_shadow_replay = true`，**切換點 B 的外層 `if` 為 false，不進入**。兩個切換點完全互斥。

#### 切換點 B：Degraded 或 Normal 路徑（行 5406）

```cpp
// 切換點 B 的完整條件樹
if (C10_UNLIKELY(is_degraded_) && !ran_shadow_replay) {
    // ── Degraded 分支 ──────────────────────────────────
    if (opType == ALLREDUCE && proxy_comm_ready_) {
        execute_shadow_allreduce(...);  // 走 Shadow Ping-Pong 降級代傳
    } else {
        fn(inputs[0], outputs[0], comm, ncclStream);  // Non-AllReduce：沿用原始 comm（可能失敗）
    }
} else {
    // ── Normal 分支（100% 正常訓練走這裡）──────────────
    try {
        fn(inputs[0], outputs[0], comm, ncclStream);  // 原生 ncclAllReduce
    } catch (NCCLFaultToleranceError& e) {
        if (!ft_disabled_) {
            // 吞掉同步錯誤，讓 2PC 恢復機制接手
            work->ncclEndEvent_->record(ncclStream);
            work->future_->markCompleted(...);
            return work;  // 透明返回，Python 不感知
        }
        throw;
    }
}
```

### 6.4 兩層防線的完整狀態機

```
每次 collective() 呼叫時的決策流程：

seqCollective_++  (bump)
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  第一層防線（2PC barrier）                                         │
│  條件：!ft_disabled_ && !rollback_done_ && final_commit_op_ > 0   │
│                                                                   │
│  agreed_ss == UINT64_MAX ────→ TOPO REBUILD ONLY                  │
│                                is_degraded_ = true                │
│                                fall through ──────────────────┐   │
│                                                               │   │
│  current_op > agreed_ss+1 ──→ DRAIN                          │   │
│                                return work (early exit) ──────┼───┼──→ 結束
│                                                               │   │
│  current_op == agreed_ss+1 → REBUILD                         │   │
│                                rebuild_topology()             │   │
│                                H2D restore                    │   │
│                                shadow_replay_pending_=true    │   │
│                                rollback_done_=true            │   │
│                                fall through ──────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
   pre(ncclStream, work)  [shadow_pre：D2H 快照，正常時每次執行]
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  第二層防線（執行路徑選擇）                                        │
│                                                                   │
│  切換點 A：shadow_replay_pending_ && ALLREDUCE && proxy_ready      │
│  ┌──────────────────────────────────────────────────────┐         │
│  │  REPLAY 路徑                                          │         │
│  │  seqCollective_--                                    │         │
│  │  execute_shadow_allreduce()  [用降級 comm]            │         │
│  │  shadow_replay_pending_ = false                      │         │
│  │  ran_shadow_replay = true                            │         │
│  │  rollback_done_ = false  ← 重置，下輪可用            │         │
│  └──────────────────────────────────────────────────────┘         │
│          │ (ran_shadow_replay=true → 切換點 B 不進入)              │
│                                                                   │
│  切換點 B：is_degraded_ && !ran_shadow_replay                      │
│  ┌──────────────────────────────────────────────────────┐         │
│  │  DEGRADED 路徑                                        │         │
│  │  ALLREDUCE & proxy_ready → execute_shadow_allreduce() │         │
│  │  其他 op → fn() on orig comm（fallback，可能失敗）    │         │
│  └──────────────────────────────────────────────────────┘         │
│          │ (is_degraded_=false → 進入 else)                        │
│  ┌──────────────────────────────────────────────────────┐         │
│  │  NORMAL 路徑                                          │         │
│  │  fn()（原生 ncclAllReduce）                           │         │
│  │  catch NCCLFaultToleranceError → 透明吞掉             │         │
│  └──────────────────────────────────────────────────────┘         │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
   post() → ncclEndEvent_->record() → Bug10 comm 修正 → workEnqueue()
```

### 6.5 狀態變數對照表

| 變數 | 類型 | Normal | Replay | Degraded | 說明 |
|------|------|--------|--------|----------|------|
| `is_degraded_` | `bool` | false | true | true | 拓撲已降級，永不回到 false（除非重啟） |
| `shadow_replay_pending_` | `bool` | false | true | false | 由 REBUILD path 設為 true，replay 後清除 |
| `rollback_done_` | `bool` | false | true | false | 防止 REBUILD 重入，replay 完成後清除 |
| `final_commit_op_` | `atomic<uint64_t>` | 0 | 0（rebuild 後重置） | 0 | side-car 寫非零觸發 barrier，主執行緒重置為 0 |
| `proxy_comm_ready_` | `atomic<bool>` | false | true | true | rebuild_topology() 完成後設為 true |
| `ran_shadow_replay` | `bool`（local） | false | true | false | 防止切換點 A 和 B 同時觸發 |

### 6.6 Bug 10 修正：work->ncclComm_ 的正確對應

執行路徑確定後，`work->ncclComm_` 必須指向**實際使用的 comm**，否則 Watchdog 的 `checkAndSetException()` 會查詢已死亡的 comm 造成誤報：

```
ran_shadow_replay || (is_degraded_ && ALLREDUCE)：
  ├── is_faulty_rank == true  → work->ncclComm_ = local_nvlink_comm_
  ├── is_faulty_rank == false → work->ncclComm_ = proxy_global_comm_
  └── fallback（comm 為 null）→ work->ncclComm_ = ncclComm  [warning]

其他情況（normal 或 non-AllReduce degraded）：
  └── work->ncclComm_ = ncclComm  [原始 comm]
```

---

## 七、collective() 主要執行流程（含切換點標注）

```
collective()
  │
  ├─ 1. 取或建立 ncclComm (getNCCLComm / initNCCLComm)
  │
  ├─ 2. [Lazy FT init - 首次執行]
  │      ├── ncclCommRegisterFaultCallback(comm, nccl_ft_global_fault_callback)
  │      └── initLocalNvlinkComm() [try-catch]
  │
  ├─ 3. [第一層防線：2PC Commit-Aware Barrier]
  │      guard: !ft_disabled_ && !rollback_done_ && final_commit_op_ > 0
  │      ├── agreed_ss == UINT64_MAX → TOPO REBUILD ONLY, fall through
  │      ├── current_op > agreed_ss+1 → DRAIN, return work (early exit)
  │      └── current_op == agreed_ss+1 → REBUILD
  │              ncclStream.synchronize()
  │              rebuild_shadow_ping_pong_topology()
  │              H2D restore → shadow_restore_event_.block(ncclStream)
  │              shadow_replay_pending_ = true, rollback_done_ = true
  │
  ├─ 4. shadow_restore_event_ fence（若 shadow_restore_pending_）
  │
  ├─ 5. pre(ncclStream, work)  ← shadow_pre lambda（D2H 快照，正常時每次執行）
  │
  ├─ 6. [第二層防線：執行路徑選擇]
  │      ├── [切換點 A: Replay]
  │      │     shadow_replay_pending_ && ALLREDUCE && proxy_comm_ready_
  │      │     → seqCollective_--, execute_shadow_allreduce()
  │      │     → shadow_replay_pending_=false, ran_shadow_replay=true
  │      │     → rollback_done_=false（重置，為下輪 fault 準備）
  │      │
  │      ├── [切換點 B: Degraded] is_degraded_ && !ran_shadow_replay
  │      │     ALLREDUCE → execute_shadow_allreduce()
  │      │     非 ALLREDUCE → fn() on orig comm
  │      │
  │      └── [Normal] else
  │            fn() with try-catch(NCCLFaultToleranceError)
  │            → 吞掉例外，記錄 end event，透明返回 work
  │
  ├─ 7. post(ncclStream, work)
  ├─ 8. ncclEndEvent_->record(ncclStream)
  ├─ 9. [Bug 10] work->ncclComm_ 指向實際使用的 comm
  └─ 10. workEnqueue(work) → Watchdog 監控
```

---

## 八、WorkNCCLFT 生命週期

```
initWork() → 建立 WorkNCCLFT，分配 ncclStartEvent_ / ncclEndEvent_
    │
    ├── 加入 workMetaList_（workEnqueue）
    │
    ├── [主執行緒] wait() → synchronize() → ncclEndEvent_->block(currentStream)
    │     stashed_for_allocator_safety_->unstash()
    │     若有 exception：handleException(CleanUpOnlyFT) [不 rethrow]
    │
    └── [Watchdog]
          ├── FT enabled + exception：清除 exception，記錄 ncclEndEvent_，erase work
          ├── FT disabled + exception：正常錯誤處理
          ├── early-abort（final_commit_op_ > 0）：強制記錄 ncclEndEvent_，setException
          └── 成功完成：
                retire FlightRecorder entry
                in_flight_shadow_bufs_[seq] → free_shadow_bufs_
                erase from workMetaList_
```

---

## 九、記憶體與串流管理

| 串流 | 用途 |
|------|------|
| `ncclStreams_[deviceKey]` | NCCL collective 執行串流 |
| `shadow_copy_stream_` | Shadow buffer D2H（GPU→CPU）及 H2D（CPU→GPU）複製串流 |
| current CUDA stream | 使用者計算串流（AllReduce sync mode 時也是 NCCL 串流） |

| 事件 | 用途 |
|------|------|
| `ncclStartEvent_` | 記錄 collective 開始（timing 啟用時） |
| `ncclEndEvent_` | 記錄 collective 結束；block() 用於 wait() 同步 |
| `shadow_copy_event_` | 記錄 D2H shadow copy 完成 |
| `shadow_restore_event_` | 記錄 H2D restore copy 完成，block NCCL stream |

---

## 十、環境變數與設定

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

## 十一、故障完整時序

```
t=0: NIC mlx5_0 硬體故障
t=1: NCCL progress thread → ncclRemoteError → nccl_ft_global_fault_callback(dev_idx=0)
t=2: trigger_fault_proposal() → local_hardware_fault_mask_ |= 1<<0
                                  pending_shadow_seq_ = shadow_seq_ (once)
t=3: Side-car 偵測到 fault_mask ≠ 0，寫 PROPOSE key
t=4: 其他 rank 偵測到有 PROPOSE → 寫自己的 PROPOSE (mask=0x0)
t=5: 所有 PROPOSE 就緒，rank 0 計算 agreed_ss = min(shadow_seqs)
t=6: rank 0 寫 COMMIT key
t=7: Side-car abort globalComm（喚醒主執行緒的 ncclAllReduce hang）
t=8: Side-car 設定 final_commit_op_ = cur_round + 1
t=9: Watchdog early-abort：偵測 final_commit_op_ > 0，force-clear in-flight work
t=10: 主執行緒 collective() barrier 感知 commit_signal
       current_op == agreed_ss + 1 → REBUILD path
       rebuild_shadow_ping_pong_topology() → proxy_global_comm_ 就緒
       in_flight_shadow_bufs_[agreed_ss] → H2D restore → shadow_restore_event_
t=11: execute_shadow_allreduce() replay → 訓練繼續 ✓
```
