# ProcessGroupNCCLFT — End-to-End Prototype Gap Analysis & Plan

**Version:** 2025-07 全面審查
**範圍:** 基於原始碼完整靜態分析，找出所有阻礙端到端測試的缺口並提供修復方案
**狀態:** Bug 1, 2, 4+5, 6, 9, 10 已全部修復並寫入 ProcessGroupNCCLFT.cpp / .hpp

---

## 一、系統現狀總結

目前 `ProcessGroupNCCLFT` 已完成以下核心骨架：

| 模組 | 狀態 | 所在位置 |
|---|---|---|
| Watchdog FT 錯誤清除路徑 | ✅ 已完成 (含 Bug A SIGSEGV 修復) | Line ~2477–2565 |
| `initLocalNvlinkComm()` (NVLink comm) | ✅ 已完成 | Line ~3945–4005 |
| `rebuild_shadow_ping_pong_topology()` | ✅ 已完成 | Line ~4022–4116 |
| `execute_shadow_allreduce()` (4-step relay) | ✅ 已完成 | Line ~4149–4341 |
| `trigger_fault_proposal()` | ✅ 已完成 | Line ~4343–4364 |
| `start_ft_negotiator_thread()` (2PC) | ✅ 已完成 | Line ~4367–4634 |
| Shadow Buffer checkpoint/restore | ✅ 已完成 | Line ~5634–5723 |
| `collective()` 2PC barrier + replay | ✅ 已完成 | Line ~4807–4924 |
| Multi-fault `faulty_local_devs_` set | ✅ 已完成 | HPP ~Line 1092 |
| TCPStore key cleanup | ✅ 已完成 | Line ~4835–4856 |

---

## 二、已確認的 Bug 與缺口（阻礙端到端測試）

以下依照**嚴重程度**排序，從必須修復（P0）到建議修復（P2）。

---

### Bug 1 (P0): Watchdog 在 FT 路徑中仍然先寫入 `COMM_ERROR`，再清除為 `SUCCESS`

**位置:** `ProcessGroupNCCLFT.cpp` Line 2424–2430（在 FT 路徑之前）

**問題描述:**  
Watchdog 迴圈的結構是：
```
if (work.exception()) {
    // Line 2424: 先無條件寫入 COMM_ERROR   <--- 問題在這裡
    pg_->error_ = ErrorType::COMM_ERROR;
}
// ... timeout check ...
if (work.exception()) {
    if (!pg_->ft_disabled_) {
        // Line 2487: 再清除回 SUCCESS
        pg_->error_ = ErrorType::SUCCESS;
```

在 FT 路徑清除之前，`error_` 已短暫被設為 `COMM_ERROR`。雖然之後會被清除，但如果主執行緒在這個窗口呼叫了任何會讀取 `error_` 的函數（例如 `checkError()`），就會觸發不必要的錯誤傳播。

**修復方案:**  
在第一個 `if (work.exception())` 區塊中，加入 FT 模式的判斷，**不要**在 FT 路徑中寫入 `COMM_ERROR`：

```cpp
if (work.exception()) {
    // [NCCL-FT] In FT mode, do NOT set COMM_ERROR here.
    // The FT branch below will clear it anyway, but the window
    // between set and clear can cause spurious checkError() failures.
    if (!pg_->ft_disabled_) {
        // FT mode: skip setting COMM_ERROR; let the FT branch handle it.
    } else {
        std::lock_guard<std::mutex> lock(pg_->errorMutex_);
        if (pg_->error_ == ErrorType::SUCCESS) {
            pg_->error_ = ErrorType::COMM_ERROR;
        }
    }
}
```

---

### Bug 2 (P0): `collective()` 主執行緒在 2PC barrier spin-wait 期間沒有保護 `errorMutex_`

**位置:** `ProcessGroupNCCLFT.cpp` Line 4809–4822

**問題描述:**  
主執行緒在 2PC barrier spin-wait 時：
```cpp
while (this->final_commit_op_.load(std::memory_order_relaxed) != current_op) {
    std::this_thread::yield();
}
```
此時 Watchdog 可能正在清除多個 poisoned work（多個 in-flight ops 在 dead comm 上全部失敗）。Watchdog 每次清除都會設/清 `error_`，主執行緒同時也在呼叫 `collective()` 的前置作業。雖然 `error_` 有 `errorMutex_` 保護，但 2PC spin-wait 沒有 `errorMutex_`，這本身沒問題。  

**真正的問題**是：如果主執行緒在 2PC spin-wait 之前（Line 4805）呼叫了 `pre(ncclStream, work)` ── `pre` 就是 `shadow_pre` lambda ── 但在 FT 模式下 `shadow_replay_pending_` 可能尚未被設置（Watchdog 還沒跑到），因此 `shadow_pre` 會做一次新的 checkpoint，覆蓋掉本應用來 replay 的 shadow buffer 內容。

**根本原因:** `pre(ncclStream, work)` 在 Line 4805 **先被呼叫**，2PC barrier 在 Line 4812 **後被檢查**。若故障恰好在 `pre` 到 2PC barrier 之間觸發，順序倒置。

**修復方案:**  
把 2PC barrier 的檢查（`boundary > 0 && current_op == boundary`）**移到 `pre()` 呼叫之前**，或者在 `pre()` 內部確保此時 `shadow_replay_pending_` 已被正確設置：

```cpp
// CORRECT order:
// 1. Check shadow restore event wait (already at Line 4782)
// 2. Check 2PC barrier FIRST (move from Line 4812 to before pre())
// 3. Call pre(ncclStream, work)   <-- checkpoint only after barrier clears
// 4. Execute fn() or shadow replay
```

實際修改：將整個 `if (!this->ft_disabled_ && C10_UNLIKELY(boundary > 0 && ...))` 區塊從 Line 4812 移到 Line 4805 `pre(ncclStream, work)` **之前**。

---

### Bug 3 (P0): `collective()` 降級路徑在 `ran_shadow_replay=true` 後，仍然呼叫了 `post(ncclStream, work)` 和 `work->ncclEndEvent_->record()`

**位置:** `ProcessGroupNCCLFT.cpp` Line 4967–4972

**問題描述:**  
`execute_shadow_allreduce` 直接呼叫底層 NCCL API，不會回傳 `ncclResult_t` 給 `collective()`。`post()` 和 `ncclEndEvent_->record()` 之後被無條件呼叫，這本身沒有問題。

但更重要的是：在 replay 路徑（`ran_shadow_replay=true`）中，`work->outputs_` 已在 Line 4753 設好，`work->future_` 在 Line 4978 完成。這條路徑沒有明顯問題。

**實際問題:** 經過重新審查，此項目升級為**注意事項**而非 Bug，標記為 P1，見下方。

---

### Bug 4 (P0): `target_op` 計算使用 `seqCollective_ + 10` 但沒有理由

**位置:** `ProcessGroupNCCLFT.cpp` Line 4387

```cpp
uint64_t target_op = this->seqCollective_ + 10;
```

**問題描述:**  
`seqCollective_` 在 Watchdog 執行緒中被讀取（無鎖），同時主執行緒也在修改它。加 10 是一個任意的緩衝，確保 "所有 in-flight ops 都能完成或被清除後，再設置 barrier"。

但問題在於：當 AllReduce 失敗時，**所有 in-flight ops 都會一起失敗**（因為 NCCL comm 已標記為 error）。Watchdog 會清除所有 poisoned works。主執行緒在這些 ops 完成之前已被阻塞（因為 `asyncOp=false` 的 `collective()` 回傳 nullptr，不會 `wait()`，但 DDP 也不會進行下一個 allreduce 直到前一個完成）。

實際上 `seqCollective_` 讀取時沒有 memory fence，且 side-car 執行緒和主執行緒的 `seqCollective_` 值可能不一致。

**更重要的問題:** 固定加 10 可能造成 `target_op` 太大，主執行緒在 barrier 等待時的 `current_op` 永遠不等於 `target_op`，導致**永久死結**。

**例如:** 故障時 `seqCollective_=37`，`target_op=47`。但只有 4 個 ops (34,35,36,37) 在 in-flight，主執行緒在清除後的下一個 op 是 38。`current_op == 47` 需要主執行緒又執行了 9 個 ops，但 2PC 在 op 47 之前不允許任何跨節點通訊，造成死結。

**修復方案:**  
`target_op` 應設為**當前正在執行的 op + 1**（即下一個要執行的 op），而不是 +10：
```cpp
// 故障時，主執行緒被 Watchdog 清除後，下一個 op 就是 barrier
// seqCollective_ 是已失敗的最後一個 op，下一個 collective 是 seqCollective_+1
uint64_t target_op = this->seqCollective_ + 1;
```

但由於 `seqCollective_` 在 side-car 執行緒讀取時無鎖，且實際值需要 memory ordering，更安全的做法是：
```cpp
// 從 Watchdog 清除的最後一個 seq 推算（最大 seq + 1）
// 因為 side-car 在觸發後立即讀取，此時 seqCollective_ 還沒被下一個 op bump
uint64_t target_op = this->seqCollective_.load(std::memory_order_acquire) + 1;
```
注意 `seqCollective_` 在 HPP 宣告為 `uint64_t`（非 atomic），需要確認它是否受到足夠的保護，或改成 `std::atomic<uint64_t>`。

---

### Bug 5 (P0): `seqCollective_` 不是 atomic，但 side-car 執行緒讀取它

**位置:** `ProcessGroupNCCLFT.hpp` (繼承自 ProcessGroupNCCL)，`ProcessGroupNCCLFT.cpp` Line 4387

**問題描述:**  
`seqCollective_` 是 `uint64_t`（非 atomic），主執行緒寫入，side-car 執行緒在 `trigger_fault_proposal()` 和 `start_ft_negotiator_thread()` 中讀取。這是一個 **data race**，屬於 undefined behavior。

**修復方案:**  
在讀取 `seqCollective_` 前加 `std::memory_order_acquire`，或在 side-car 執行緒中改用 `pgStatus_->lastEnqueuedSeq`（已是 atomic-safe 的）來推算 target_op：
```cpp
// 使用 lastEnqueuedSeq 替代 seqCollective_
uint64_t current_seq = static_cast<uint64_t>(
    pg_->pgStatus_->lastEnqueuedSeq.load(std::memory_order_acquire));
uint64_t target_op = current_seq + 1;
```

---

### Bug 6 (P1): `allreduce()` 中的 `intraNodeComm_` 路徑繞過了 Shadow Buffer 和降級路徑

**位置:** `ProcessGroupNCCLFT.cpp` Line 5741–5752

**問題描述:**  
```cpp
if (opts.reduceOp == ReduceOp::SUM) {
    if (intraNodeComm_ != nullptr) {
        auto algo = intraNodeComm_->selectAllReduceAlgo(tensor);
        if (algo != intra_node_comm::AllReduceAlgo::NONE) {
            intraNodeComm_->allReduce(tensor, algo);
            return c10::make_intrusive<IntraNodeCommWork>();  // <-- 直接 return！
        }
    }
}
```
這條路徑完全繞過了 `allreduce_impl()`，因此：
- 沒有 `current_shadow_reduce_op_` 設定
- 沒有 Shadow Buffer checkpoint
- 降級後不會走 Shadow Ping-Pong relay

如果 `intraNodeComm_` 被初始化（它需要特定的環境變數），在故障後此路徑仍然走 intraNode comm 而非降級路徑，導致**靜默數學錯誤或崩潰**。

**修復方案:**  
在 FT 模式下，如果 `is_degraded_` 為 true，**跳過** intraNodeComm 路徑：
```cpp
if (opts.reduceOp == ReduceOp::SUM && !is_degraded_) {  // <-- 加上 !is_degraded_
    if (intraNodeComm_ != nullptr) {
        ...
    }
}
```

---

### Bug 7 (P1): `execute_shadow_allreduce` 中 PROXY 步驟 2 的 ATen 操作（`output.copy_` / `output.add_`）可能在 ncclRecv 完成前執行

**位置:** `ProcessGroupNCCLFT.cpp` Line 4282–4297

**問題描述:**  
```cpp
C10D_NCCL_FT_CHECK(ncclGroupEnd(), std::nullopt);  // ncclRecv enqueued

{
    at::cuda::CUDAStreamGuard guard(stream);
    output.copy_(input);     // ATen kernel enqueued
    for (const auto& wb : ward_bufs) {
        output.add_(wb);     // reads ward_bufs data
    }
}
```

NCCL 的 `ncclGroupEnd()` 只保證 NCCL kernel 被**提交**到 CUDA stream，但不保證執行完成。在同一個 `stream` 上，CUDA kernel 是串行的，所以 `output.copy_` 會在 `ncclRecv` 之後執行。這**理論上是正確的**，只要 `CUDAStreamGuard` 正確地設定了 stream。

但 Bug 5 診斷 LOG 顯示：如果 `CUDAStreamGuard` 沒有正確切換（`active.stream() != stream.stream()`），ATen ops 就會落在錯誤的 stream 上，造成 race。

**目前狀態:** 已有 Bug5 診斷 LOG，但沒有自動修復。如果 WARNING 觸發，需要手動介入。

**建議修復:** 改用 `at::cuda::setCurrentCUDAStream(stream)` + 手動恢復，或確認 `CUDAStreamGuard` 在 NCCL 進度執行緒上下文中是否正確運作。

---

### Bug 8 (P1): `rebuild_shadow_ping_pong_topology()` 從**已壞掉的** global comm split

**位置:** `ProcessGroupNCCLFT.cpp` Line 4067–4097

**問題描述:**  
```cpp
auto it = devNCCLCommMap_.find(key);
...
globalComm = it->second;  // 這是已失敗的原始 global comm
...
proxy_global_comm_ = NCCLFTComm::split(globalComm.get(), color, rank_, config);
```

從一個已經報告 `ncclRemoteError` 的 comm 呼叫 `ncclCommSplit`，行為是**未定義的**。NCCL 可能：
- 直接回傳 error → `NCCLFTComm::split` 回傳 nullptr → `TORCH_CHECK` 失敗（對 healthy rank）
- 或者 NCCL 在 split 之前先完成 comm abort，再從 abort 狀態 split（也是未定義）

**正確做法（根據先前分析）:**  
先呼叫 `ncclCommBanNic(dev_idx)`（在 custom NCCL 中），然後 `ncclCommSplit` 會自動排除被 ban 的 NIC，並基於**仍然健康的**部分建立新 comm。這依賴於 custom NCCL 的 `ncclCommBanNic` 實作正確地更新 comm 的 NIC 可見性，而不是讓整個 comm 進入 error 狀態。

**需要確認的問題:**  
在你的 custom NCCL 中，`ncclCommBanNic` 呼叫後，原始 comm 是否還能被用來呼叫 `ncclCommSplit`？還是必須使用一個從來沒壞過的 comm（例如 `local_nvlink_comm_`，它永遠不走有問題的 NIC）？

**建議的防禦性修復:**  
在 `rebuild_shadow_ping_pong_topology()` 中，加入 comm 健康檢查：
```cpp
// Check if global comm is still usable before split
if (globalComm->isAborted()) {
    LOG(ERROR) << "[NCCL-FT] Global comm is aborted; cannot split from it. "
               << "Rebuild failed.";
    proxy_comm_ready_.store(false);
    return;
}
```

---

### Bug 9 (P1): `pending_shadow_seq_` 初始化為 0，但 0 也是合法的 shadow_seq

**位置:** `ProcessGroupNCCLFT.hpp` Line 1172

**問題描述:**  
```cpp
std::atomic<uint64_t> pending_shadow_seq_{0};
```

在 `start_ft_negotiator_thread()` 中：
```cpp
uint64_t my_ss = this->pending_shadow_seq_.load(std::memory_order_acquire);
if (my_ss == 0) {
    my_ss = proposal_shadow_seq;  // 使用 proposer 的值作為 fallback
}
```

如果訓練在第 1 個 op 就失敗（`shadow_seq_=0` 是合法的，`seqCollective_` 從 0 開始），`pending_shadow_seq_=0` 會被誤判為 "尚未設定"，導致非故障節點的 ranks 使用 proposer 的 shadow_seq 而非自己的 0。

**修復方案:**  
使用 sentinel 值，例如 `UINT64_MAX` 表示 "未設定"：
```cpp
std::atomic<uint64_t> pending_shadow_seq_{UINT64_MAX};
// 在 trigger_fault_proposal():
this->pending_shadow_seq_.store(this->shadow_seq_, std::memory_order_release);
// 在 negotiator thread:
uint64_t my_ss = this->pending_shadow_seq_.load(std::memory_order_acquire);
if (my_ss == UINT64_MAX) {
    my_ss = proposal_shadow_seq;
}
// Reset after use:
this->pending_shadow_seq_.store(UINT64_MAX, std::memory_order_release);
```

---

### Bug 10 (P1): `collective()` 降級路徑在 ALLREDUCE 走 execute_shadow_allreduce 後，`fn()` 被跳過，但 WorkNCCLFT 仍然被 enqueue 到 Watchdog

**位置:** `ProcessGroupNCCLFT.cpp` Line 4900–5016

**問題描述:**  
在 shadow replay 和降級路徑下，`execute_shadow_allreduce()` 直接呼叫 `ncclSend/ncclRecv/ncclAllReduce`，這些是立即提交到 stream 的 NCCL 操作。之後仍然呼叫了：
```cpp
work->ncclEndEvent_->record(ncclStream);  // Line 4971
workEnqueue(work);                         // Line 5013
```

`ncclEndEvent_` 被 record 在 `ncclStream` 上，這個 stream 上有 `execute_shadow_allreduce` 的 kernel。Watchdog 用 `ncclEndEvent_` 來偵測 work 是否完成，這是正確的。

**但問題在於:** `work->ncclComm_` 在 Line 4973 被設為**已壞的** `ncclComm`（原始 global comm），而不是 `proxy_global_comm_`。Watchdog 呼叫 `work.checkAndSetException()` 時，它查詢的是 `ncclComm->getNcclCommFailureReason()`，即**已壞的** comm，這會永遠回傳 error，讓 Watchdog 再次誤判這個 work 為失敗。

**修復方案:**  
在 shadow replay 和降級路徑下，設定 `work->ncclComm_` 為 `proxy_global_comm_` （對 healthy/proxy ranks）或 `local_nvlink_comm_` （對 faulty ranks）：
```cpp
if (ran_shadow_replay || (is_degraded_ && opType == OpType::ALLREDUCE)) {
    int local_rank = rank_ % localDeviceCount_;
    bool is_faulty = (faulty_local_devs_.count(local_rank) > 0);
    if (is_faulty) {
        work->ncclComm_ = local_nvlink_comm_;
    } else if (proxy_global_comm_ != nullptr) {
        work->ncclComm_ = proxy_global_comm_;
    }
} else {
    work->ncclComm_ = ncclComm;  // 原始路徑
}
```

---

### Bug 11 (P2): 連續第二次故障時，`last_handled_event` 去重邏輯可能導致新故障被忽略

**位置:** `ProcessGroupNCCLFT.cpp` Line 4431–4432

**問題描述:**  
```cpp
if (event_str.rfind("PROPOSE:", 0) == 0 &&
    event_str != last_handled_event) {
```

`last_handled_event` 儲存的是上一次已處理的 proposal 字串（含 target_op, node, dev, shadow_seq）。第二次故障時如果恰好有相同的參數組合（例如相同 dev 再次失敗），`event_str == last_handled_event`，新故障被忽略。

**修復方案:**  
改用 `target_op` 作為去重 key，而非整個 proposal 字串：
```cpp
uint64_t last_handled_target_op = 0;
// ...
if (target_op != last_handled_target_op) {
    last_handled_target_op = target_op;
    // ... process proposal
}
```

---

### Bug 12 (P2): `do_not_cross_op_` 和 `final_commit_op_` 重置順序可能導致第二次故障的 side-car 立即讀到 stale NCCL_FT_EVENT

**位置:** `ProcessGroupNCCLFT.cpp` Line 4859–4861

```cpp
this->do_not_cross_op_.store(0, std::memory_order_release);
this->final_commit_op_.store(0, std::memory_order_release);
```

TCPStore cleanup 在 Line 4835–4856 中刪除了 `NCCL_FT_EVENT`，但 side-car 執行緒在 Line 4427 `check({"NCCL_FT_EVENT"})` 可能在 delete 和下一個 PROPOSE 之間的時窗讀到一個剛被寫入的新 PROPOSE（因為 TCPStore `set` 是冪等的，`deleteKey` 後立刻 `set` 相當於覆寫）。  

這實際上是正確的（新 PROPOSE 應該被處理），但 `last_handled_event` 機制需要確保不重複處理同一個 proposal。目前邏輯看起來是正確的，標記為注意事項。

---

## 三、NCCL Custom Source 端的必要修改

（提醒事項，不需要在 PyTorch 端修改）

| 項目 | 說明 |
|---|---|
| **Bug A (P0)**: FT init hook 在 topology 建構期間自動 ban NIC | 必須移除或加 guard，只允許透過 `ncclCommBanNic()` 顯式呼叫 |
| **Bug B (P0)**: Ban function 每個 rank 觸發兩次 | 加 `bool ft_init_done` guard |
| **確認事項**: `ncclCommBanNic` 後，原始 comm 是否可用於 `ncclCommSplit` | 如不可用，需改從 `local_nvlink_comm_` split |

---

## 四、端到端測試前的最小修復清單（Must-Fix Before Testing）

按優先順序：

### Step 1: 修復 `target_op` 計算（Bug 4 + Bug 5）
```cpp
// ProcessGroupNCCLFT.cpp Line 4387
// BEFORE:
uint64_t target_op = this->seqCollective_ + 10;
// AFTER:
uint64_t current_seq = static_cast<uint64_t>(
    pg_->pgStatus_->lastEnqueuedSeq.load(std::memory_order_acquire));
uint64_t target_op = current_seq + 1;
```

### Step 2: 修復 Watchdog COMM_ERROR 提前設定（Bug 1）
在 `if (work.exception())` 第一個區塊中（Line 2424），FT 模式下跳過 `COMM_ERROR` 設定。

### Step 3: 移動 2PC barrier 到 `pre()` 之前（Bug 2）
將 Line 4812 的 2PC barrier 判斷移到 Line 4805 `pre(ncclStream, work)` 之前。

### Step 4: 修復 `work->ncclComm_` 降級路徑指向壞掉的 comm（Bug 10）
在 replay/降級路徑下，`work->ncclComm_` 改指向 `proxy_global_comm_` 或 `local_nvlink_comm_`。

### Step 5: 修復 `intraNodeComm_` 繞過降級路徑（Bug 6）
在 `allreduce()` 中加入 `!is_degraded_` 判斷。

### Step 6: 修復 `pending_shadow_seq_` 0 sentinel（Bug 9）
改用 `UINT64_MAX` 作為 "未設定" 的 sentinel 值。

---

## 五、實作計劃（有序執行）

```
[Step 1] Bug 4+5: 修復 target_op 計算 → 讓 2PC barrier 在正確的 op 觸發          ✅ 已完成
[Step 2] Bug 1:   修復 Watchdog COMM_ERROR 提前設定 → 避免假性錯誤傳播            ✅ 已完成
[Step 3] Bug 2:   移動 2PC barrier 到 pre() 之前 → 正確的 shadow checkpoint 時序   ✅ 已完成
[Step 4] Bug 10:  修復 work->ncclComm_ 在降級路徑 → 避免 Watchdog 誤判 replay work ✅ 已完成
[Step 5] Bug 6:   intraNodeComm_ bypass 修復 → 確保降級後 allreduce 走正確路徑    ✅ 已完成
[Step 6] Bug 9:   pending_shadow_seq_ sentinel 修復 → 首個 op 故障時正確運作        ✅ 已完成
[Step 7] NCCL端: 修復 FT init hook 自動 ban NIC（需使用者在 custom NCCL 修改）     ⏳ 待處理
[Step 8] 驗證:   2-node × 8-GPU 端到端測試，注入單 NIC 故障，觀察訓練繼續執行     ⏳ 待處理
```

---

## 六、端到端測試驗證清單

測試完成後，以下 LOG 應出現且順序正確：

```
[初始化期]
[NCCL-FT] Set NCCL_IB_HCA=mlx5_<local_rank>              # 每個 rank
[NCCL-FT] First collective detected; registering fault callback...
[NCCL-FT] Fault callback registered on comm ...
[NCCL-FT] local_nvlink_comm_ ready: ...

[故障觸發期]
[NCCL-FT-TRACE] !!! NCCL 底層成功觸發 Callback !!! 故障網卡: <dev>
[NCCL-FT] 瞬間攔截本地網卡故障，標記 dev_idx: <dev>
[NCCL-FT] Watchdog: clearing poisoned work for device <dev>
[NCCL-FT] Shadow restore enqueued for seq=<N>

[2PC 協商期]
[NCCL-FT] Side-car relayed proposal to TCPStore: PROPOSE:...
[NCCL-FT] Side-car intercepted proposal; fence OP=<M>
[NCCL-FT] All ranks ACKed; COMMIT written for OP <M>
[NCCL-FT] final_commit_op_ set to <M>

[降級重建期]
[NCCL-FT] 2PC committed for OP=<M>; rebuilding proxy topology
[NCCL-FT] proxy_global_comm_ ready: ...
[NCCL-FT] TCPStore: deleted per-rank keys for OP=<M>

[Shadow Ping-Pong 執行期]
[NCCL-FT] Sub-Task 5 REPLAY after rollback for seq=<N>
[NCCL-FT] execute_shadow_allreduce: role=FAULTY/PROXY/HEALTHY ...
[NCCL-FT][FAULTY] Steps 1+4 enqueued via proxy=<proxy>
[NCCL-FT][PROXY] All steps enqueued for wards=[<wards>]
[NCCL-FT][HEALTHY] ncclAllReduce enqueued on proxy_global_comm_.

[穩定降級期（後續 ops）]
[NCCL-FT] Degraded mode active, opType=ALLREDUCE, rank=<R>
[NCCL-FT] execute_shadow_allreduce: role=... seq=<N+1>
```

---

## 七、已完成項目（不需再處理）

- ✅ `initLocalNvlinkComm()` — ncclCommSplit 方式，已驗證邏輯正確
- ✅ `rebuild_shadow_ping_pong_topology()` — symmetric degradation，multi-fault 集合
- ✅ `execute_shadow_allreduce()` — 4-step relay，AVG 修正，multi-ward 支援
- ✅ `trigger_fault_proposal()` — atomic CAS，pending_shadow_seq snapshot
- ✅ 2PC 完整協議 — PROPOSE/ACK/COMMIT，shadow_seq 共識（min 策略）
- ✅ Shadow Buffer — checkpoint/restore，grow-on-demand，replay_pending guard
- ✅ SIGSEGV Bug A 修復 — nullptr guard on `work.outputs_`
- ✅ seqCollective_ double-count 修復 — replay 前 decrement
- ✅ TCPStore key cleanup — 每輪故障後清理
- ✅ Multi-fault `faulty_local_devs_` unordered_set

---

## 八、已知不在 Prototype 範圍的項目

| 項目 | 原因 |
|---|---|
| AllGather 降級路徑 | DDP only prototype；FSDP/ZeRO 為 P0 後續工作 |
| ReduceScatter 降級路徑 | 同上 |
| Broadcast 降級路徑 | 使用原始 comm 的 warning 已實作 |
| Error type 分類 | 目前所有 NCCL error 視為 NIC 故障；精細分類為後續工作 |
| Proxy 負載均衡 | 固定 `(failed+1) % N`；動態選擇為後續工作 |
| Tensor shard fan-out | 單一代理，非多路廣播；效能優化為後續工作 |
