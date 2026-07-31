# ProcessGroupNCCLFT — End-to-End Prototype Gap Analysis & Plan

---

## 一、系統現狀總結

ProcessGroupNCCLFT 是基於 ProcessGroupNCCL 複製並擴充的自訂 backend（已在 c10d 中以 `nccl_ft` 名稱註冊）。

目前架構：三執行緒模型
- **主執行緒**：執行 collective（AllReduce 等），遇到故障後在 2PC barrier 停下等待
- **Watchdog**（`pt_nccl_watchdg`）：偵測失敗的 work、清除 poison、恢復 shadow buffer、設置 replay 旗標
- **Side-car negotiator**（`pt_nccl_ft_side`）：透過 TCPStore 2PC 協商故障邊界，建立所有 rank 的共識

硬體假設：2 台 server × 8 GPU，每張 GPU 對應一張 NIC（NUMA-local），同 server 內所有 GPU 以 NVLink 互連。

---

## 二、已確認完成的三大核心拼圖

> 使用者原計劃中提到的三個「尚未完成」項目，**在程式碼中均已實作完畢**。

### 拼圖 1：initLocalNvlinkComm() — ✅ 已完成
- 位置：`ProcessGroupNCCLFT.cpp` lines 3966–4027
- 使用 `ncclCommSplit(color=node_id)` 從 global comm 切割，每個 node 獨立取得純 NVLink sub-communicator
- 在第一次 collective 時 lazy init（因為 ncclCommSplit 需要 global comm 已存在）
- 已處理 `NCCL_HAS_COMM_SPLIT` 缺失的 warning 路徑

### 拼圖 2：rebuild_shadow_ping_pong_topology() — ✅ 已完成
- 位置：`ProcessGroupNCCLFT.cpp` lines 4043–4143
- 對稱降級：`faulty_local_devs_` 中的本地 rank 在所有 node 上全部排除
- 健康 rank 傳入 `color=1`，故障 rank 傳入 `NCCL_SPLIT_NOCOLOR`
- 正確計算 `proxy_comm_rank_` / `proxy_comm_size_`
- 使用 `ncclCommSplit` 而非 `ncclCommInitRank`，避免重新掃描 NIC（在 NUMA-local NIC 環境下 `ncclCommInitRank` 會 crash）

### 拼圖 3：AllReduce 攔截與 Shadow Ping-Pong 執行 — ✅ 已完成
- `execute_shadow_allreduce()` 位置：`ProcessGroupNCCLFT.cpp` lines 4175–4362
- 四步驟 relay（FAULTY → PROXY → cross-node AllReduce → PROXY → FAULTY）
- `collective()` 中的攔截邏輯：
  - Replay path（lines 5030–5052）：`shadow_replay_pending_` 旗標觸發
  - 常態降級 path（lines 5054–5080）：`is_degraded_` 旗標觸發
- AVG 修正：`proxy_global_comm_` size 不同，改用 ncclSum + 手動除以原始 world size
- Multi-ward 支援：一個 proxy 可以同時替多個故障 rank 中繼

---

## 三、目前阻塞端對端測試的根本問題：2PC Barrier 設計缺陷

> 以下是透過 `test.log` 確認的真正阻塞點。三大拼圖已實作，但訓練仍會在此掛住。

### Bug B（P0）：`current_op == boundary` 精確匹配永遠不觸發

**症狀（test.log 確認）：**
- 全部 16 個 rank 的 side-car 都正確攔截 proposal，fence OP=48
- `do_not_cross_op_` 正確設為 48
- 但 log 中完全沒有「抵達警戒線」訊息
- 訓練掛住，無 COMMIT，無拓撲重建

**根本原因：**

DDP 使用 `asyncOp=true`（bucket overlap），在 op 43 的 NCCL 網路錯誤返回之前，
op 44、45、46、47 已經被 enqueue 到 NCCL stream 上。Watchdog 清除這些失敗的 work 時，
`seqCollective_` 已經是 47 或 48。

Side-car 讀取 `pgStatus_->lastEnqueuedSeq`（例如 47）然後設 `target_op = 48`。
但 `seqCollective_` 在 collective() 的開頭就已經 bump，當 op 49 進來時
`current_op = 49`，不等於 48。**精確匹配永遠錯過，barrier 永遠不觸發。**

### Bug C（P0）：多個提案者同時覆寫 NCCL_FT_EVENT

**症狀（test.log 確認）：**
- rank0 在 T+0ms 寫入 PROPOSE，rank1 在 T+46ms 覆寫同一個 key，rank0 在 T+1255ms 再次覆寫
- 每次覆寫都重置 `target_op`，2PC coordinator（rank0）的 ACK 輪詢變成追打不同目標的無限迴圈

**根本原因：**

每個 rank 的 side-car 都各自讀到 `local_hardware_fault_mask_ != 0` 然後獨立寫 `NCCL_FT_EVENT`。
寫入是覆寫（`set`），不是 CAS。多個 rank 寫入不同 `target_op` 的 PROPOSE 後，
coordinator 在 ACK polling 時目標一直在變。

---

## 四、修復計劃：Shadow-Seq 滾動回滾機制

> 這是取代原本 `target_op` 精確匹配的核心重設計。目標：讓 barrier 從「等到某個特定 op」
> 改為「任何 rank 感知到 commit 就立即回滾並重播」。

### Step A（P0）：修復 Bug C — 只讓一個 rank 寫 PROPOSE

**策略：** 只有本地 `local_hardware_fault_mask_` 不為 0 的 rank 才寫 `NCCL_FT_EVENT`。
其他 rank 的 side-car 只讀，不寫（但仍然 ACK）。

**實作細節：**

`start_ft_negotiator_thread()` 中 Task 1（Relay Write）邏輯：
```
// 只有「本地發生故障」的 rank 才寫 PROPOSE（目前行為 — 正確）。
// 不需要額外保護：exchange(0) 確保每個 fault_mask 只被一個迴圈迭代消耗。
// 若兩個 rank 同時 fault，各自寫入，後者覆寫前者，這是已知問題（Bug C）。
```

**修復方案：** 使用 TCPStore 的 `compareAndSet` 語意（若 key 不存在才寫）：
- 改為先嘗試 `check({"NCCL_FT_EVENT"})`，若 key 不存在才寫（搶佔式）
- 或使用 TCPStore 的 add/counter key 作為 mutex（若 counter 為 0 才可寫）

**最簡實作（Prototype）：** 讓每個 rank 寫自己的 per-rank key：
```
"NCCL_FT_PROPOSE_<rank>" = "PROPOSE:<target_op>:<node_id>:0x<mask>:<shadow_seq>"
```
rank 0 的 coordinator 輪詢所有 `NCCL_FT_PROPOSE_*` key，取得所有 `target_op` 的最小值（最保守的邊界）以及所有 `shadow_seq` 的最小值，然後寫入 COMMIT。

---

### Step B（P0）：修復 Bug B — 用 `>=` 替換 `==`，配合 `committed_shadow_seq_` 主動排水

**核心洞察：** 不應期待 `seqCollective_` 正好等於 `target_op`。
故障發生後，DDP 可能繼續 enqueue 更多 op（async bucket overlap）。
這些多餘的 op 需要被**排水並丟棄**，然後從 `agreed_shadow_seq` 重播。

**新 barrier 邏輯（替換 lines 4900-4954）：**

```cpp
// 新 barrier 條件：只要 final_commit_op_ 被設定（任何值 > 0），
// 且主執行緒尚未完成 rollback（用新的 rollback_done_ bool 防止重入），
// 就進入排水與重播流程。
uint64_t commit_op = this->final_commit_op_.load(std::memory_order_acquire);
if (!this->ft_disabled_ && !this->rollback_done_ && C10_UNLIKELY(commit_op > 0)) {
    LOG(INFO) << logPrefix()
              << "[NCCL-FT] 偵測到 COMMIT (commit_op=" << commit_op
              << " current_op=" << current_op << ")，開始排水與回滾";

    // 1. Drain: 跳過所有 seqCollective_ > committed_shadow_seq_ 的 op。
    //    這些 op 是故障後 DDP 繼續 enqueue 的多餘 op，需要被靜默丟棄。
    //    具體做法：在此處直接 return（不執行 fn/shadow_allreduce），
    //    並將 seqCollective_-- 以撤銷本次 bump（讓下一輪 op 從正確 seq 開始）。
    uint64_t agreed_ss = this->committed_shadow_seq_.load(std::memory_order_acquire);
    if (current_op > agreed_ss + 1) {
        // 這個 op 是多餘的（故障後 async 多 enqueue 的），靜默丟棄。
        if (!coalescing_state_) seqCollective_--;
        // 回傳一個已完成的空 work，讓 DDP 的 wait() 不掛住。
        // [見 Step D 的 NullWork 設計]
        return make_null_work(device, rank_, opType, inputs, outputs);
    }

    // 2. 等待拓撲重建完成（negotiator 在 final_commit_op_ 設定前已呼叫 rebuild）。
    rebuild_shadow_ping_pong_topology();
    this->is_degraded_ = true;
    this->rollback_done_ = true;

    // ... TCPStore cleanup ...
    // ... reset do_not_cross_op_, final_commit_op_ ...
}
```

---

### Step C（P0）：seqCollective_ 重置對齊

**問題：** 排水後 `seqCollective_` 的值可能與 `agreed_shadow_seq` 不一致。
Shadow replay 時 `seqCollective_--` 的目的是讓 replay 用同一個 seq，但若中間排水了多個 op，
`seqCollective_` 已經超前很多，單次 decrement 不夠。

**修復：**
```cpp
// 在排水完成後，將 seqCollective_ 強制設回 agreed_ss，讓 replay 從正確的 seq 開始。
// 所有 rank 都執行相同的 2PC commit，所以 agreed_ss 在所有 rank 上是一致的。
seqCollective_ = agreed_ss;
```

---

### Step D（P1）：NullWork — 讓 DDP 的 wait() 不掛住

**問題：** DDP 呼叫 `allreduce_bucket(bucket)` 後會呼叫 `work->wait()`。
若主執行緒在排水時直接 return 一個正常的 work（但沒有實際執行任何 kernel），
`wait()` 可能卡在等待 endEvent。

**修復：** 返回一個 `ncclEndEvent_` 已 record 的空 work，使 `wait()` 立即完成。

```cpp
// 在 Step B 的排水路徑中：
work->ncclEndEvent_->record(ncclStream);
work->setCompleted();
return work;
```

---

### Step E（P1）：rollback_done_ 重置與多輪故障

**問題：** `rollback_done_` 防止重入，但下一次故障需要它被重置。

**修復：** 在每輪故障的 TCPStore cleanup 後，重置 `rollback_done_ = false`，
並且在 negotiator 設定下一輪 `final_commit_op_` 之前不重置（用 `final_commit_op_ = 0` 作為哨兵）。

---

## 五、實作清單（有序執行）

```
[已完成]
[✅] initLocalNvlinkComm()                        — lines 3966-4027
[✅] rebuild_shadow_ping_pong_topology()           — lines 4043-4143
[✅] execute_shadow_allreduce() 四步驟 relay        — lines 4175-4362
[✅] AllReduce 攔截（degraded / replay 路徑）       — collective() lines 5030-5080
[✅] Bug 12 multi-NIC bitmask (fetch_or)           — trigger_fault_proposal + side-car
[✅] Shadow Buffer 改為 pinned CPU 記憶體            — ensure_shadow_buffer()
[✅] Shadow copy stream 分離（D2H 零延遲）          — shadow_copy_stream_
[✅] Watchdog SIGSEGV Bug A 修復                   — nullptr guard on work.outputs_
[✅] seqCollective_ double-count 修復（replay decrement）
[✅] TCPStore key cleanup
[✅] 多 ward faulty_local_devs_ unordered_set

[已完成]
[✅] Step A: 每個 rank 寫自己的 NCCL_FT_PROPOSE_<rank>_<round>；非故障 rank 寫 0x0 佔位；rank 0 coordinator 聚合所有 key 後計算 min shadow_seq 並寫 COMMIT
[✅] Step B: barrier 改為 commit 感知 (final_commit_op_ > 0)；current_op > agreed_ss+1 → DRAIN；current_op == agreed_ss+1 → REBUILD
[✅] Step C: REBUILD path 後立即 seqCollective_ = agreed_ss，讓 replay 的 seqCollective_-- 正確落地
[✅] Step D: DRAIN path 在 return 前 record ncclEndEvent_ + 設定 future_（DDP wait() 立即返回）
[✅] Step E: rollback_done_ bool 新增至 HPP；replay 完成後 reset；no-replay 安全路徑 reset

[後續（Prototype 後）]
[ ] Step 10: Bug 13 Per-Server NIC Pool 非對稱降級
[ ] AllGather/ReduceScatter 降級路徑（FSDP/ZeRO）
```

---

## 六、各 Step 詳細實作

### Step A 詳細：Per-Rank PROPOSE Key

**HPP 修改：** 無（沿用現有變數）

**CPP 修改 — start_ft_negotiator_thread() Task 1：**

```cpp
// 舊：統一寫入 "NCCL_FT_EVENT"（多個 rank 互相覆蓋）
// 新：寫入自己的 per-rank key
if (fault_mask != 0) {
    std::string propose_key = "NCCL_FT_PROPOSE_" + std::to_string(this->rank_);
    // ... 組裝 proposal 字串（格式不變）...
    this->globalStore_->set(propose_key, vec);
    LOG(INFO) << ... "[NCCL-FT] Side-car wrote per-rank propose key: " << propose_key;
}
```

**CPP 修改 — start_ft_negotiator_thread() Task 2（原先讀 NCCL_FT_EVENT）：**

rank 0 coordinator 在全部 per-rank propose key 都出現後，再統一計算 min_target_op 和 min_shadow_seq，然後走後續 2PC 流程（COMMIT key 機制不變）。

非 rank 0 的 rank：每 50ms 輪詢 `NCCL_FT_COMMIT_<target_op>` key 是否出現，確認後執行原有的 ACK + SHADOW_SEQ write 流程，並等待最終 COMMIT。

**注意：** 這個設計讓 coordinator 在收到所有節點的 PROPOSE 後才計算 target_op，
而非每個 rank 分別猜測自己的 target_op，更加健壯。

---

### Step B 詳細：Commit 感知 + 排水替換精確匹配

**HPP 修改：**
```cpp
bool rollback_done_{false};  // 新增：防止同一輪故障重複 rollback
```

**CPP 修改 — collective() 中的 barrier 區塊（lines 4900-4954）：**

```cpp
uint64_t commit_op = this->final_commit_op_.load(std::memory_order_acquire);
if (!this->ft_disabled_ && !this->rollback_done_ && C10_UNLIKELY(commit_op > 0)) {
    uint64_t agreed_ss = this->committed_shadow_seq_.load(std::memory_order_acquire);
    uint64_t current_op = this->seqCollective_;  // 已被 bump

    LOG(INFO) << logPrefix()
              << "[NCCL-FT] Commit 感知: commit_op=" << commit_op
              << " agreed_ss=" << agreed_ss
              << " current_op=" << current_op;

    if (current_op > agreed_ss + 1) {
        // 這是故障後多 enqueue 的多餘 op，靜默丟棄。
        if (!coalescing_state_) seqCollective_--;
        LOG(INFO) << logPrefix()
                  << "[NCCL-FT] 排水多餘 op " << current_op
                  << " (agreed_ss=" << agreed_ss << ")，返回空 work";
        // 記錄 endEvent 讓 work->wait() 立即完成
        work->ncclEndEvent_->record(ncclStream);
        return work;
    }

    // current_op == agreed_ss + 1：這是 replay op，執行拓撲重建。
    LOG(INFO) << logPrefix()
              << "[NCCL-FT] 抵達重播點 OP=" << current_op
              << "，執行拓撲重建";
    rebuild_shadow_ping_pong_topology();
    this->is_degraded_ = true;
    this->rollback_done_ = true;
    // seqCollective_ 強制對齊（Step C）
    seqCollective_ = agreed_ss;

    // TCPStore cleanup（現有邏輯，不變）...
    // reset do_not_cross_op_ / final_commit_op_（現有邏輯，不變）...
}
```

---

### Step C 詳細：seqCollective_ 重置

在 Step B 的 rebuild 完成後立即執行：
```cpp
// After rebuild, seqCollective_ must equal agreed_ss so that the replay
// call (which will bump it again to agreed_ss+1) ends up at the right seq.
seqCollective_ = agreed_ss;
```

然後在 replay path（lines 5037-5038）的 `seqCollective_--` 仍然保留：它把 `agreed_ss+1` 調回 `agreed_ss`，讓 replay 使用 seq=agreed_ss。

---

### Step D 詳細：NullWork（排水路徑）

```cpp
// 排水路徑：record endEvent，然後立即返回 work。
// ncclEndEvent 已 record → WorkNCCLFT::wait() 不會 hang。
// work->outputs_ 已設定（lines ~4852）→ DDP 的 result() 可安全呼叫。
work->ncclEndEvent_->record(ncclStream);
return work;
```

不需要新的 NullWork class；利用現有的 work 物件，僅提前 record endEvent。

---

## 七、NCCL Custom Source 端的必要修改（使用者在 custom NCCL fork 中完成）

| 修改點 | 原因 | 狀態 |
|---|---|---|
| `ncclIbResiliencyHandleDeviceFailure`: `ncclSystemError` → `ncclRemoteError` | FT callback 路徑需要 ncclRemoteError | ✅ 已完成 |
| `ncclIbResiliencyProbeHandleCompletionEvent`: `ncclSuccess` → `ncclRemoteError` | 同上 | ✅ 已完成 |
| `topo.cc ncclTopoPopulateNics`: 移除 ban check block | 避免 NCCL 自行過濾已 ban 的 NIC | ✅ 已完成 |
| `init.cc`: 新增 `ncclCommBanNicReset()` API | 支援從 PyTorch 側重置 ban 狀態 | ✅ 已完成 |
| `init.cc nccl_ft_trigger_fault`: 呼叫 `trigger_fault_proposal` callback | NCCL 偵測到 IB 故障時通知 PyTorch | ✅ 已完成 |

---

## 八、端到端測試驗證清單

Step A–E 完成後，以下 LOG 應出現且順序正確：

```
[初始化期]
[NCCL-FT] Set NCCL_IB_HCA=mlx5_<local_rank>
[NCCL-FT] First collective detected; registering fault callback...
[NCCL-FT] Fault callback registered on comm ...
[NCCL-FT] local_nvlink_comm_ ready: ...

[故障觸發期]
[NCCL-FT-CALLBACK] !!! NCCL fault callback fired !!! dev_idx=<dev>
[NCCL-FT] 瞬間攔截本地網卡故障，標記 dev_idx: <dev> mask=0x...
[NCCL-FT] Watchdog: clearing poisoned work for device <dev>
[NCCL-FT] Shadow restore enqueued for seq=<N>

[2PC 協商期]
[NCCL-FT] Side-car wrote per-rank propose key: NCCL_FT_PROPOSE_<rank>
[NCCL-FT] (rank0) All per-rank proposals received; min_target_op=<T> min_shadow_seq=<S>
[NCCL-FT] All ranks ACKed; COMMIT written for OP <T> agreed_shadow_seq=<S>
[NCCL-FT] final_commit_op_ set to <T>

[排水期]
[NCCL-FT] Commit 感知: commit_op=<T> agreed_ss=<S> current_op=<X>  (X > S+1)
[NCCL-FT] 排水多餘 op <X> (agreed_ss=<S>)，返回空 work   (重複 X-S-1 次)

[降級重建期]
[NCCL-FT] 抵達重播點 OP=<S+1>，執行拓撲重建
[NCCL-FT] proxy_global_comm_ ready: ...
[NCCL-FT] TCPStore: deleted per-rank keys for OP=<T>

[Shadow Ping-Pong 執行期]
[NCCL-FT] Sub-Task 5 REPLAY after rollback for seq=<S>
[NCCL-FT] execute_shadow_allreduce: role=FAULTY/PROXY/HEALTHY ...
[NCCL-FT][FAULTY] Steps 1+4 enqueued via proxy=<proxy>
[NCCL-FT][PROXY] All steps enqueued for wards=[<wards>]
[NCCL-FT][HEALTHY] ncclAllReduce enqueued on proxy_global_comm_.

[穩定降級期（後續 ops）]
[NCCL-FT] Degraded mode active, opType=ALLREDUCE, rank=<R>
[NCCL-FT] execute_shadow_allreduce: role=... seq=<S+1>
```

---

## 九、已知不在 Prototype 範圍的項目

| 項目 | 原因 |
|---|---|
| AllGather 降級路徑 | DDP only prototype；FSDP/ZeRO 為 P0 後續工作 |
| ReduceScatter 降級路徑 | 同上 |
| Broadcast 降級路徑 | 使用原始 comm 的 warning 已實作 |
| Error type 分類 | 目前所有 NCCL error 視為 NIC 故障；精細分類為後續工作 |
| Proxy 負載均衡 | 固定 `(failed+1) % N`；動態選擇為後續工作 |
| Tensor shard fan-out | 單一代理，非多路廣播；效能優化為後續工作 |
| Per-Server NIC Pool（Bug 13）| Prototype 後進階功能，Step 10 |
