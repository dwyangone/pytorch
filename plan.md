# ProcessGroupNCCLFT — Shadow Ping-Pong Failover: Complete Implementation Plan

---

## 一、系統架構總結

ProcessGroupNCCLFT 是基於 ProcessGroupNCCL 複製並擴充的自訂 backend（已在 c10d 中以 `nccl_ft` 名稱註冊）。

### 三執行緒模型
- **主執行緒**：執行 collective（AllReduce 等），故障後在 2PC barrier 停下等待
- **Watchdog**（`pt_nccl_watchdg`）：偵測失敗的 work、清除 poison、觸發 shadow restore
- **Side-car negotiator**（`pt_nccl_ft_side`）：透過 TCPStore 2PC 協商故障邊界

### 硬體假設
- 2 台 server × 8 GPU（rank 0-7 on server 0, rank 8-15 on server 1）
- 每張 GPU 對應一張 NUMA-local NIC（mlx5_0~mlx5_7）
- 同 server 內所有 GPU 以 NVLink 互連
- `torchrun --nproc_per_node=8 --nnodes=2`

---

## 二、三大核心拼圖（已確認程式碼完成）

### 拼圖 1：initLocalNvlinkComm() ✅
- 位置：`ProcessGroupNCCLFT.cpp` lines 3998–4059
- 使用 `ncclCommSplit(color=node_id)` 從 global comm 切割純 NVLink sub-communicator
- lazy init（第一次 collective 觸發），避免 constructor 中 global comm 尚未建立
- `NCCL_HAS_COMM_SPLIT` 缺失時記錄 WARNING 並優雅降級

### 拼圖 2：rebuild_shadow_ping_pong_topology() ✅
- 位置：`ProcessGroupNCCLFT.cpp` lines 4075–4175
- 對稱降級：所有 node 的相同 local_rank 一起退出跨節點群組
- 健康 rank 傳入 `color=1`，故障 rank 傳入 `NCCL_SPLIT_NOCOLOR`
- 用 `ncclCommSplit` 而非 `ncclCommInitRank`，繼承已驗證的拓撲，避免重新掃描 NIC
- 正確計算 `proxy_comm_rank_` / `proxy_comm_size_`

### 拼圖 3：execute_shadow_allreduce() 四步驟 relay ✅
- 位置：`ProcessGroupNCCLFT.cpp` lines 4207–4394
- FAULTY：Step 1 NVLink send；Step 4 NVLink recv
- PROXY：Step 1 recv wards；Step 2 pre-aggregate；Step 3 cross-node AllReduce；Step 4 send back
- HEALTHY：直接 cross-node AllReduce on `proxy_global_comm_`
- AVG workaround：`ncclSum` + 手動除以原始 world size
- `collective()` 中的攔截邏輯（lines 5155–5208）：replay path + 常態降級 path

---

## 三、2PC Barrier 機制（已確認程式碼完成）

### Step A：Per-Rank PROPOSE Key（Bug C 修復）✅
- 位置：`start_ft_negotiator_thread()` lines 4477–4530
- 每個 rank 寫自己的 `NCCL_FT_PROPOSE_<rank>_<round>`，不會互相覆寫
- 無本地故障的 rank 偵測到其他 rank 已 PROPOSE 後，寫入 `0x0` 佔位 key（lines 4562–4600）
- rank 0 coordinator 等所有 `size_` 個 key 齊全後，計算 `min(shadow_seq)` 並寫入
  `NCCL_FT_COMMIT_<round>` + `NCCL_FT_SS_AGREED_<round>`（lines 4671–4688）
- 所有 rank 讀取 COMMIT 後，先 release-store `committed_shadow_seq_`，
  再 release-store `final_commit_op_ = cur_round+1`，保證 acquire ordering 正確

### Step B：Commit 感知 Drain + Rebuild（Bug B 修復）✅
- 位置：`collective()` lines 4958–5061
- 用 `final_commit_op_ > 0` 替換原本 `current_op == boundary` 精確匹配
- `current_op > agreed_ss + 1`：DRAIN path，靜默丟棄多餘 op，回傳 NullWork
- `current_op == agreed_ss + 1`：REBUILD path，呼叫 `rebuild_shadow_ping_pong_topology()`

### Step C：seqCollective_ 重置對齊 ✅
- REBUILD path 後立即 `seqCollective_ = agreed_ss`（line 5018）
- 讓後續 replay path 的 `seqCollective_--`（line 5162）正確落地於 `agreed_ss`

### Step D：NullWork — 讓 DDP wait() 立即返回 ✅
- DRAIN path 中 `work->ncclEndEvent_->record(ncclStream)` + `work->future_->markCompleted()`（lines 4992–5003）

### Step E：rollback_done_ 防重入 + 多輪重置 ✅
- `rollback_done_` 在 REBUILD path 設為 `true`，防止同一輪故障重複 rebuild（line 5020）
- replay 完成後 reset（line 5179）
- 無 replay 的安全路徑 reset（lines 5141–5151）
- HPP 中已宣告（`bool rollback_done_{false}`）

---

## 四、其他已修復的 Bugs

| Bug | 位置 | 狀態 |
|-----|------|------|
| Bug A：Watchdog SIGSEGV（work.outputs_ nullptr）| Watchdog FT path | ✅ |
| Bug 2：seqCollective_ double-count | replay path seqCollective_-- | ✅ |
| Bug 10：work->ncclComm_ 仍指向壞掉的 global comm | lines 5247–5272 | ✅ |
| Bug 12：multi-NIC bitmask (fetch_or) | trigger_fault_proposal + Watchdog | ✅ |
| Fix 1：fault callback 每個 rank 各自獨立呼叫 | lazy init，移除 ft_root_comm_ guard | ✅ |
| Fix 3 base：Watchdog early-abort（2PC commit 已知） | lines 2454–2479 | ✅ |
| Shadow Buffer → pinned CPU 記憶體（OOM fix） | ensure_shadow_buffer() | ✅ |
| Shadow copy stream 分離（並行 D2H） | shadow_copy_stream_ | ✅ |
| TCPStore key cleanup | REBUILD path，lines 5027–5050 | ✅ |
| `ft_round_` 命名空間隔離多輪故障 | side-car Task 1 + REBUILD path | ✅ |

---

## 五、剛剛完成的修復：Fix 3 安全性補丁

### 問題（已解決）

Fix 3 原本的實作在 `final_commit_op_ > 0` 時會 force-fail **所有** in-flight work，
包括 rollback 後才提交的正常 work（seq > ckpt_seq）。

在 REBUILD path 執行 `ncclCommSplit`（集體呼叫，耗時數十 ms）期間，
`final_commit_op_` 尚未被 reset 到 0。若此時 Watchdog 輪詢到一個 seq > ckpt_seq 的
新 work，它會被錯誤地 force-fail，導致訓練立即崩潰。

### 修復（剛才已套用至 lines 2454–2481）

```cpp
if (!pg_->ft_disabled_ && !work.exception()) {
    uint64_t commit_signal =
        pg_->final_commit_op_.load(std::memory_order_acquire);
    if (C10_UNLIKELY(commit_signal > 0)) {
        uint64_t ckpt_seq =
            pg_->committed_shadow_seq_.load(std::memory_order_acquire);
        // Only abort work that predates or is at the checkpoint seq.
        // work.seq_ > ckpt_seq means it was submitted after the rollback
        // point — it is a post-fault op that must NOT be force-failed.
        if (work.seq_ <= ckpt_seq) {
            // ... force-fail logic ...
        }
    }
}
```

---

## 六、端到端測試前的剩餘 Gaps（有序）

### Gap 1（P0）：`pending_shadow_seq_` UINT64_MAX 轉換產生的 agreed_ss = 0 ✅ 已修復

**問題描述：**

在 `start_ft_negotiator_thread()` 的 PROPOSE 組裝中（Task 1 和 no-fault placeholder），
若 `pending_shadow_seq_` 仍為哨兵 `UINT64_MAX`，程式碼將其轉換為 `0`：

```cpp
if (my_shadow_seq == UINT64_MAX) {
    my_shadow_seq = 0;  // <--- 轉換為 0
}
```

然後在 coordinator 聚合時：

```cpp
if (p_ss != UINT64_MAX) {
    agreed_ss = min(agreed_ss, p_ss);  // 0 進來 → agreed_ss = 0
}
```

**後果：** 若任一 rank 尚未完成任何 AllReduce checkpoint（shadow_seq 從未被設定），
`agreed_ss` 就會是 `0`。DRAIN + REBUILD path 以此作為基準，
所有 op 都被排水，然後 replay 試圖重放 seq=0 — 但 seq=0 從未被 checkpoint，
shadow_buf_ 中無有效資料。

**修復方案：**

UINT64_MAX 哨兵的含義是「此 rank 無 checkpoint」。coordinator 應只考慮有真實
checkpoint 的 rank（`p_ss != UINT64_MAX`，且 `p_ss > 0`）來計算 `agreed_ss`。
若所有 rank 都無 checkpoint（例如故障發生在第一次 AllReduce 之前），
`agreed_ss` 應保持 `UINT64_MAX`，REBUILD path 偵測到此情況後應跳過 replay
（無需 restore，只需重建拓撲然後直接繼續）。

**需修改的位置：**

1. `start_ft_negotiator_thread()` 中 no-fault placeholder 的組裝（line ~4581）：
   不要將 `UINT64_MAX` 轉換為 `0`，保留 `UINT64_MAX` 讓 coordinator 正確識別。
   寫入字串時用一個特殊值（例如 `18446744073709551615` 即 `UINT64_MAX` 本身）。

2. `start_ft_negotiator_thread()` 中 Task 1 的組裝（line ~4506）：
   同上，保留 `UINT64_MAX` 而非轉換為 `0`。

3. coordinator 聚合邏輯（line ~4634）：
   ```cpp
   // 舊：
   if (p_ss != UINT64_MAX) {
       agreed_ss = min(agreed_ss, p_ss);
   }
   // 新：
   if (p_ss != UINT64_MAX && p_ss > 0) {
       agreed_ss = min(agreed_ss, p_ss);
   }
   // agreed_ss == UINT64_MAX 代表無任何 rank 有 checkpoint
   ```

4. REBUILD path 中（line ~5012）：
   ```cpp
   if (agreed_ss == UINT64_MAX) {
       // 無 checkpoint，只需重建拓撲，不需 replay。
       LOG(INFO) << "[NCCL-FT] No checkpoint available; topology rebuild only.";
       rebuild_shadow_ping_pong_topology();
       is_degraded_ = true;
       rollback_done_ = false;  // 不需 replay，直接清除
       // ... reset final_commit_op_, ft_round_++ ...
       return;  // 或 continue 到正常 fn() 路徑
   }
   ```

---

### Gap 2（P0）：`sscanf` format string vs. `0x` prefix

**問題描述：**

PROPOSE 字串在 Task 1 的格式為：
```
"PROPOSE:0:0:0x1:83"
```

但 `sscanf` 使用 `%lx`，而 `%lx` 在 C 標準中會跳過 `0x` 前綴（這是正確的）。
測試驗證：`sscanf("0x1", "%lx", &val)` → `val = 1`，無問題。

**結論：** 此項無需修改，sscanf 格式正確。

---

### Gap 3（P1）：rebuild_shadow_ping_pong_topology() 呼叫時機的集體性保證

**問題描述：**

`ncclCommSplit` 是 NCCL 集體呼叫（collective call）——所有參與 comm 的 rank 必須
「同時」呼叫它。目前的設計是每個 rank 的主執行緒在 `collective()` 的 REBUILD path
中呼叫 `rebuild_shadow_ping_pong_topology()`。

由於 `final_commit_op_` 是由 side-car 設定（所有 rank 的 side-car 都在讀到
`NCCL_FT_COMMIT_<round>` 後立即設定），不同 rank 的主執行緒抵達 REBUILD path 的
時間可能相差數個 DDP bucket 的 collective 呼叫量（取決於 DDP overlap 的深度）。

**現況分析：**

- REBUILD path 以 `current_op == agreed_ss + 1` 為條件進入
- 所有 rank 的 `final_commit_op_` 在 COMMIT 之後都被設定
- 但不同 rank 的主執行緒可能因為排水不同數量的 op 而在不同時刻呼叫 `rebuild_shadow_ping_pong_topology()`

**影響評估：**

若 NCCL 的非阻塞 comm（`NCCL_HAS_COMM_NONBLOCKING`）已啟用，`ncclCommSplit` 是流程式
（不阻塞 CPU），各 rank 呼叫時間差影響較小。若使用 blocking=1（目前設定），
先呼叫的 rank 會在 NCCL 層面等待所有 rank 都呼叫後才返回。

**結論：** blocking=1 模式下，先到的 rank 會等後到的 rank，自然形成 barrier，
不需額外同步。可以在 Prototype 測試中觀察是否有 hang。若有，再加明確 TCPStore barrier。

---

### Gap 4（P1）：DRAIN path 中 `work->outputs_` 內容的正確性

**問題描述：**

DRAIN path 返回的 NullWork 中，`work->outputs_` 是在 `collective()` 開頭從 `outputs`
複製的（line 4900）。`outputs` 此時存放的是 DDP 的梯度 tensor（被中斷的那個 bucket）。

被排水的 op 並未執行任何 AllReduce，所以 `outputs` 中的梯度值是故障前的舊值（或部分計算結果）。
DDP 在 `wait()` 後會使用這個值更新參數。

**影響：**

這是 **數學上不正確** 的：被排水的 buckets 中的梯度沒有 AllReduce，參數更新不同步。
但這些 op 的 seq > agreed_ss，它們本不該被執行——它們是故障期間的「髒 op」。
DDP 在 replay 完成後會以正確的 AllReduce 結果重新發起更新（Prototype 假設 DDP
會在 Watchdog 清除 exception 後重新執行這些 buckets）。

**Prototype 接受度：** 暫時接受。如 DDP 不會自動重新執行，需在排水 path 中清零
`work->outputs_` 的梯度（`at::zero_()` on each tensor）。

---

### Gap 5（P0）：rebuild 後的 global comm（壞掉的 comm）狀態

**問題描述：**

`rebuild_shadow_ping_pong_topology()` 中呼叫 `ncclCommSplit(globalComm, color, ...)` 時，
`globalComm` 是已經報錯的 comm（狀態為 `ncclRemoteError` 或更嚴重）。

**NCCL 行為：** `ncclCommSplit` 使用的 parent comm 如果已經 abort，會返回
`ncclInvalidUsage` 或直接 crash。

**診斷：** 程式碼中已加入診斷 log（line 4142）：
```
LOG(INFO) << "[NCCL-FT] globalComm isAborted=" << globalComm->isAborted()
          << " before ncclCommSplit";
```

**期望行為：**
- `ncclRemoteError` 狀態的 comm：NCCL 仍允許 `ncclCommSplit`（取決於 NCCL 版本）
- 若 `isAborted() == true`：`ncclCommSplit` 必然失敗

**需確認的 NCCL 行為（在 custom NCCL fork 中）：**
確認 `ncclRemoteError` 後的 comm 是否可以呼叫 `ncclCommSplit`，
還是需要先呼叫 `ncclCommAbort` + `ncclCommInitRank` 重新初始化。

若 NCCL fork 的實作允許 RemoteError 後的 comm 繼續做 Split（只有傳輸路徑壞掉），
則目前設計正確。若不允許，需要：
1. 在 REBUILD path 前先 `ncclCommAbort(globalComm->getNcclComm())`
2. 用 `ncclCommInitRank` 重新建立一個不走故障 NIC 的 comm
（但這會觸發 NIC 重新掃描，需確認 custom NCCL 的 `ncclTopoPopulateNics`
是否已正確過濾 banned NIC）

---

### Gap 6（P1）：DDP 的 exception handling 路徑

**問題描述：**

當 Watchdog force-fail 一個 work（`work.setException(...)`）後，
DDP 的 `Reducer::mark_variable_ready_dense()` 最終會呼叫 `work->wait()`，
此時 `wait()` 會 rethrow 這個 exception。

PyTorch DDP 在 rethrow exception 後的行為取決於版本和設定：
- 新版 DDP：呼叫 `process_group_->abort()` 然後 re-raise（訓練終止）
- 舊版 DDP：直接 re-raise（用戶的 training loop 需 try-catch）

**Prototype 假設：** 用戶的 training loop 有 try-catch 保護，捕獲 exception 後
繼續下一個 batch。若訓練 loop 在 rethrow 後終止，Shadow Ping-Pong 無法發揮作用。

**需要的測試配置：** 訓練腳本的 main loop 要有：
```python
try:
    loss.backward()
    optimizer.step()
except Exception as e:
    if "FT" in str(e) or "DistBackendError" in str(e):
        continue  # FT handled, skip this batch
    raise
```

---

## 七、端到端測試步驟

### 測試前確認清單

```
[x] initLocalNvlinkComm()             — lines 3998–4059
[x] rebuild_shadow_ping_pong_topology() — lines 4075–4175
[x] execute_shadow_allreduce() 四步驟 — lines 4207–4394
[x] AllReduce 攔截 degraded/replay     — collective() lines 5119–5208
[x] Step A: per-rank PROPOSE key       — lines 4477–4530
[x] Step B: drain + rebuild barrier    — lines 4958–5061
[x] Step C: seqCollective_ align       — line 5018
[x] Step D: NullWork DRAIN             — lines 4992–5003
[x] Step E: rollback_done_ lifecycle   — lines 5020, 5148, 5179
[x] Fix 1: callback registration       — lines 4818–4841
[x] Fix 3: Watchdog early-abort (safe) — lines 2454–2481
[x] Gap 1: UINT64_MAX sentinel kept; no-checkpoint fast-path in REBUILD block
[ ] Gap 5: 確認 ncclCommSplit on RemoteError comm          ← 待確認
[ ] Gap 6: training loop exception 保護                    ← 待測試腳本確認
```

### 期望 Log 順序（故障後）

```
[初始化]
[NCCL-FT] Set NCCL_IB_HCA=mlx5_<local_rank>
[NCCL-FT] First collective detected; registering fault callback...
[NCCL-FT] Fault callback registered successfully on comm ...
[NCCL-FT] local_nvlink_comm_ ready: ...

[故障觸發]
[NCCL-FT-CALLBACK] !!! NCCL fault callback fired !!! dev_idx=<D>
[NCCL-FT] 瞬間攔截本地網卡故障，標記 dev_idx: <D> mask=0x...
[NCCL-FT] Watchdog early-abort: 2PC committed (commit_signal=... ckpt_seq=...), force-clearing work seq=<N> ...

[2PC 協商]
[NCCL-FT] Side-car wrote per-rank propose key: NCCL_FT_PROPOSE_<rank>_0 val=PROPOSE:0:...
[NCCL-FT] Side-car wrote no-fault propose key: NCCL_FT_PROPOSE_<rank>_0   (非故障 rank)
[NCCL-FT] All 16 per-rank proposals received for round=0 agg_fault_mask=0x... agreed_ss=<S>
[NCCL-FT] (rank 0) COMMIT written for round=0 agreed_shadow_seq=<S>
[NCCL-FT] committed_shadow_seq_=<S> for round=0
[NCCL-FT] final_commit_op_ set to 1; main thread will proceed.

[排水]
[NCCL-FT] Commit 感知: commit_signal=1 agreed_ss=<S> current_op=<X>  (X > S+1)
[NCCL-FT] 排水多餘 op current_op=<X> (agreed_ss=<S>); seqCollective_ restored to ...  (重複 X-S-1 次)

[降級重建]
[NCCL-FT] 抵達重播點 OP=<S+1> (agreed_ss=<S>)，執行拓撲重建
[NCCL-FT] globalComm isAborted=false before ncclCommSplit   ← 關鍵診斷
[NCCL-FT] proxy_global_comm_ ready: ... (proxy_rank=..., proxy_size=14)
[NCCL-FT] TCPStore: deleted per-rank propose key for round=0

[Shadow Ping-Pong]
[NCCL-FT] Sub-Task 5 REPLAY after rollback for seq=<S>
[NCCL-FT] execute_shadow_allreduce: role=FAULTY/PROXY/HEALTHY ...
[NCCL-FT][FAULTY] Steps 1+4 enqueued via proxy=<P>
[NCCL-FT][PROXY] All steps enqueued for wards=[<W>]
[NCCL-FT][HEALTHY] ncclAllReduce enqueued on proxy_global_comm_.
[NCCL-FT] rollback_done_ reset (replay complete).

[穩定降級期]
[NCCL-FT] Degraded mode active, opType=ALLREDUCE, rank=<R>
[NCCL-FT] execute_shadow_allreduce: role=... seq=<S+1>
```

---

## 八、NCCL Custom Fork 端的必要確認

| 項目 | 狀態 |
|------|------|
| `ncclIbResiliencyHandleDeviceFailure`: `ncclSystemError → ncclRemoteError` | ✅ 已完成 |
| `ncclIbResiliencyProbeHandleCompletionEvent`: `ncclSuccess → ncclRemoteError` | ✅ 已完成 |
| `topo.cc ncclTopoPopulateNics`: 移除 ban check | ✅ 已完成 |
| `init.cc`: 新增 `ncclCommBanNicReset()` | ✅ 已完成 |
| `init.cc nccl_ft_trigger_fault`: 呼叫 `trigger_fault_proposal` callback | ✅ 已完成 |
| **`ncclCommSplit` 在 `ncclRemoteError` comm 上的行為** | ❓ 待確認 |

---

## 九、後續工作（Prototype 後）

| 項目 | 說明 |
|------|------|
| Gap 1 pending_shadow_seq_ 修復 | P0 — 第一次測試前必做 |
| Gap 5 ncclCommSplit on RemoteError comm 確認 | P0 — 第一次測試前必做 |
| Gap 6 training loop exception 保護 | P0 — 測試腳本需要 |
| Bug 13: Per-Server NIC Pool 非對稱降級 | P1 — Prototype 後 |
| AllGather/ReduceScatter 降級路徑（FSDP/ZeRO） | P1 — Prototype 後 |
| Proxy 負載均衡（動態選擇）| P2 |
| Tensor shard fan-out（多路中繼）| P2 |
| Error type 精細分類 | P2 |
