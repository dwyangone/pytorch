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

## 二、已確認的 Bug 與缺口

### 已修復（不需再處理）
- ✅ Bug 1: Watchdog 提前寫入 COMM_ERROR
- ✅ Bug 2: 2PC barrier spin-wait 位置在 pre() 之後
- ✅ Bug 4+5: target_op 計算錯誤 / seqCollective_ 非 atomic 跨執行緒讀取
- ✅ Bug 6: intraNodeComm_ bypass 繞過降級路徑
- ✅ Bug 7: PROXY step 2 ATen op 在 ncclRecv 完成前執行
- ✅ Bug 9: pending_shadow_seq_ 初始 sentinel 為 0
- ✅ Bug 10: work->ncclComm_ 在降級路徑指向壞掉的 comm
- ✅ Bug 11: last_handled_event 去重邏輯用字串比較而非 target_op 數值
- ✅ Bug A (SIGSEGV): Watchdog 對 work.outputs_ 的 nullptr 解引用

---

> **2025-07 更新：Shadow Buffer OOM 修復**
> `ensure_shadow_buffer()` 改為分配 **Pinned CPU 記憶體**（非 GPU HBM）。
> 對 `model_size_gb=16` 的訓練腳本，此修改節省 16 GiB GPU 記憶體，解決 OOM 問題。
> Checkpoint (GPU→CPU) 和 Restore (CPU→GPU) 均使用 `cudaMemcpyAsync` via `copy_(non_blocking=true)`。

---

### Bug 12 (P2): multi-NIC 同時故障時 `local_hardware_fault_dev_` 遺失後續故障

**問題描述：**
`local_hardware_fault_dev_` 是 `atomic<int>`，只能存一個值。
若 NIC 0 和 NIC 1 幾乎同時故障：
- 第一次 CAS 成功（例如 NIC 0 寫入）
- 第二次 CAS 失敗（`expected=-1` 但實際已是 0），NIC 1 的故障被靜默丟棄
- Watchdog 的 fallback CAS 也會同樣失敗

同時，PROPOSE 訊息的 `dev_idx` 欄位只有一個整數，無法一次傳遞多個故障裝置。

**修復策略：**
- 將 `local_hardware_fault_dev_` 改為 `atomic<uint64_t>` bitmask，使用 `fetch_or` 而非 CAS
- bit `i` 表示本地裝置 `i` 發生故障
- PROPOSE 訊息的 `dev_idx` 欄位由一個整數改為十六進位 bitmask 字串（例如 `0x5` = NIC 0 和 NIC 2 同時故障）
- 收到 PROPOSE 時，解析 bitmask 並將所有對應 bit 的裝置都加入 `faulty_local_devs_`

---

### Bug 13 (P1): 非對稱降級（per-server NIC pool）設計分析

**現況的對稱降級問題：**
目前 `rebuild_shadow_ping_pong_topology()` 假設所有 server 上發生故障的本地裝置索引完全相同（symmetric degradation）：所有 node 都從 `faulty_local_devs_` 中讀取同一組本地 rank 並排除。

但當 Server 0 的 NIC 2 故障、Server 1 的 NIC 3 故障時（不同本地索引），目前行為會讓**所有 node 都丟棄本地 2 和本地 3**，造成過度降級（每個 server 多損失一張健康的 GPU）。

**Per-Server NIC Pool 策略：**
- 每個 server 獨立維護自己的健康 NIC 集合
- `proxy_global_comm_` 的 size 由各 server 最小健康 NIC 數決定
- 不同 server 可以有不同的故障 NIC，只要每個 server 能提供足夠的健康 GPU 參與就行

**ncclCommSplit 的限制：**
`ncclCommSplit` 是 **collective 操作**，所有 rank 必須同時呼叫，傳入的 `color` 決定是否參與。
當不同 server 有不同的健康 NIC 組合時，需要一個全域共識機制來決定「哪些 global rank 參與新的 proxy_global_comm_」。

**分析結論：**
Per-Server NIC Pool 在數學上可行，但需要以下額外工作：
1. PROPOSE 訊息需攜帶 `(node_id, dev_mask)` 而非 `(node_id, dev_idx)`
2. 收到所有節點的故障資訊後（2PC COMMIT 前），rank 0 計算全域 color 表（哪些 global rank 應 color=1 / NOCOLOR）
3. 計算結果寫入 TCPStore，所有 rank 在 barrier 處讀取後再呼叫 ncclCommSplit

Prototype 階段（2 × 8 環境）的簡化做法：仍使用對稱降級（Step 7），後續再實作非對稱（Step 8）。

---

## 三、NCCL Custom Source 端的必要修改

以下修改需由使用者在 custom NCCL fork 中完成（不在本計劃的 PyTorch 修改範圍內）：

| 修改點 | 原因 |
|---|---|
| `ncclIbResiliencyHandleDeviceFailure`: `ncclSystemError` → `ncclRemoteError` | FT callback 路徑需要 ncclRemoteError 而非 system error，避免直接 abort |
| `ncclIbResiliencyProbeHandleCompletionEvent`: `ncclSuccess` → `ncclRemoteError` | 同上 |
| `topo.cc ncclTopoPopulateNics`: 移除 ban check block | 避免 NCCL 自行在 topo 掃描時過濾掉已 ban 的 NIC |
| `init.cc`: 新增 `ncclCommBanNicReset()` API | 支援從 PyTorch 側重置 ban 狀態 |
| `init.cc nccl_ft_trigger_fault`: 呼叫 PyTorch 的 `trigger_fault_proposal` callback | NCCL 偵測到 IB 故障時通知 PyTorch |

---

## 四、端到端測試前的最小修復清單（Must-Fix Before Testing）

| Step | Bug | 狀態 |
|---|---|---|
| Step 1 | Bug 4+5: target_op 計算 | ✅ 已完成 |
| Step 2 | Bug 1: Watchdog COMM_ERROR 提前設定 | ✅ 已完成 |
| Step 3 | Bug 2: 2PC barrier 移到 pre() 之前 | ✅ 已完成 |
| Step 4 | Bug 10: work->ncclComm_ 降級路徑 | ✅ 已完成 |
| Step 5 | Bug 6: intraNodeComm_ bypass | ✅ 已完成 |
| Step 6 | Bug 9: pending_shadow_seq_ sentinel | ✅ 已完成 |

---

## 五、實作計劃（有序執行）

```
[Step 1] Bug 4+5: 修復 target_op 計算 → 讓 2PC barrier 在正確的 op 觸發          ✅ 已完成
[Step 2] Bug 1:   修復 Watchdog COMM_ERROR 提前設定 → 避免假性錯誤傳播            ✅ 已完成
[Step 3] Bug 2:   移動 2PC barrier 到 pre() 之前 → 正確的 shadow checkpoint 時序   ✅ 已完成
[Step 4] Bug 10:  修復 work->ncclComm_ 在降級路徑 → 避免 Watchdog 誤判 replay work ✅ 已完成
[Step 5] Bug 6:   intraNodeComm_ bypass 修復 → 確保降級後 allreduce 走正確路徑    ✅ 已完成
[Step 6] Bug 9:   pending_shadow_seq_ sentinel 修復 → 首個 op 故障時正確運作        ✅ 已完成
[Step 7] Bug 12:  multi-NIC 同時故障 bitmask 修復                                  ⏳ 待處理
[Step 8] NCCL 端: 修復 FT init hook 自動 ban NIC（需使用者在 custom NCCL 修改）    ⏳ 待處理
[Step 9] 驗證:   2-node × 8-GPU 端到端測試，注入單 NIC 故障，觀察訓練繼續執行     ⏳ 待處理
[Step 10] Bug 13: Per-Server NIC Pool 非對稱降級（進階功能，Prototype 後再做）      ⏳ 待處理
```

---

## 六、Step 7 詳細實作計劃：Bug 12 Multi-NIC Bitmask 修復

### 6.1 Sub-Task A：HPP 變數型別修改

**Intent:** 將單一整數的故障信號改為 bitmask，讓多張 NIC 同時故障時不丟失任何一個。

**Relevant Context:**
- `ProcessGroupNCCLFT.hpp` line ~1087: `std::atomic<int> local_hardware_fault_dev_{-1};`

**Todo List:**
1. 將 `local_hardware_fault_dev_` 型別由 `std::atomic<int>` 改為 `std::atomic<uint64_t>`，初始值改為 `0`（0 表示無故障）
2. 更新 HPP 中的說明註解：bit `i` 表示本地裝置 `i` 發生故障

**Expected Outcomes:**
- 多個 NIC 同時故障時，所有 bit 都被記錄，無任何靜默丟棄

---

### 6.2 Sub-Task B：`trigger_fault_proposal()` 改用 `fetch_or`

**Intent:** 讓 NCCL callback 和 Watchdog 都能透過 fetch_or 安全寫入多個故障 bit，不互相覆蓋。

**Relevant Context:**
- `ProcessGroupNCCLFT.cpp` lines 4349-4370: `trigger_fault_proposal(int dev_idx)`

**Todo List:**
1. 移除 CAS 邏輯，改為 `local_hardware_fault_dev_.fetch_or(1ULL << dev_idx, std::memory_order_release)`
2. 更新 `pending_shadow_seq_` 快照邏輯：只在 `pending_shadow_seq_` 還是 sentinel（UINT64_MAX）時才寫入（避免第二次故障覆蓋第一次的 seq）

**Expected Outcomes:**
- 連續兩次 `trigger_fault_proposal(0)` + `trigger_fault_proposal(1)` 後，bitmask = `0x3`

---

### 6.3 Sub-Task C：Side-car negotiator 讀取 bitmask、組裝並解析新格式 PROPOSE

**Intent:** PROPOSE 訊息需攜帶整個 bitmask（所有同時故障的 NIC），讓所有 node 在一輪 2PC 內收到完整故障集合。

**Relevant Context:**
- `ProcessGroupNCCLFT.cpp` lines ~4430-4460: 組裝 PROPOSE 字串的程式碼
- `ProcessGroupNCCLFT.cpp` lines ~4470-4490: 解析 PROPOSE 的 `sscanf` 呼叫
- 目前協議格式：`"PROPOSE:<target_op>:<node_id>:<dev_idx>:<shadow_seq>"`

**Todo List:**
1. **讀取：** 讀取 `local_hardware_fault_dev_` 的完整 bitmask（`uint64_t`），用 `exchange(0)` 一次清空，避免重複 relay
2. **組裝：** 將格式由 `:<dev_idx>:` 改為 `:<dev_mask_hex>:`（例如 `0x5`），使用 `std::hex`
3. **清除：** 在 relay 完成後不需要額外清除（已在 step 1 用 exchange(0) 清空）
4. **解析：** 將接收端的 `sscanf` 改為解析十六進位 bitmask；對 bitmask 的每個 set bit 都呼叫 `faulty_local_devs_.insert(bit_index)`

**Expected Outcomes:**
- 格式範例：`"PROPOSE:42:0:0x6:1234"` 表示 node 0 的 NIC 1 和 NIC 2 同時故障
- 所有 rank 收到後，`faulty_local_devs_` 中同時包含 1 和 2

---

### 6.4 Sub-Task D：Watchdog fallback 路徑改用 `fetch_or`

**Intent:** Watchdog 在掃描 NCCL error 時，若發現故障是 NIC 問題，需要透過同一個 bitmask 通知 negotiator。

**Relevant Context:**
- `ProcessGroupNCCLFT.cpp` Watchdog FT recoverable path（lines ~2420-2572）：目前有 CAS fallback 寫入 `local_hardware_fault_dev_`

**Todo List:**
1. 找出所有 Watchdog 中寫入 `local_hardware_fault_dev_` 的地方
2. 將 CAS (`compare_exchange_strong`) 改為 `fetch_or(1ULL << dev_idx)`
3. 確認 Watchdog 中 `trigger_fault_proposal()` 的呼叫已使用新 bitmask 邏輯

**Expected Outcomes:**
- Watchdog 發現故障時，`local_hardware_fault_dev_` 的對應 bit 被設置
- 不影響同時由 NCCL callback 設置的其他 bit

---

## 七、端到端測試驗證清單

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

## 八、已完成項目（不需再處理）

- ✅ `initLocalNvlinkComm()` — ncclCommSplit 方式，已驗證邏輯正確
- ✅ `rebuild_shadow_ping_pong_topology()` — symmetric degradation，multi-fault 集合
- ✅ `execute_shadow_allreduce()` — 4-step relay，AVG 修正，multi-ward 支援
- ✅ `trigger_fault_proposal()` — atomic CAS（Step 7 後改為 fetch_or），pending_shadow_seq snapshot
- ✅ 2PC 完整協議 — PROPOSE/ACK/COMMIT，shadow_seq 共識（min 策略）
- ✅ Shadow Buffer — checkpoint/restore，grow-on-demand，replay_pending guard
- ✅ SIGSEGV Bug A 修復 — nullptr guard on `work.outputs_`
- ✅ seqCollective_ double-count 修復 — replay 前 decrement
- ✅ TCPStore key cleanup — 每輪故障後清理
- ✅ Multi-fault `faulty_local_devs_` unordered_set

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
