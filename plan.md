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

## 二、test.log（2026-07-31）完整診斷

### 觀察 1：new_mask 擴散到所有 8 個 rank（最關鍵的發現）

你只關掉 **mlx5_0**（一張網卡），但 log 中 rank 0~7 全部都寫入了故障 bit：

| Rank | new_mask | device_idx |
|------|----------|------------|
| rank0 | `0x1` | device 0 |
| rank1 | `0x2` | device 1 |
| rank2 | `0x4` | device 2 |
| rank3 | `0x8` | device 3 |
| rank4 | `0x10` | device 4 |
| rank5 | `0x20` | device 5 |
| rank6 | `0x40` | device 6 |
| rank7 | `0x80` | device 7 |

**根本原因：** Watchdog 的 early-abort 路徑（和正常故障清除路徑）中，`device_idx` 是從
`work.device_.index() % localDeviceCount_` 計算來的。這是每個 rank 對應的 **本地 GPU index**，
不是「哪張 NIC 故障」的資訊。

在 `ncclRemoteError` 的場景中，**所有 rank 的 comm 都回報錯誤**——因為 AllReduce 是一個集體
操作，任何一端的 NIC 故障都會讓整個 ring 上的所有 rank 的 comm 回報 `ncclRemoteError`。
所以 Watchdog 在每個 rank 上都把自己的 local device index 寫入 `local_hardware_fault_mask_`，
導致 `agg_fault_mask` 在 coordinator 聚合時包含了 8 個 bit，然後 `faulty_local_devs_` 包含
了 0~7 全部 8 個 local device index。

結果：`rebuild_shadow_ping_pong_topology()` 計算出
`proxy_comm_size_ = 16 - 8*2 = 0`，所有 rank 都是 faulty，沒有任何 rank 可以組成新的 comm。
整個機制崩潰。

### 觀察 2：ncclCommSplit 完全未執行（log 截止在 rank1 的最後一筆 work 清除）

Log 在 rank1 清除 seq=31 後結束，沒有任何 `isAborted=`、`ncclCommSplit`、
`proxy_global_comm_ ready` 等訊息。這代表：

訓練主執行緒在 `collective()` 的 REBUILD path 試圖呼叫
`rebuild_shadow_ping_pong_topology()` → `NCCLFTComm::split()` → `source->getNcclComm()` 時，
因為 `globalComm` 已經處於 `aborted_` 狀態（`ncclComm_ = nullptr`），`getNcclComm()` 內部
直接 throw `NCCLFaultToleranceError`，導致 REBUILD path 以 exception 終止，訓練 hang 住。

### 觀察 3：NCCL 硬體 cache 問題（環境變數 NCCL_IB_HCA 失效）

NCCL 在第一次 `ncclCommInitRank` 時會掃描所有可用 NIC 並建立 topo/transport graph cache。
後續如果要排除某張 NIC（例如設定 `NCCL_IB_HCA` 環境變數），在已經 init 過的 NCCL 進程中
**無法生效**，因為 NIC 清單已被 cache。

這使得「先 `ncclCommAbort` 再設新環境變數再 `ncclCommInitRank`」的方案不可行。

---

## 三、正確的故障隔離邏輯（Bug Fix A：device_idx 計算錯誤）

### 問題本質

`local_hardware_fault_mask_` 的語義應該是：「**本地哪個 GPU（local device index）對應的 NIC
發生了硬體故障**」。這個 bit 只有在以下兩種情況應該被設定：

1. **NCCL callback 路徑**（`trigger_fault_proposal`）：NCCL 直接告訴我們是哪個 `dev_idx` 故障
   → 這是正確的，callback 傳入的就是故障的 device index
2. **Watchdog 清除路徑**（`if (work.exception())`）：這裡的 `device_idx` 是 Watchdog 的 fallback，
   用於 callback 沒有觸發時。此時 Watchdog 看到的是「某個 AllReduce 失敗了」，但失敗原因是
   **對端（remote）** NIC 故障造成的 `ncclRemoteError`，不是本端 NIC 故障。

**結論：Watchdog 的 fallback 路徑不應該設定 `local_hardware_fault_mask_`**。

在 `ncclRemoteError` 的情況下：
- 真正故障的 NIC 在 server A 的 rank1（mlx5_0）
- callback 已正確觸發（log line 58-63 確認 rank1 的 callback fired，dev_idx=0）
- **Watchdog fallback 應該完全不寫 fault mask**，因為 callback 已經處理了

### 修復方案

Watchdog 清除路徑的 `fetch_or` 寫入應該移除（或加上「只在 callback 未觸發時才 fallback」的判斷）。

---

## 四、rebuild 方案選擇分析

### 方案 A：ncclCommSplit（目前使用）

**優點：** 繼承已驗證的 topo，不需要重新掃描 NIC，速度快。

**致命缺陷：** `ncclCommSplit` 要求 source (parent) comm 是 **健康且未 abort** 的狀態。
但故障後 global comm 的狀態是 `ncclRemoteError`。

- NCCL 內部對 RemoteError 的 comm 呼叫 `ncclCommSplit` 的行為是**未定義的**。
- 從你的 log 可以確認：`ncclCommSplit` 後沒有任何訊息出現，說明它要麼 hang 要麼 crash。
- `NCCLFTComm::split()` 的第一步是 `source->getNcclComm()`，如果 `aborted_` 為 true，
  直接 throw exception（`NCCLFTUtils.cpp:148-156`）。

### 方案 B：ncclCommAbort + ncclCommInitRank（重新初始化）

**問題：**
1. `ncclCommInitRank` 重新掃描所有 NIC（NCCL hardware cache 問題）
2. 需要所有 rank 重新 rendezvous（需要 TCPStore 協調新的 `ncclUniqueId`）
3. 如果 NCCL_IB_HCA 環境變數在 init 後已被 cache，設新的 HCA 不會生效

**繞過 NCCL hardware cache 的方法：**
你 custom NCCL fork 中已有 `ncclTopoPopulateNics` 移除 ban check 的修改，以及
`ncclCommBanNicReset()` API。問題是 ban 機制在 `ncclCommInitRank` 時才生效，而 topo 圖
在 init 後已 cache 在 NCCL 內部。

**可行路徑：**
- 在你的 NCCL fork 中，`ncclCommInitRank` 是否**每次都重新做 topo discovery**？
  如果是，那麼重新 init 時透過 ban 機制可以排除故障 NIC。
- 如果有 `nccl.conf` 設定 `NCCL_IB_HCA`，每次 init 都會讀取，所以可以在 init 前動態修改。

### 方案 C：ncclCommShrink（你的 NCCL fork 已有實作）

**發現：** `NCCLFTUtils.cpp:275-331` 已實作 `NCCLFTComm::shrink()`，這是 NCCL 的
`ncclCommShrink` API。

`ncclCommShrink` 的語義正好符合需求：
- 不需要 abort parent comm 再重新 init
- 直接從現有 comm 縮減，排除特定 rank
- **但需要確認：`ncclCommShrink` 在 parent comm 有 ncclRemoteError 狀態時是否可以正常呼叫**

### 方案 D（建議的最簡可行方案）：兩步驟策略

1. **在 REBUILD 前先 `ncclCommAbort` global comm**（這讓 comm 進入 clean abort 狀態）
2. **新建一個只包含健康 rank 的 comm** — 方式：所有健康 rank 通過 TCPStore rendezvous 新的
   `ncclUniqueId`，然後呼叫 `ncclCommInitRankConfig`。
3. **NCCL NIC 排除：** 在 init 前透過 `setenv("NCCL_IB_HCA", "mlx5_1,...", 1)` 設定新的 HCA 清單。
   在你的 custom NCCL fork 中，`ncclTopoPopulateNics` 的 ban check 已移除，topo discovery 是否
   在每次 `ncclCommInitRank` 時重新執行取決於實作——需要確認。

---

## 五、修復計劃（有序執行）

### Fix A（P0）：移除 Watchdog fallback 路徑的 fault mask 寫入 ✅ 已完成

**問題：** Watchdog 清除 `ncclRemoteError` poisoned work 時，每個 rank 都把自己的
local device index 寫入 `local_hardware_fault_mask_`，導致所有 GPU 都被標記為故障。

**修復（已套用）：** 移除 `pg_->local_hardware_fault_mask_.fetch_or(...)` 和相關 LOG，
同時更新 comment 說明 Watchdog 不寫 fault_mask 的原因。

`local_hardware_fault_mask_` 現在只由 `trigger_fault_proposal()`（NCCL callback 路徑）
寫入，callback 傳入的 `dev_idx` 才是真正故障的 NIC 對應的 GPU index。

---

### Fix B（P0）：ncclCommSplit → ncclCommInitRank 重新初始化方案

#### B1：在 REBUILD path 前先 abort 並 rendezvous 新 comm

**現在的問題：** `NCCLFTComm::split()` 呼叫 `source->getNcclComm()`，當 `aborted_` 為 true
時直接 throw。即使 aborted_ 不為 true，RemoteError 狀態的 comm 傳給 `ncclCommSplit` 也會失敗。

**新方案：**

1. 在 REBUILD path 開始時，先呼叫 `globalComm->abort()` 讓 comm 進入 clean 狀態
2. 所有健康 rank 透過 TCPStore 協商一個新的 `ncclUniqueId`（rank 0 生成，廣播給所有人）
3. 在 init 前透過 `setenv` 動態設定 `NCCL_IB_HCA` 排除故障 NIC
4. 呼叫 `NCCLFTComm::create()` 建立新的 `proxy_global_comm_`
5. 故障 rank 不參與（跳過 init，等待 proxy 轉發）

#### B2：NCCL hardware cache 繞過方法

你的 custom NCCL fork 中需要確認兩件事：

**確認項 1：** `ncclTopoPopulateNics` 中移除 ban check 後，每次 `ncclCommInitRank` 是否
**重新執行** topo discovery？還是只有第一次執行？

- 如果每次重新執行：在 init 前設定 `NCCL_IB_HCA=mlx5_1:mlx5_2:...:mlx5_7`（排除 mlx5_0）
  然後呼叫 `ncclCommInitRank`，新的 comm 就不會使用 mlx5_0。
- 如果有 process-level cache：需要在 NCCL fork 中加一個 API 讓 PyTorch 主動 invalidate topo
  cache，或者利用已有的 `ncclCommBanNic` 機制在 topo discovery 時過濾。

**確認項 2：** `ncclCommBanNic(dev_idx)` 的作用域是什麼？
- 如果是 per-process global ban（process-level）：在 reinit 前呼叫 `ncclCommBanNic(0)` 就可以
  讓新的 `ncclCommInitRank` 跳過 mlx5_0。
- 如果是 per-comm ban：那 ban 在 abort 後就失效了，需要在 reinit 前再呼叫一次。

#### B3：具體實作草圖（需確認 NCCL fork 行為後定稿）

```cpp
void ProcessGroupNCCLFT::rebuild_shadow_ping_pong_topology() {
    // Step 1: 確認哪些 dev 是真正故障的（只來自 callback，不來自 Watchdog fallback）
    std::unordered_set<int> faulty_devs;
    { std::lock_guard<std::mutex> lk(faulty_devs_mutex_); faulty_devs = faulty_local_devs_; }
    if (faulty_devs.empty()) { return; }

    int local_rank = rank_ % localDeviceCount_;
    bool is_faulty = (faulty_devs.count(local_rank) > 0);

    // Step 2: 取得 global comm，先 abort 讓它進入 clean 狀態
    auto globalComm = getGlobalComm();
    if (!globalComm->isAborted()) {
        globalComm->abort();
    }

    // Step 3: 健康 rank 重新 init proxy comm
    if (!is_faulty) {
        // Step 3a: 通知 NCCL 排除故障 NIC
        // (方法依 NCCL fork 行為而定：setenv 或 ncclCommBanNic)
        for (int d : faulty_devs) {
            ncclCommBanNic(d);  // 若是 process-level ban
        }

        // Step 3b: TCPStore rendezvous 新的 ncclUniqueId
        ncclUniqueId new_id;
        if (rank_ == 0) {  // coordinator 生成
            ncclGetUniqueId(&new_id);
            // 寫入 TCPStore
            globalStore_->set("NCCL_FT_NEW_ID_" + std::to_string(ft_round_), ...);
        } else {
            // 等待 TCPStore 中的 new_id
        }

        // Step 3c: 計算新的 rank 和 size（排除故障 dev 的對稱降級）
        // ... （計算 proxy_comm_rank_, proxy_comm_size_）

        // Step 3d: 呼叫 ncclCommInitRankConfig 建立 proxy_global_comm_
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.blocking = 1;
        proxy_global_comm_ = NCCLFTComm::create(proxy_comm_size_, proxy_comm_rank_,
                                                  new_id, device.index(), config);
    }
    proxy_comm_ready_.store(true, std::memory_order_release);
}
```

---

## 六、已完成的項目（程式碼確認）

| 項目 | 位置 | 狀態 |
|------|------|------|
| initLocalNvlinkComm() | lines 3998–4059 | ✅ |
| rebuild_shadow_ping_pong_topology() | lines 4075–4218 | ✅ 結構正確，但 ncclCommSplit 需換成 reinit |
| execute_shadow_allreduce() 四步驟 | lines 4207–4394 | ✅ |
| Step A–E barrier 機制 | collective() lines 4978–5114 | ✅ |
| Fix 1: callback 每 rank 獨立注冊 | lines 4818–4841 | ✅ |
| Fix 3: Watchdog early-abort (safe) | lines 2454–2481 | ✅ |
| Gap 1: UINT64_MAX sentinel | lines 4513, 4591, 4689 | ✅ |
| Gap 6: ncclEndEvent_ record before setException | lines 2486, 2639 | ✅ |
| Bug 12: multi-NIC bitmask (fetch_or) | trigger_fault_proposal | ✅ |
| Shadow buffer pinned CPU | ensure_shadow_buffer() | ✅ |

---

## 七、待修復項目（有序）

### Fix A（P0）：移除 Watchdog fallback 的 fault_mask 寫入

**位置：** `ProcessGroupNCCLFT.cpp` lines ~2625-2637

**修改：** 刪除 `pg_->local_hardware_fault_mask_.fetch_or(...)` 這段。

只依靠 NCCL callback（`trigger_fault_proposal`）寫入 `local_hardware_fault_mask_`。
Watchdog 只負責清除 exception 和觸發 shadow restore，不寫 fault mask。

---

### Fix B（P0）：確認 NCCL fork 行為，選擇 reinit 策略

**你需要在 custom NCCL fork 中確認以下兩個問題：**

**問題 1：** `ncclCommBanNic(dev_idx)` 的作用域：
- 是否是 process-level global？（即使後面再 `ncclCommInitRank` 也生效）
- 還是僅對某個特定 comm 生效？

**問題 2：** Topo discovery 是否在每次 `ncclCommInitRank` 時重新執行？
- 確認方法：呼叫 `ncclCommBanNic(0)` 後再 `ncclCommInitRank`，檢查新 comm 是否不使用 mlx5_0。

---

### Fix C（P0）：替換 rebuild_shadow_ping_pong_topology() 的實作

根據 Fix B 的確認結果，選擇以下其中一條路：

**路徑 1（若 ncclCommBanNic 是 process-level global）：**
```
1. globalComm->abort()
2. 呼叫 ncclCommBanNic(faulty_local_dev)
3. TCPStore rendezvous 新的 ncclUniqueId（rank 0 生成並廣播）
4. NCCLFTComm::create(new_size, new_rank, new_id, device, config)
5. 健康 rank 建立 proxy_global_comm_；故障 rank 跳過
```

**路徑 2（若需要 setenv 方式）：**
```
1. globalComm->abort()
2. 構建新的 NCCL_IB_HCA 字串（排除故障 NIC 的 mlx5_X）
3. setenv("NCCL_IB_HCA", new_hca_str, 1)
4. TCPStore rendezvous 新的 ncclUniqueId
5. NCCLFTComm::create(...)
```

**路徑 3（若 ncclCommShrink 在 RemoteError comm 上可用）：**
```
// 先確認 ncclCommShrink API 是否接受 RemoteError 狀態的 source comm
// 若接受，直接使用現有的 NCCLFTComm::shrink() 實作即可
1. 呼叫 NCCLFTComm::shrink(globalComm, faulty_ranks_to_exclude, config)
2. proxy_global_comm_ = shrink result
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
| **`ncclCommBanNic` 是 process-level 還是 per-comm？** | ❓ 待確認（Fix B 依賴此） |
| **Topo discovery 在每次 `ncclCommInitRank` 時是否重新執行？** | ❓ 待確認（Fix B 依賴此） |
| **`ncclCommShrink` 是否接受 RemoteError 狀態的 source comm？** | ❓ 待確認（Fix C 路徑 3 依賴此） |

---

## 九、端到端測試前的確認清單（更新）

```
[x] initLocalNvlinkComm()
[x] execute_shadow_allreduce() 四步驟
[x] Step A–E barrier 機制
[x] Fix 1, Fix 3, Gap 1, Gap 6
[ ] Fix A: 移除 Watchdog fallback 的 fault_mask 寫入
[ ] Fix B: 確認 NCCL fork 的 ncclCommBanNic scope 和 topo cache 行為
[ ] Fix C: 替換 rebuild_shadow_ping_pong_topology() 為 reinit 方案
[ ] 第一次 end-to-end 測試：注入單張 NIC 故障，觀察 proxy_global_comm_ ready log
```

---

## 十、後續工作（Prototype 後）

| 項目 | 說明 |
|------|------|
| AllGather/ReduceScatter 降級路徑 | FSDP/ZeRO，P1 |
| Proxy 負載均衡 | 固定 `(failed+1) % N`，P2 |
| Per-Server NIC Pool 非對稱降級 | Bug 13，P1 |
| Error type 精細分類 | 目前所有 ncclRemoteError 視為 NIC 故障，P2 |
