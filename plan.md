# ProcessGroupNCCLFT — Shadow Ping-Pong Failover: Implementation Plan

> 以 `ProcessGroupNCCLFT.cpp` 目前實作狀態為準（2026-08）

---

## 一、系統架構速覽

`ProcessGroupNCCLFT` 是基於 `ProcessGroupNCCL` 複製並擴充的自訂 NCCL 容錯 backend（c10d 名稱 `nccl_ft`）。詳細架構請參見 `system_arch.md`。

### 三執行緒模型

| 執行緒 | 名稱 | 職責 |
|--------|------|------|
| 主執行緒 | — | 執行 collective；故障後在 2PC barrier spin-wait；REBUILD path 執行拓撲重建與 replay |
| Watchdog | `pt_nccl_watchdg` | 掃描 workMetaList_，偵測失敗 work，FT 模式下清除 exception + ncclEndEvent_ 而非 rethrow |
| Side-car negotiator | `pt_nccl_ft_side` | 透過 TCPStore 2PC 協商故障邊界；fault_mask 僅由 NCCL callback 寫入（不是 Watchdog） |

### 硬體假設

- 2 台 server × 8 GPU（rank 0–7 on server 0, rank 8–15 on server 1）
- 每張 GPU 對應一張 NUMA-local NIC（mlx5_0 ~ mlx5_7）
- 同 server 內所有 GPU 以 NVLink 互連
- `torchrun --nproc_per_node=8 --nnodes=2`

---

## 二、已完成的實作項目

| 項目 | 位置（cpp 行號） | 狀態 |
|------|-----------------|------|
| `initLocalNvlinkComm()` — 獨立建立 NVLink comm（不依賴 ncclCommSplit） | ~4060 | ✅ |
| `rebuild_shadow_ping_pong_topology()` — NCCLFTComm::create reinit + ncclCommBanNic | ~4133 | ✅ |
| `execute_shadow_allreduce()` — FAULTY/PROXY/HEALTHY 四步驟角色邏輯 | ~4297 | ✅ |
| `trigger_fault_proposal()` — NCCL callback，bitmask fetch_or | ~4488 | ✅ |
| `start_ft_negotiator_thread()` — 2PC side-car，per-rank PROPOSE key，min agreed_ss | ~4538 | ✅ |
| `get_or_allocate_shadow_buffer()` — ring pool pinned memory | ~4519 | ✅ |
| `allreduce_impl()` shadow_pre lambda — per-seq in-flight 快照（D2H） | ~6223 | ✅ |
| `collective()` 2PC barrier（DRAIN / REBUILD / replay） | ~5076 | ✅ |
| Fix A：移除 Watchdog fallback fault_mask 寫入 | Watchdog runLoop | ✅ |
| Fix 1：每個 rank 獨立呼叫 `ncclCommRegisterFaultCallback` | collective() lazy-init | ✅ |
| Fix 3：Watchdog early-abort（final_commit_op_ > 0 時強制清除飛行中 work） | Watchdog runLoop ~2460 | ✅ |
| Fix D：collective() 原生路徑捕獲 NCCLFaultToleranceError，不傳到 Python | collective() ~5442 | ✅ |
| Fix D2：`initLocalNvlinkComm()` try-catch + `nvlink_init_attempted_` flag | collective() lazy-init ~4975 | ✅ |
| Bug 1 fix：FT 模式下 Watchdog 不設 COMM_ERROR | Watchdog runLoop ~2436 | ✅ |
| Bug 10 fix：降級模式下 work->ncclComm_ 指向實際使用的 comm | collective() ~5515 | ✅ |
| Bug 12 fix：多 NIC 同時故障用 bitmask fetch_or | trigger_fault_proposal | ✅ |
| UINT64_MAX sentinel（agreed_ss == UINT64_MAX 表示無 checkpoint） | side-car & barrier | ✅ |
| ncclEndEvent_ 在 early-abort / Watchdog 清除時立即 record | Watchdog runLoop | ✅ |
| 2PC side-car 強制 abort globalComm 以喚醒卡死的主執行緒 | start_ft_negotiator_thread ~4841 | ✅ |
| `allreduce()` 降級模式跳過 intraNodeComm fast-path | allreduce() ~6301 | ✅ |
| Watchdog GC：AllReduce 成功後釋放 in_flight_shadow_bufs_ 到 free pool | Watchdog runLoop ~2822 | ✅ |

---

## 三、待確認與待完成項目

### Fix B（P0）：確認 NCCL fork 行為

**問題 1：** `ncclCommBanNic(dev_idx)` 是 process-level global 還是 per-comm？
- 如果是 process-level：rebuild 前呼叫一次即可，後續 `NCCLFTComm::create` 建立的新 comm 自動跳過被 ban 的 NIC。
- 如果是 per-comm：ban 在 abort 後失效，需要在 reinit 前再次呼叫（目前 `rebuild_shadow_ping_pong_topology` 已在 reinit 前重複呼叫，應可覆蓋此情況）。

**問題 2：** `ncclCommInitRank`（即 `NCCLFTComm::create`）是否每次都重新執行 topo discovery？
- 需要實驗確認：呼叫 `ncclCommBanNic(0)` 後再建立新 comm，確認新 comm 不使用 mlx5_0。

### 端對端測試（P0）

```
[ ] 測試 1：NIC 在 DDP init 期間故障
      預期：訓練繼續，無 DistBackendError 傳到 Python
[ ] 測試 2：NIC 在訓練中途故障（已完成 N 個 AllReduce 後）
      預期：2PC 完整流程 → rebuild → replay → 訓練以降級模式繼續
[ ] 測試 3：兩個 NIC 同時故障
      預期：bitmask 正確捕捉兩個 dev，agg_fault_mask 有兩個 bit
[ ] 測試 4：NCCL_FT_DISABLE=1
      預期：完全走原生 NCCL 路徑，無任何 FT 邏輯
```

### 後續 P1 工作

| 項目 | 說明 |
|------|------|
| AllGather / ReduceScatter 降級路徑 | FSDP / ZeRO2/3 場景 |
| Per-Server NIC Pool 非對稱降級 | 不同節點不同 NIC 索引（Bug 13） |
| Proxy 負載均衡 | 目前固定 `(faulty+1) % N`，多 faulty 時可能集中在同一個 proxy |
| Error type 精細分類 | 目前所有 ncclRemoteError 視為 NIC 硬體故障，應區分網路抖動 vs. 實際 NIC failure |
| AllReduce AVG 正確性驗證 | proxy 用 ncclSum 再除以原始 size_，需驗證浮點精度 |

---

## 四、已解決的關鍵 Bug（根因分析）

### Bug A（已解決）：所有 rank 都被標記為故障

**根因：** Watchdog 清除 `ncclRemoteError` poisoned work 時，`device_idx = work.device_.index() % localDeviceCount_`。這是每個 rank 自己的 GPU index，不是故障 NIC 的 index。AllReduce 是集體操作，`ncclRemoteError` 會傳播到所有 rank 的 comm，導致每個 rank 都把自己的 device index 寫入 fault_mask，`agg_fault_mask` 有全部 8 個 bit，`proxy_comm_size_` 計算為 0，系統崩潰。

**修復：** `local_hardware_fault_mask_` 只由 `trigger_fault_proposal()`（NCCL callback，接收的是真正故障的 `dev_idx`）寫入。Watchdog 完全不寫 fault_mask。

### Bug B（已解決）：ncclCommSplit 在 RemoteError comm 上失敗

**根因：** `NCCLFTComm::split()` 的第一步呼叫 `source->getNcclComm()`，當 `aborted_` 為 true 時直接 throw `NCCLFaultToleranceError`。即使 `aborted_` 不為 true，RemoteError 狀態的 comm 傳給 `ncclCommSplit` 結果也是未定義的。

**修復：** 在 `rebuild_shadow_ping_pong_topology()` 中改用 `NCCLFTComm::create`（從頭初始化），配合 `ncclCommBanNic` 排除故障 NIC，透過 TCPStore rendezvous 新的 `ncclUniqueId`。

### Bug D（已解決）：同步 NCCL 錯誤傳到 Python

**根因：** NIC 在第一個 collective（DDP init 的 `_verify_param_shape_across_processes`）期間故障時，`C10D_NCCL_FT_CHECK_TIMEOUT` 在 `fn()` 呼叫中同步拋出 `NCCLFaultToleranceError`，沒有任何 catch，直接傳到 Python 為 `DistBackendError`，訓練崩潰。

**修復：** `collective()` 原生路徑的 `fn()` 呼叫外加 try-catch，捕獲 `NCCLFaultToleranceError`，記錄 `ncclEndEvent_`，設定 work 的必要欄位，return 完好的 work。2PC 恢復由 NCCL fault callback 機制驅動，主執行緒不感知。

---

## 五、NCCL Custom Fork 端的確認狀態

| 項目 | 狀態 |
|------|------|
| `ncclIbResiliencyHandleDeviceFailure`: `ncclSystemError → ncclRemoteError` | ✅ 已完成 |
| `ncclIbResiliencyProbeHandleCompletionEvent`: `ncclSuccess → ncclRemoteError` | ✅ 已完成 |
| `topo.cc ncclTopoPopulateNics`: 移除 ban check（讓 `ncclCommBanNic` 在 reinit 時生效） | ✅ 已完成 |
| `init.cc`: 新增 `ncclCommBanNicReset()` | ✅ 已完成 |
| `init.cc nccl_ft_trigger_fault`: 呼叫已註冊的 callback | ✅ 已完成 |
| `ncclCommRegisterFaultCallback` API 可用 | ✅ 已確認（在 first collective 成功呼叫） |
| **`ncclCommBanNic` 作用域（process-level vs. per-comm）** | ❓ 待確認 |
| **`NCCLFTComm::create` 在 reinit 時是否每次重新做 topo discovery** | ❓ 待確認 |
