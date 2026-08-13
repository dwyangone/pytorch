# ProcessGroupNCCLFT — Shadow Ping-Pong Failover: Implementation Plan

> 以 `ProcessGroupNCCLFT.cpp` 目前實作狀態為準（2026-08）

---

## 一、系統架構速覽

`ProcessGroupNCCLFT` 是基於 `ProcessGroupNCCL` 複製並擴充的自訂 NCCL 容錯 backend（c10d 名稱 `nccl_ft`）。詳細架構請參見 `system_arch.md`。

### 三執行緒模型

| 執行緒 | 名稱 | 職責 |
|--------|------|------|
| 主執行緒 | — | 執行 collective；故障後由 `wait()` 的攔截點 1 觸發 `recover_and_replay_inflight_ops()` |
| Watchdog | `pt_nccl_watchdg` | 掃描 workMetaList_，偵測失敗 work，FT 模式下清除 exception（不寫 fault mask，不 rethrow）；GC shadow buffers |
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
| `get_or_allocate_shadow_context()` — ShadowContext ring pool（取代舊 at::Tensor） | ~4933 | ✅ |
| `recover_and_replay_inflight_ops()` — 全域重播中心，abort→rebuild→H2D→replay | ~5440 | ✅ |
| `WorkNCCLFT::wait()` 攔截點 1 — `final_commit_op_ > 0` 觸發 recover_and_replay | ~907 | ✅ |
| `WorkNCCLFT::wait()` 攔截點 2 — `is_degraded_` 時等待 `replayed_end_event` | ~914 | ✅ |
| `allreduce_impl()` shadow_pre lambda — per-seq ShadowContext（含 reduce_op / per-seq event） | ~6146 | ✅ |
| `collective()` 前置分流（行 5138-5169）— `is_degraded_` 直接執行 shadow allreduce | ~5138 | ✅ |
| Fix A：移除 Watchdog fallback fault_mask 寫入 | Watchdog runLoop | ✅ |
| Fix 1：每個 rank 獨立呼叫 `ncclCommRegisterFaultCallback` | collective() lazy-init ~5014 | ✅ |
| Fix 3：Watchdog early-abort（final_commit_op_ > 0 時強制記錄 ncclEndEvent_） | Watchdog runLoop ~2486 | ✅ |
| Fix D：collective() 原生路徑捕獲 NCCLFaultToleranceError，不傳到 Python | collective() ~5307 | ✅ |
| Fix D2：`initLocalNvlinkComm()` try-catch + `nvlink_init_attempted_` flag | collective() lazy-init ~5014 | ✅ |
| Bug 1 fix：FT 模式下 Watchdog 不設 COMM_ERROR | Watchdog runLoop ~2470 | ✅ |
| Bug 6 fix：`allreduce()` 降級模式跳過 intraNodeComm fast-path | allreduce() ~6225 | ✅ |
| Bug 10 fix：降級模式下 work->ncclComm_ 指向實際使用的 comm | collective() ~5369 | ✅ |
| Bug 12 fix：多 NIC 同時故障用 bitmask fetch_or | trigger_fault_proposal | ✅ |
| UINT64_MAX sentinel（agreed_ss == UINT64_MAX 表示無 checkpoint） | side-car & barrier | ✅ |
| 2PC side-car 強制 abort globalComm 以喚醒卡死的主執行緒 | start_ft_negotiator_thread ~4841 | ✅ |
| Watchdog GC：AllReduce 成功後釋放 in_flight_shadow_bufs_ 到 free pool（清 Tensor 參照） | Watchdog runLoop ~2860 | ✅ |
| `ShadowContext` struct（含 per-seq copy_event / replayed_end_event / reduce_op） | hpp ~1181 | ✅ |

---

## 三、待確認與待修復項目

### P0：必須修復的 Bugs

#### Bug 1（嚴重）：collective() 降級路徑的 execute_shadow_allreduce 被呼叫兩次

**位置：** 行 5142-5148（新前置分流）和行 5250-5278（舊切換點 B）

**現象：** `is_degraded_ = true` 時，新前置分流執行 `execute_shadow_allreduce` 後 fall-through，舊切換點 B 的條件 `is_degraded_ && !ran_shadow_replay` 也成立（`ran_shadow_replay` 未在前置分流中設定），導致同一個 AllReduce **執行兩次**。

**影響：** 梯度數值錯誤（double reduce），或 NCCL stream 序列混亂。

**修復方向：** 選其一：
1. 在前置分流（行 5148）後設定 `ran_shadow_replay = true`，讓切換點 B 不觸發
2. 完整刪除舊的切換點 A/B 邏輯（行 5203-5343），整合到前置分流

#### Bug 2（中）：前置分流的 early return 未設置 work->future_

**位置：** 行 5157-5165（catch NCCLFaultToleranceError 的 early return）

**現象：** `return work` 前未設置 `work->future_`。DDP 呼叫 `getFuture()` 時會因 nullptr future 崩潰。

**修復：** 在 return 前加入與行 5307-5340（舊邏輯）相同的 future_ 設置邏輯：
```cpp
c10::cuda::CUDAMultiStreamGuard sg(ncclStream);
std::vector<at::Device> devs{device};
work->future_ = c10::make_intrusive<at::ivalue::Future>(
    c10::ListType::create(c10::TensorType::get()), devs);
work->future_->markCompleted(at::IValue(*work->outputs_));
```

#### Bug 3（中）：recover_and_replay_inflight_ops 的 seq 邊界條件

**位置：** 行 5465：`if (seq >= agreed_ss)`

**現象：** 若 `seq == agreed_ss` 的 AllReduce **尚未被 Watchdog GC**，它已成功完成但仍在 in_flight_shadow_bufs_ 中，會被 replay（重複 reduce）。

**修復：** 改為 `if (seq > agreed_ss)`，只 replay 在故障時還未完成的 ops。

### P1：效能問題

#### Bug 4（低）：get_or_allocate_shadow_context 在鎖內呼叫 pin_memory()

**位置：** 行 4934, 4948

**現象：** `pin_memory()` = `cudaHostAlloc`（同步 CUDA call）在 `shadow_buf_mutex_` 鎖住狀態下執行，可能 stall Watchdog GC。

**緩解：** ring pool 命中時不執行，低機率，可接受。正式修復：在鎖外配置後再鎖定插入。

### P0：確認 NCCL Fork 行為

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
      預期：2PC 完整流程 → recover_and_replay_inflight_ops → is_degraded_ mode 繼續
[ ] 測試 3：兩個 NIC 同時故障
      預期：bitmask 正確捕捉兩個 dev，agg_fault_mask 有兩個 bit
[ ] 測試 4：NCCL_FT_DISABLE=1
      預期：完全走原生 NCCL 路徑，無任何 FT 邏輯
[ ] 測試 5：故障後梯度數學正確性驗證（修復 Bug 1 後）
      預期：故障 step 的梯度與無故障版本 allclose
```

### 後續 P1 工作

| 項目 | 說明 |
|------|------|
| 清理 collective() 舊邏輯 | 刪除或重構行 5203-5343 的舊切換點 A/B（含舊 2PC barrier），與前置分流整合，消除 Bug 1 |
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

### Bug 6（已解決）：降級模式下 intraNodeComm fast-path 繞過 FT 邏輯

**根因：** `allreduce()` 中，`intraNodeComm_->allReduce()` 返回的 `IntraNodeCommWork` 完全繞過 `allreduce_impl()` 的 shadow buffer checkpoint 和 `collective()` 的容錯邏輯，在降級後使用舊 topology 產生錯誤結果或崩潰。

**修復：** `allreduce()` 中加入 `!is_degraded_` 條件，降級模式下強制走 `allreduce_impl`。

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
