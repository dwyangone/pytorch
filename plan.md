# ProcessGroupNCCLFT — Shadow Ping-Pong Failover: Implementation Plan

> 以 `ProcessGroupNCCLFT.cpp` 目前實作狀態為準（2026-08）

---

## 一、系統架構速覽

`ProcessGroupNCCLFT` 是基於 `ProcessGroupNCCL` 複製並擴充的自訂 NCCL 容錯 backend（c10d 名稱 `nccl_ft`）。詳細架構請參見 `system_arch.md`。

### 三執行緒模型

| 執行緒 | 名稱 | 職責 |
|--------|------|------|
| 主執行緒 | — | 執行 collective；故障後由 `wait()` 攔截點 1 觸發 `recover_and_replay_inflight_ops()` |
| Watchdog | `pt_nccl_watchdg` | 掃描 workMetaList_，偵測失敗 work，FT 模式下清除 exception（不寫 fault mask，不 rethrow）；正常完成 GC shadow buffers |
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
| `initLocalNvlinkComm()` — 獨立建立 NVLink comm（不依賴 ncclCommSplit） | ~4040 | ✅ |
| `rebuild_shadow_ping_pong_topology()` — NCCLFTComm::create reinit + ncclCommBanNic | ~4106 | ✅ |
| `execute_shadow_allreduce()` — FAULTY/PROXY/HEALTHY 四角色邏輯（含 findProxy + my_wards） | ~4270 | ✅ |
| `trigger_fault_proposal()` — NCCL callback，bitmask fetch_or | ~4461 | ✅ |
| `start_ft_negotiator_thread()` — 2PC side-car，per-rank PROPOSE key，min agreed_ss | ~4511 | ✅ |
| `get_or_allocate_shadow_context()` — ShadowContext ring pool，兩段式鎖設計 | ~4865 | ✅ |
| `recover_and_replay_inflight_ops()` — 全域重播中心，abort→rebuild→H2D→replay | ~5201 | ✅ |
| `WorkNCCLFT::wait()` 攔截點 1 — `final_commit_op_ > 0` 觸發 recover_and_replay | ~907 | ✅ |
| `WorkNCCLFT::wait()` 攔截點 2 — `is_degraded_` 時等待 `replayed_end_event` + GC | ~914 | ✅ |
| `allreduce_impl()` shadow_pre lambda — per-seq ShadowContext（含 reduce_op / per-seq event） | ~5929 | ✅ |
| `collective()` 二分支設計（行 5085-5152）— `is_degraded_` 直接執行 shadow allreduce | ~5085 | ✅ |
| Fix A：移除 Watchdog fallback fault_mask 寫入 | Watchdog runLoop ~2562 | ✅ |
| Fix 1：每個 rank 獨立呼叫 `ncclCommRegisterFaultCallback` | collective() lazy-init ~4956 | ✅ |
| Fix 3：Watchdog early-abort（final_commit_op_ > 0 時強制記錄 ncclEndEvent_） | Watchdog runLoop ~2503 | ✅ |
| Fix D：collective() 原生路徑捕獲 NCCLFaultToleranceError，補齊 future_，不傳到 Python | collective() ~5124 | ✅ |
| Fix D2：`initLocalNvlinkComm()` try-catch + `nvlink_init_attempted_` flag | collective() lazy-init ~4956 | ✅ |
| Bug 1 fix：FT 模式下 Watchdog 不設 COMM_ERROR | Watchdog runLoop ~2489 | ✅ |
| Bug 2 fix：catch NCCLFaultToleranceError 的 early return 補齊 work->future_ | collective() ~5133 | ✅ |
| Bug 3 確認：recover_and_replay seq >= agreed_ss 是正確的（agreed_ss 本身狀態未知，必須重播） | ~5232 | ✅ |
| Bug 4 fix：get_or_allocate_shadow_context 兩段式鎖設計（pin_memory 在鎖外執行） | ~4865 | ✅ |
| Bug 5 確認：降級後 shadow_pre 仍執行 D2H copy — **設計決定**，為二次故障 replay 預備 | shadow_pre | ✅ |
| Bug 6 fix：`allreduce()` 降級模式跳過 intraNodeComm fast-path | allreduce() ~6008 | ✅ |
| Bug 7 fix：proxy 的 ATen 算術（copy_/add_）在 ncclStream 上執行（setCurrentCUDAStream） | execute_shadow_allreduce ~4401 | ✅ |
| Bug 10 fix：降級模式下 work->ncclComm_ 指向實際使用的 comm | collective() ~5108 | ✅ |
| Bug 12 fix：多 NIC 同時故障用 bitmask fetch_or | trigger_fault_proposal ~4472 | ✅ |
| Bug A fix：`ft_round_++` 在 recover_and_replay 末尾執行 + TCPStore cleanup | ~5273 | ✅ |
| Bug B fix：用 `final_commit_op_ == 0` 取代 `rollback_done_` 判斷重入 | ~5207 | ✅ |
| Bug C fix：side-car Phase 1b OR 累積更新 `faulty_local_devs_`（逐步故障支援） | ~4752 | ✅ |
| UINT64_MAX sentinel（agreed_ss == UINT64_MAX 表示無 checkpoint） | side-car & barrier | ✅ |
| 2PC side-car 強制 abort globalComm 以喚醒卡死的主執行緒 | start_ft_negotiator_thread ~4814 | ✅ |
| Watchdog GC：AllReduce 成功後釋放 in_flight_shadow_bufs_ 到 free pool | Watchdog runLoop ~2792 | ✅ |
| wait() 攔截點 2 GC：replay 完成後由主執行緒回收 shadow buffer | wait() ~937 | ✅ |
| `ShadowContext` struct（含 per-seq copy_event / replayed_end_event / reduce_op） | hpp ~1181 | ✅ |
| findProxy()：round-robin 尋找下一個健康 proxy（支援多 faulty） | ~4259 | ✅ |
| my_wards vector：proxy 可代理多個 faulty rank | execute_shadow_allreduce ~4297 | ✅ |

---

## 三、待確認與待修復項目

### P0：確認 NCCL Fork 行為

**問題 1：** `ncclCommBanNic(dev_idx)` 是 process-level global 還是 per-comm？
- 如果是 process-level：rebuild 前呼叫一次即可，後續 `NCCLFTComm::create` 建立的新 comm 自動跳過被 ban 的 NIC。
- 如果是 per-comm：ban 在 abort 後失效，需要在 reinit 前再次呼叫（目前 `rebuild_shadow_ping_pong_topology` 已在所有 rank 呼叫 ban，應可覆蓋此情況）。

**問題 2：** `NCCLFTComm::create`（即 `ncclCommInitRank`）是否每次都重新執行 topo discovery？
- 需要實驗確認：呼叫 `ncclCommBanNic(0)` 後再建立新 comm，確認新 comm 不使用 mlx5_0。

### P0：端對端測試

```
[ ] 測試 1：NIC 在 DDP init 期間故障
      預期：訓練繼續，無 DistBackendError 傳到 Python
[ ] 測試 2：NIC 在訓練中途故障（已完成 N 個 AllReduce 後）
      預期：2PC 完整流程 → recover_and_replay_inflight_ops → is_degraded_ mode 繼續
[ ] 測試 3：兩個 NIC 同時故障
      預期：bitmask 正確捕捉兩個 dev，agg_fault_mask 有兩個 bit，findProxy 正確分配
[ ] 測試 4：NCCL_FT_DISABLE=1
      預期：完全走原生 NCCL 路徑，無任何 FT 邏輯
[ ] 測試 5：故障後梯度數學正確性驗證
      預期：故障 step 的梯度與無故障版本 allclose
[ ] 測試 6：逐步故障（第一次故障後，第二張 NIC 再次故障）
      預期：ft_round_ 遞增，第二輪 2PC 完整流程，faulty_local_devs_ 累積兩個故障
```

### P1：後續工作

| 項目 | 說明 |
|------|------|
| AllGather / ReduceScatter 降級路徑 | FSDP / ZeRO2/3 場景；目前降級時非 AllReduce op 走 fallback fn()，可能崩潰 |
| Per-Server NIC Pool 非對稱降級 | 不同節點不同 NIC 索引時（對稱降級假設目前固定） |
| Proxy 負載均衡 | 目前固定 `(faulty+1) % N`，多 faulty 時 findProxy 可能將多個 ward 集中在同一 proxy |
| Error type 精細分類 | 目前所有 ncclRemoteError 視為 NIC 硬體故障，應區分網路抖動 vs. 實際 NIC failure |
| AllReduce AVG 正確性驗證 | proxy 用 ncclSum 再除以原始 size_，需驗證浮點精度是否與 ncclAvg 一致 |
| 降級後 `is_degraded_` 不恢復 | 目前一旦降級不回到正常模式；若 NIC 可熱插拔可考慮恢復路徑 |

---

## 四、已解決的關鍵 Bug（根因分析）

### Bug A（已解決）：所有 rank 都被標記為故障

**根因：** Watchdog 清除 `ncclRemoteError` poisoned work 時，`device_idx = work.device_.index() % localDeviceCount_`。這是每個 rank 自己的 GPU index，不是故障 NIC 的 index。AllReduce 是集體操作，`ncclRemoteError` 會傳播到所有 rank 的 comm，導致每個 rank 都把自己的 device index 寫入 fault_mask，`agg_fault_mask` 有全部 8 個 bit，`proxy_comm_size_` 計算為 0，系統崩潰。

**修復：** `local_hardware_fault_mask_` 只由 `trigger_fault_proposal()`（NCCL callback，接收的是真正故障的 `dev_idx`）寫入。Watchdog 完全不寫 fault_mask（Fix A）。

### Bug B（已解決）：ncclCommSplit 在 RemoteError comm 上失敗

**根因：** `NCCLFTComm::split()` 的第一步呼叫 `source->getNcclComm()`，當 `aborted_` 為 true 時直接 throw `NCCLFaultToleranceError`。即使 `aborted_` 不為 true，RemoteError 狀態的 comm 傳給 `ncclCommSplit` 結果也是未定義的。

**修復：** 在 `rebuild_shadow_ping_pong_topology()` 中改用 `NCCLFTComm::create`（從頭初始化），配合 `ncclCommBanNic` 排除故障 NIC，透過 TCPStore rendezvous 新的 `ncclUniqueId`。

### Bug D（已解決）：同步 NCCL 錯誤傳到 Python

**根因：** NIC 在第一個 collective（DDP init 的 `_verify_param_shape_across_processes`）期間故障時，`C10D_NCCL_FT_CHECK_TIMEOUT` 在 `fn()` 呼叫中同步拋出 `NCCLFaultToleranceError`，沒有任何 catch，直接傳到 Python 為 `DistBackendError`，訓練崩潰。

**修復：** `collective()` 原生路徑的 `fn()` 呼叫外加 try-catch，捕獲 `NCCLFaultToleranceError`，記錄 `ncclEndEvent_`，補齊 `work->future_`，workEnqueue 後 return work。2PC 恢復由 NCCL fault callback 機制驅動，主執行緒不感知（Bug 2 fix 同時修復了 future_ 未設置問題）。

### Bug 6（已解決）：降級模式下 intraNodeComm fast-path 繞過 FT 邏輯

**根因：** `allreduce()` 中，`intraNodeComm_->allReduce()` 返回的 `IntraNodeCommWork` 完全繞過 `allreduce_impl()` 的 shadow buffer checkpoint 和 `collective()` 的容錯邏輯，在降級後使用舊 topology 產生錯誤結果或崩潰。

**修復：** `allreduce()` 中加入 `!is_degraded_` 條件，降級模式下強制走 `allreduce_impl`（行 6008）。

### Bug 7（已解決）：proxy 的 ATen 算術未在 ncclStream 上執行

**根因：** proxy step 2 的 `output.copy_()` 和 `output.add_()` 預設在 compute stream 執行，但 `ward_bufs` 的數據是由 ncclRecv 寫到 ncclStream 上的。兩個 stream 並行執行時，ATen 可能讀到 ncclRecv 尚未完成的數據。

**修復：** 使用 `setCurrentCUDAStream(stream)` + restore，確保算術核心在 ncclStream 上排入（行 4401-4410）。

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

---

## 六、遺留效能注意事項

| 項目 | 影響 | 說明 |
|------|------|------|
| early-abort 的 ncclEndEvent_ 被 record 兩次 | 極低 | 行 2517（early-abort）和 2633（FT clearing path）；多一次 CUDA API 呼叫，無害 |
| 降級後仍執行 D2H copy | 輕微額外記憶體頻寬 | **設計決定**，為二次故障 replay 預備 shadow buffer |
| proxy step 2 建立 ward_bufs（at::empty_like） | GPU allocator 呼叫 | 每個降級 AllReduce 都會配置，影響較輕；可考慮預配置 |
