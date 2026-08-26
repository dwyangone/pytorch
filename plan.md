# ProcessGroupNCCLFT — Shadow Ping-Pong Failover: Implementation Plan

> 以 `ProcessGroupNCCLFT.cpp` / `.hpp` 目前實作狀態為準（2026-08，第四次修訂）

---

## 一、系統架構速覽

`ProcessGroupNCCLFT` 是基於 `ProcessGroupNCCL` 複製並擴充的自訂 NCCL 容錯 backend（c10d 名稱 `nccl_ft`）。詳細架構請參見 `system_arch.md`。

### 三執行緒模型

| 執行緒 | 名稱 | 職責 |
|--------|------|------|
| 主執行緒 | — | 執行 collective；故障後 `wait()` 進入 FT 安全等待區，呼叫 `future_->wait()` 阻塞等側車的 `markCompleted`，不輪詢 final_commit_op_ |
| Watchdog | `pt_nccl_watchdg` | 掃描 workMetaList_，偵測失敗 work，FT 模式下清除 exception（不寫 fault mask，不 rethrow）；正常完成 GC shadow buffers |
| Side-car negotiator | `pt_nccl_ft_side` | 透過 TCPStore 2PC 協商故障邊界；2PC 完成後**直接呼叫 recover_and_replay_inflight_ops()**，不依賴主執行緒 |

### 硬體假設

- 2 台 server × 8 GPU（rank 0–7 on server 0, rank 8–15 on server 1）
- 每張 GPU 對應一張 NUMA-local NIC（mlx5_0 ~ mlx5_7）
- 同 server 內所有 GPU 以 NVLink 互連
- `torchrun --nproc_per_node=8 --nnodes=2`

---

## 二、已完成的實作項目

| 項目 | 位置（cpp 行號） | 狀態 |
|------|-----------------|------|
| `initLocalNvlinkComm(attempt)` — 使用 NCCLFTComm::create 從頭建立 NVLink comm（不用 ncclCommSplit）| ~4043 | ✅ |
| `rebuild_shadow_ping_pong_topology()` — NCCLFTComm::create reinit + ncclCommBanNic（單一 for 迴圈，不重複） | ~4176 | ✅ |
| `execute_shadow_allreduce()` — FAULTY/PROXY/HEALTHY 四角色邏輯（含 findProxy + my_wards） | ~4425 | ✅ |
| `trigger_fault_proposal()` — NCCL callback，bitmask fetch_or，shadow_seq_ atomic load | ~4638 | ✅ |
| `start_ft_negotiator_thread()` — 2PC side-car，per-rank PROPOSE key，any_proposed 快速路徑，min agreed_ss | ~4674 | ✅ |
| Side-car 2PC 完成後直接呼叫 `recover_and_replay_inflight_ops()`（不等主執行緒） | ~4994 | ✅ |
| `get_or_allocate_shadow_context()` — 三段式設計（Inline GC + Best-Fit + Memory Backpressure + 鎖外 alloc） | ~5012 | ✅ |
| `ShadowContext::compute_event` — 提升為 ShadowContext 成員，不再每次 allreduce 建立新 event | hpp ~1182 | ✅ |
| `ShadowContext::work_ptr` — 綁定 Work，供 Inline GC 主動回收成功 bucket | hpp ~1183 | ✅ |
| `recover_and_replay_inflight_ops()` — 全域重播中心，abort→rebuild→H2D→replay→setException/markCompleted | ~5419 | ✅ |
| recover 內部直接 `work_ptr->setException(nullptr)` + `future_->markCompleted` 解鎖 DDP | ~5499-5506 | ✅ |
| `WorkNCCLFT::wait()` Future 機制 FT 安全等待區（is_ft_exception / is_ft_managed + future_->wait() + Inline GC） | ~915 | ✅ |
| `allreduce_impl()` shadow_pre lambda — work_ptr + compute_event + atomic shadow_seq_ | ~6185 | ✅ |
| `collective()` future_ 提前初始化（pre() 呼叫前），消除 Race Condition | ~5287 | ✅ |
| `collective()` 二分支設計 — `is_degraded_` 直接執行 shadow allreduce | ~5300 | ✅ |
| Fix A：移除 Watchdog fallback fault_mask 寫入 | Watchdog runLoop | ✅ |
| Fix 1：每個 rank 獨立呼叫 `ncclCommRegisterFaultCallback` | collective() lazy-init ~5159 | ✅ |
| Fix 3：Watchdog early-abort（final_commit_op_ > 0 時強制 setException） | Watchdog runLoop | ✅ |
| Fix D：collective() 原生路徑捕獲 NCCLFaultToleranceError，設 exception，提早 return | collective() ~5345 | ✅ |
| Fix D2：`initLocalNvlinkComm()` try-catch + `nvlink_init_attempted_` flag | collective() lazy-init ~5183 | ✅ |
| Bug 1 fix：FT 模式下 Watchdog 不設 COMM_ERROR | Watchdog runLoop ~2536 | ✅ |
| Bug 2 fix：future_ 在 pre() 前初始化（消除 catch 區塊需要重建 future_ 的問題） | collective() ~5287 | ✅ |
| Bug 3 確認：recover_and_replay seq >= agreed_ss 是正確的 | ~5447 | ✅ |
| Bug 4 fix：get_or_allocate_shadow_context 三段式設計（鎖外 alloc） | ~5012 | ✅ |
| Bug 5 確認：降級後 shadow_pre 仍執行 D2H copy — **設計決定** | shadow_pre | ✅ |
| Bug 6 fix：`allreduce()` 降級模式跳過 intraNodeComm fast-path | allreduce() ~6281 | ✅ |
| Bug 7 fix：proxy 的 ATen 算術（copy_/add_）在 ncclStream 上執行（setCurrentCUDAStream） | execute_shadow_allreduce ~4563 | ✅ |
| Bug 10 fix：降級模式下 work->ncclComm_ 指向實際使用的 comm | collective() ~5317 | ✅ |
| Bug 12 fix：多 NIC 同時故障用 bitmask fetch_or | trigger_fault_proposal ~4651 | ✅ |
| Bug A fix：`ft_round_++` 在 recover_and_replay 末尾執行 + TCPStore cleanup | ~5509 | ✅ |
| Bug B fix：用 `final_commit_op_ == 0` 取代 `rollback_done_` 判斷重入 | ~5422 | ✅ |
| Bug C fix：side-car Phase 1b OR 累積更新 `faulty_local_devs_`（逐步故障支援） | ~4914 | ✅ |
| shadow_seq_ 資料競爭修復：改為 `std::atomic<uint64_t>` | hpp ~1201 | ✅ |
| compute_done event per-call fix：提升為 ShadowContext::compute_event | shadow_pre / get_or_allocate | ✅ |
| ncclCommSplit 在 RemoteError comm 上失敗 fix：改用 NCCLFTComm::create（TCPStore rendezvous） | initLocalNvlinkComm ~4043 | ✅ |
| 側車依賴主執行緒 fix：側車在 2PC 完成後直接呼叫 recover_and_replay | ~4994 | ✅ |
| UINT64_MAX sentinel（agreed_ss == UINT64_MAX 表示無 checkpoint） | side-car & barrier | ✅ |
| 2PC side-car 強制 abort globalComm 以喚醒卡死的主執行緒 | start_ft_negotiator_thread ~4981 | ✅ |
| Watchdog GC：AllReduce 成功後釋放 in_flight_shadow_bufs_ 到 free pool | Watchdog runLoop | ✅ |
| Inline GC：get_or_allocate 內主動回收已成功的 bucket | ~5023 | ✅ |
| Memory Backpressure：in-flight > 2GB 時 sleep 等待 | ~5065 | ✅ |
| Best-Fit buffer 搜尋：取代舊版第一個 numel >= 的策略 | ~5043 | ✅ |
| wait() FT 後處理 GC：replay 完成後由 wait() 回收 shadow buffer | wait() | ✅ |
| `ShadowContext` struct（含 per-seq copy_event / replayed_end_event / compute_event / work_ptr / reduce_op） | hpp ~1178 | ✅ |
| findProxy()：round-robin 尋找下一個健康 proxy（支援多 faulty） | ~4414 | ✅ |
| my_wards vector：proxy 可代理多個 faulty rank | execute_shadow_allreduce ~4452 | ✅ |
| **多維 Tensor Shape Mismatch fix**：shadow_pre D2H copy + H2D restore 均加 `.flatten()` | shadow_pre, recover_and_replay ~5490 | ✅ |
| **迴圈內 CUDAEvent 效能 fix**：recover_and_replay 迴圈外宣告 `restore_done`，迴圈內重複 record | recover_and_replay ~5476 | ✅ |
| **Watchdog 例外清除後 GC 保底**：clearing path 將非空 stash push 到 shelvesToUnstash_ | Watchdog runLoop ~2679 | ✅ |
| any_proposed 快速路徑：side-car 無故障時 sleep 50ms 避免空轉 | ~4806 | ✅ |
| **logPrefix() static 返回懸空參照 fix**：移除 `static`，改為回傳值 | WorkNCCLFT::logPrefix | ✅ |
| **`initLocalNvlinkComm` 三屏障設計**（entry-arrival + ID 生成 + all-ready）| initLocalNvlinkComm ~4058-4132 | ✅ |
| **`collective()` FT catch block 補齊 `numelIn_`/`numelOut_`** | collective() catch | ✅ |
| **`initLocalNvlinkComm` TCPStore ID key 清除**：all-ready barrier 通過後立即 deleteKey | initLocalNvlinkComm ~4122 | ✅ |
| **`initLocalNvlinkComm(attempt)` — 呼叫者傳入 attempt 消除 lazy-init vs recovery key 衝突** | lazy-init 傳 0，recovery 傳 `ft_round_+1` | ✅ |
| **GC 執行緒加入 abort()** — 確保舊 comm 實際清理（`~NCCLFTComm` 不自動 abort） | rebuild ~4219-4240 | ✅ |
| **`pending_shadow_seq_` 在 recover_and_replay 末尾重置為 UINT64_MAX** — 消除逐步故障的 stale checkpoint | recover_and_replay ~5565 | ✅ |
| **移除所有 DEBUG-HANG / DEBUG-RUNAWAY log** — 清除暫時性調試輸出 | execute_shadow_allreduce, shadow_pre | ✅ |
| **[P0 fix] HEALTHY rank 在 execute_shadow_allreduce 加入 stream sync + ncclCommGetAsyncError**，確保所有 rank 都能感知失敗並觸發 rebuild | execute_shadow_allreduce ~4771 | ✅ |
| **[P1 方案二] ncclTopoGetLocalNetType 在 modulo 前過濾 banned NIC**，防止 channel 分配落在死亡 NIC | nccl/src/graph/topo.cc ~1809 | ✅ |
| **[Revert 方案三] 移除 ncclConfig_t::bannedNicsMask**，採用純方案二（無 ABI 變更） | nccl.h.in, topo.h, topo.cc, init.cc, ProcessGroupNCCLFT.cpp | ✅ |

---

## 三、待確認與待修復項目

### P0：端對端測試（進行中）

**問題 1：** `ncclCommBanNic(dev_idx)` 是 process-level global。
- 已確認：`g_nccl_ft_banned_nics_mask` 是 process-level global，`ncclTopoPopulateNics` 讀取它在每次 `ncclCommInitRankConfig` 的 topo XML 建構時生效。
- `rebuild_shadow_ping_pong_topology` 在所有 rank 呼叫 `ncclCommBanNic(d)` 後再建 comm，確保新 comm 的 topo discovery 跳過 banned NIC。

**問題 2：** `NCCLFTComm::create` 每次 reinit 都重新執行 topo discovery？
- 已確認：每次 `ncclCommInitRankConfig` 都呼叫 `ncclTopoGetSystem`，重新從 XML 建構 topo graph。Global ban mask 在此時生效（`ncclTopoPopulateNics` 設 speed=0）。
- 方案二額外保護：`ncclTopoGetLocalNetType` 的 modulo 選取也過濾 banned NIC，雙重防護。

**問題 3：** 側車在 recovery_mutex_ 上的執行緒競爭
- 現行設計：只有側車呼叫 `recover_and_replay_inflight_ops()`，主執行緒的 `wait()` 只呼叫 `future_->wait()`，不再觸發 recover
- recovery_mutex_ 只作防重入保護（避免兩次側車回合重疊），主執行緒不持鎖
- 結論：無死結風險；確認 `future_->wait()` 在側車 markCompleted 後能正確解除阻塞

### P1：低危清理

| 項目 | 位置 | 說明 |
|------|------|------|
| `get_or_allocate_shadow_buffer` 殭屍函式宣告 | hpp ~1229 | `at::Tensor get_or_allocate_shadow_buffer(const at::Tensor& t)` 仍保留在 hpp 宣告；應刪除 |
| `sscanf` 格式字串可移植性 | side-car ~4751 | `%lu`/`%lx` 在 Windows 是 32-bit；改用 `SCNu64`/`SCNx64` |

### P0：端對端測試

```
[ ] 測試 1：NIC 在 DDP init 期間故障
      預期：訓練繼續，無 DistBackendError 傳到 Python
[ ] 測試 2：NIC 在訓練中途故障（已完成 N 個 AllReduce 後）
      預期：2PC 完整流程 → recover_and_replay_inflight_ops（側車觸發）→ is_degraded_ mode 繼續
[ ] 測試 3：兩個 NIC 同時故障
      預期：bitmask 正確捕捉兩個 dev，agg_fault_mask 有兩個 bit，findProxy 正確分配
[ ] 測試 4：NCCL_FT_DISABLE=1
      預期：完全走原生 NCCL 路徑，無任何 FT 邏輯
[ ] 測試 5：故障後梯度數學正確性驗證
      預期：故障 step 的梯度與無故障版本 allclose
[ ] 測試 6：逐步故障（第一次故障後，第二張 NIC 再次故障）
      預期：ft_round_ 遞增，第二輪 2PC 完整流程，faulty_local_devs_ 累積兩個故障
[ ] 測試 7：主執行緒卡死時側車獨立完成重播
      預期：即使主執行緒未呼叫 recover_and_replay，側車也能完成並讓訓練繼續
```

### P2：後續工作

| 項目 | 說明 |
|------|------|
| AllGather / ReduceScatter 降級路徑 | FSDP / ZeRO2/3 場景；目前降級時非 AllReduce op 走 fallback fn()，可能崩潰 |
| Per-Server NIC Pool 非對稱降級 | 不同節點不同 NIC 索引時（對稱降級假設目前固定） |
| Proxy 負載均衡 | 目前固定 `(faulty+1) % N`，多 faulty 時 findProxy 可能將多個 ward 集中在同一 proxy |
| Error type 精細分類 | 目前所有 ncclRemoteError 視為 NIC 硬體故障，應區分網路抖動 vs. 實際 NIC failure |
| AllReduce AVG 正確性驗證 | proxy 用 ncclSum 再除以原始 size_，需驗證浮點精度是否與 ncclAvg 一致 |
| 降級後 `is_degraded_` 不恢復 | 目前一旦降級不回到正常模式；若 NIC 可熱插拔可考慮恢復路徑 |
| ward_bufs 預配置 | 目前每次 execute_shadow_allreduce 都 at::empty_like；可加入 ShadowContext pool |

---

## 四、已解決的關鍵 Bug（根因分析）

### Bug A（已解決）：所有 rank 都被標記為故障

**根因：** Watchdog 清除 `ncclRemoteError` poisoned work 時，`device_idx = work.device_.index() % localDeviceCount_`。這是每個 rank 自己的 GPU index，不是故障 NIC 的 index。AllReduce 是集體操作，`ncclRemoteError` 會傳播到所有 rank 的 comm，導致每個 rank 都把自己的 device index 寫入 fault_mask，`agg_fault_mask` 有全部 8 個 bit，`proxy_comm_size_` 計算為 0，系統崩潰。

**修復：** `local_hardware_fault_mask_` 只由 `trigger_fault_proposal()`（NCCL callback，接收的是真正故障的 `dev_idx`）寫入。Watchdog 完全不寫 fault_mask（Fix A）。

### Bug B（已解決）：ncclCommSplit 在 RemoteError comm 上失敗

**根因：** `NCCLFTComm::split()` 的第一步呼叫 `source->getNcclComm()`，當 `aborted_` 為 true 時直接 throw `NCCLFaultToleranceError`。即使 `aborted_` 不為 true，RemoteError 狀態的 comm 傳給 `ncclCommSplit` 結果也是未定義的。

**修復：** `initLocalNvlinkComm()` 改用 `NCCLFTComm::create`（從頭初始化），配合 TCPStore 在同節點 rank 間交換 `ncclUniqueId`，完全不依賴已故障的 parent comm。

### Bug D（已解決）：同步 NCCL 錯誤傳到 Python

**根因：** NIC 在第一個 collective（DDP init 的 `_verify_param_shape_across_processes`）期間故障時，`C10D_NCCL_FT_CHECK_TIMEOUT` 在 `fn()` 呼叫中同步拋出 `NCCLFaultToleranceError`，沒有任何 catch，直接傳到 Python 為 `DistBackendError`，訓練崩潰。

**修復：** `collective()` 原生路徑的 `fn()` 呼叫外加 try-catch，捕獲 `NCCLFaultToleranceError`，記錄 `ncclEndEvent_`，`work->setException(e)`，workEnqueue 後 return work。2PC 恢復由 NCCL fault callback 機制驅動，主執行緒不感知。

### Bug 6（已解決）：降級模式下 intraNodeComm fast-path 繞過 FT 邏輯

**根因：** `allreduce()` 中，`intraNodeComm_->allReduce()` 返回的 `IntraNodeCommWork` 完全繞過 `allreduce_impl()` 的 shadow buffer checkpoint 和 `collective()` 的容錯邏輯，在降級後使用舊 topology 產生錯誤結果或崩潰。

**修復：** `allreduce()` 中加入 `!is_degraded_` 條件，降級模式下強制走 `allreduce_impl`。

### Bug 7（已解決）：proxy 的 ATen 算術未在 ncclStream 上執行

**根因：** proxy step 2 的 `output.copy_()` 和 `output.add_()` 預設在 compute stream 執行，但 `ward_bufs` 的數據是由 ncclRecv 寫到 ncclStream 上的。兩個 stream 並行執行時，ATen 可能讀到 ncclRecv 尚未完成的數據。

**修復：** 使用 `setCurrentCUDAStream(stream)` + restore，確保算術核心在 ncclStream 上排入。

### shadow_seq_ 資料競爭（已解決）

**根因：** `shadow_seq_` 原為普通 `uint64_t`，在主執行緒（shadow_pre）寫，同時被 NCCL callback thread（trigger_fault_proposal）讀，屬 undefined behavior。

**修復：** 改為 `std::atomic<uint64_t>`；shadow_pre 用 `store(release)` 寫，trigger_fault_proposal 用 `load(acquire)` 讀，建立正確的 happens-before 關係。

### 側車依賴主執行緒（已解決）

**根因：** 舊設計中，recover_and_replay_inflight_ops() 只由主執行緒的 wait() 呼叫。若主執行緒卡死（例如在等待某個 Future 或 barrier），重播永遠不啟動，系統掛死。

**修復：** 側車在 2PC 達成共識並設定 final_commit_op_ 後，立即自行呼叫 recover_and_replay_inflight_ops()。recover 函式受 recovery_mutex_ 保護，無論是側車還是主執行緒呼叫，只有第一個進入的執行緒真正執行，後者拿到鎖後因 final_commit_op_ == 0 直接返回。

---

## 五、NCCL Custom Fork 端的確認狀態

| 項目 | 狀態 |
|------|------|
| `ncclIbResiliencyHandleDeviceFailure`: `ncclSystemError → ncclRemoteError` | ✅ 已完成 |
| `ncclIbResiliencyProbeHandleCompletionEvent`: `ncclSuccess → ncclRemoteError` | ✅ 已完成 |
| `ncclTopoPopulateNics`：global ban mask 軟性遮蔽（speed=0, latency=999999） | ✅ 已完成 |
| `ncclTopoGetLocalNetType`：modulo 前過濾 banned NIC（方案二） | ✅ 已完成 |
| `init.cc`: 新增 `ncclCommBanNicReset()` | ✅ 已完成 |
| `init.cc nccl_ft_trigger_fault`: 呼叫已註冊的 callback | ✅ 已完成 |
| `ncclCommRegisterFaultCallback` API 可用 | ✅ 已確認（在 first collective 成功呼叫） |
| `ncclCommBanNic` 作用域：**process-level global** | ✅ 已確認 |
| `NCCLFTComm::create` 每次 reinit 重新 topo discovery | ✅ 已確認 |
| **方案三 `ncclConfig_t::bannedNicsMask` ABI 變更** | ✅ 已 revert（採用方案二，無 ABI 影響） |

---

## 六、遺留效能注意事項

| 項目 | 影響 | 說明 |
|------|------|------|
| Inline GC O(n) 掃描 in_flight_shadow_bufs_ | 極低 | 在每次 allreduce 觸發 get_or_allocate 時執行；n = 飛行中 bucket 數，通常 < 100 |
| Memory Backpressure sleep 2ms | 輕微 latency | 只在 in-flight > 2GB 時觸發，視為異常保護 |
| 降級後仍執行 D2H copy | 輕微額外記憶體頻寬 | **設計決定**，為二次故障 replay 預備 shadow buffer |
| proxy step 2 建立 ward_bufs（at::empty_like） | GPU allocator 呼叫 | 每個降級 AllReduce 都會配置；可考慮預配置 |
