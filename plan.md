# ProcessGroupNCCLFT — Shadow Ping-Pong Failover: Implementation Plan

> 以 `ProcessGroupNCCLFT.cpp` / `.hpp` 目前實作狀態為準（2026-08，最新修訂）

---

## 一、系統架構速覽

`ProcessGroupNCCLFT` 是基於 `ProcessGroupNCCL` 複製並擴充的自訂 NCCL 容錯 backend（c10d 名稱 `nccl_ft`）。詳細架構請參見 `system_arch.md`。

### 三執行緒模型

| 執行緒 | 名稱 | 職責 |
|--------|------|------|
| 主執行緒 | — | 執行 collective；故障後在 `wait()` FT 防線等待或自行呼叫 `recover_and_replay_inflight_ops()` |
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
| `initLocalNvlinkComm()` — 使用 NCCLFTComm::create 從頭建立 NVLink comm（不用 ncclCommSplit）| ~4092 | ✅ |
| `rebuild_shadow_ping_pong_topology()` — NCCLFTComm::create reinit + ncclCommBanNic（單一 for 迴圈，不重複） | ~4166 | ✅ |
| `execute_shadow_allreduce()` — FAULTY/PROXY/HEALTHY 四角色邏輯（含 findProxy + my_wards） | ~4326 | ✅ |
| `trigger_fault_proposal()` — NCCL callback，bitmask fetch_or，shadow_seq_ atomic load | ~4520 | ✅ |
| `start_ft_negotiator_thread()` — 2PC side-car，per-rank PROPOSE key，any_proposed 快速路徑，min agreed_ss | ~4554 | ✅ |
| Side-car 2PC 完成後直接呼叫 `recover_and_replay_inflight_ops()`（不等主執行緒） | ~4874 | ✅ |
| `get_or_allocate_shadow_context()` — 三段式設計（Inline GC + Best-Fit + Memory Backpressure + 鎖外 alloc） | ~4894 | ✅ |
| `ShadowContext::compute_event` — 提升為 ShadowContext 成員，不再每次 allreduce 建立新 event | hpp ~1182 | ✅ |
| `ShadowContext::work_ptr` — 綁定 Work，供 Inline GC 主動回收成功 bucket | hpp ~1183 | ✅ |
| `recover_and_replay_inflight_ops()` — 全域重播中心，abort→rebuild→H2D→replay→setException/markCompleted | ~5297 | ✅ |
| recover 內部直接 `work_ptr->setException(nullptr)` + `future_->markCompleted` 解鎖 DDP | ~5367-5373 | ✅ |
| `WorkNCCLFT::wait()` 整合式 FT 防線（情況 A/B/C + 後處理 GC） | ~952 | ✅ |
| `allreduce_impl()` shadow_pre lambda — work_ptr + compute_event + atomic shadow_seq_ | ~6034 | ✅ |
| `collective()` future_ 提前初始化（pre() 呼叫前），消除 Race Condition | ~5165 | ✅ |
| `collective()` 二分支設計（行 5180-5251）— `is_degraded_` 直接執行 shadow allreduce | ~5180 | ✅ |
| Fix A：移除 Watchdog fallback fault_mask 寫入 | Watchdog runLoop ~2644 | ✅ |
| Fix 1：每個 rank 獨立呼叫 `ncclCommRegisterFaultCallback` | collective() lazy-init ~5039 | ✅ |
| Fix 3：Watchdog early-abort（final_commit_op_ > 0 時強制 setException） | Watchdog runLoop ~2550 | ✅ |
| Fix D：collective() 原生路徑捕獲 NCCLFaultToleranceError，設 exception，提早 return | collective() ~5221 | ✅ |
| Fix D2：`initLocalNvlinkComm()` try-catch + `nvlink_init_attempted_` flag | collective() lazy-init ~5063 | ✅ |
| Bug 1 fix：FT 模式下 Watchdog 不設 COMM_ERROR | Watchdog runLoop ~2536 | ✅ |
| Bug 2 fix：future_ 在 pre() 前初始化（消除 catch 區塊需要重建 future_ 的問題） | collective() ~5165 | ✅ |
| Bug 3 確認：recover_and_replay seq >= agreed_ss 是正確的 | ~5327 | ✅ |
| Bug 4 fix：get_or_allocate_shadow_context 三段式設計（鎖外 alloc） | ~4894 | ✅ |
| Bug 5 確認：降級後 shadow_pre 仍執行 D2H copy — **設計決定** | shadow_pre | ✅ |
| Bug 6 fix：`allreduce()` 降級模式跳過 intraNodeComm fast-path | allreduce() ~6114 | ✅ |
| Bug 7 fix：proxy 的 ATen 算術（copy_/add_）在 ncclStream 上執行（setCurrentCUDAStream） | execute_shadow_allreduce ~4461 | ✅ |
| Bug 10 fix：降級模式下 work->ncclComm_ 指向實際使用的 comm | collective() ~5194 | ✅ |
| Bug 12 fix：多 NIC 同時故障用 bitmask fetch_or | trigger_fault_proposal ~4531 | ✅ |
| Bug A fix：`ft_round_++` 在 recover_and_replay 末尾執行 + TCPStore cleanup | ~5377 | ✅ |
| Bug B fix：用 `final_commit_op_ == 0` 取代 `rollback_done_` 判斷重入 | ~5302 | ✅ |
| Bug C fix：side-car Phase 1b OR 累積更新 `faulty_local_devs_`（逐步故障支援） | ~4794 | ✅ |
| shadow_seq_ 資料競爭修復：改為 `std::atomic<uint64_t>` | hpp ~1201, cpp ~5062, ~4540 | ✅ |
| compute_done event per-call fix：提升為 ShadowContext::compute_event | shadow_pre / get_or_allocate | ✅ |
| ncclCommSplit 在 RemoteError comm 上失敗 fix：改用 NCCLFTComm::create（TCPStore rendezvous） | initLocalNvlinkComm ~4092 | ✅ |
| 側車依賴主執行緒 fix：側車在 2PC 完成後直接呼叫 recover_and_replay | ~4874 | ✅ |
| UINT64_MAX sentinel（agreed_ss == UINT64_MAX 表示無 checkpoint） | side-car & barrier | ✅ |
| 2PC side-car 強制 abort globalComm 以喚醒卡死的主執行緒 | start_ft_negotiator_thread ~4858 | ✅ |
| Watchdog GC：AllReduce 成功後釋放 in_flight_shadow_bufs_ 到 free pool | Watchdog runLoop ~2839 | ✅ |
| Inline GC：get_or_allocate 內主動回收已成功的 bucket | ~4900-4921 | ✅ |
| Memory Backpressure：in-flight > 2GB 時 sleep 等待 | ~4944-4955 | ✅ |
| Best-Fit buffer 搜尋：取代舊版第一個 numel >= 的策略 | ~4924-4941 | ✅ |
| wait() FT 後處理 GC：replay 完成後由 wait() 回收 shadow buffer | wait() ~1019-1033 | ✅ |
| `ShadowContext` struct（含 per-seq copy_event / replayed_end_event / compute_event / work_ptr / reduce_op） | hpp ~1178 | ✅ |
| findProxy()：round-robin 尋找下一個健康 proxy（支援多 faulty） | ~4315 | ✅ |
| my_wards vector：proxy 可代理多個 faulty rank | execute_shadow_allreduce ~4352 | ✅ |
| **多維 Tensor Shape Mismatch fix**：shadow_pre D2H copy + H2D restore 均加 `.flatten()` | shadow_pre ~6058, recover_and_replay ~5355 | ✅ |
| **迴圈內 CUDAEvent 效能 fix**：recover_and_replay 迴圈外宣告 `restore_done`，迴圈內重複 record | recover_and_replay ~5337 | ✅ |
| **Watchdog 例外清除後 GC 保底**：clearing path 將非空 stash push 到 shelvesToUnstash_ | Watchdog runLoop ~2679 | ✅ |
| any_proposed 快速路徑：side-car 無故障時 sleep 50ms 避免空轉 | ~4686 | ✅ |

---

## 三、待確認與待修復項目

### P0：高危 Bug（應優先修復）

| Bug | 位置 | 說明 | 修復方向 |
|-----|------|------|---------|
| `WorkNCCLFT::logPrefix()` static Bug | ~724 | 行 724 程式碼有警告注解但 `static` 關鍵字**仍然存在**；function-local static 以第一個呼叫者的 `rank_` 初始化後固定不變，所有後續 Work 拿到錯誤前綴 | 移除 `static` 關鍵字（僅一字之差） |

### P0：確認 NCCL Fork 行為

**問題 1：** `ncclCommBanNic(dev_idx)` 是 process-level global 還是 per-comm？
- 如果是 process-level：rebuild 前呼叫一次即可，後續 `NCCLFTComm::create` 建立的新 comm 自動跳過被 ban 的 NIC。
- 如果是 per-comm：ban 在 abort 後失效，需要在 reinit 前再次呼叫（目前 `rebuild_shadow_ping_pong_topology` 已在所有 rank 呼叫 ban，應可覆蓋此情況）。

**問題 2：** `NCCLFTComm::create`（即 `ncclCommInitRank`）是否每次都重新執行 topo discovery？
- 需要實驗確認：呼叫 `ncclCommBanNic(0)` 後再建立新 comm，確認新 comm 不使用 mlx5_0。

**問題 3：** 側車在 recovery_mutex_ 上的執行緒競爭
- 側車自行呼叫 recover_and_replay（持鎖）；主執行緒的 wait() 也可能觸發 recover_and_replay（等鎖）
- 兩者序列化，不死結；後進者拿到鎖後 `final_commit_op_ == 0` 直接返回
- 確認主執行緒等鎖時不會造成 DDP timeout 問題

### P1：低危清理

| 項目 | 位置 | 說明 |
|------|------|------|
| `get_or_allocate_shadow_buffer` 殭屍函式宣告 | hpp ~1229 | `at::Tensor get_or_allocate_shadow_buffer(const at::Tensor& t)` 仍保留在 hpp 宣告；應刪除 |
| `initLocalNvlinkComm` TCPStore key 殘留 | ~4103 | `"NCCL_FT_LOCAL_COMM_ID_NODE_<id>_PG_<uid>"` key 永久殘留；可在成功讀取後 `deleteKey` |
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
| Inline GC O(n) 掃描 in_flight_shadow_bufs_ | 極低 | 在每次 allreduce 觸發 get_or_allocate 時執行；n = 飛行中 bucket 數，通常 < 100 |
| Memory Backpressure sleep 2ms | 輕微 latency | 只在 in-flight > 2GB 時觸發，視為異常保護 |
| 降級後仍執行 D2H copy | 輕微額外記憶體頻寬 | **設計決定**，為二次故障 replay 預備 shadow buffer |
| proxy step 2 建立 ward_bufs（at::empty_like） | GPU allocator 呼叫 | 每個降級 AllReduce 都會配置；可考慮預配置 |
