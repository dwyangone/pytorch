# ProcessGroupNCCLFT Shadow Ping-Pong Failover — Implementation Plan

## Confirmed Decisions

| Question | Decision |
|---|---|
| Proxy GPU selection | `proxy = (failed_dev + 1) % localDeviceCount_` |
| Degradation strategy | Symmetric: every node drops the same local device index |
| Initial test environment | Multi-node (2 nodes × N GPUs); test script written but run manually |

---

## Top-Level Overview

**Goal:** Extend the existing `ProcessGroupNCCLFT` C++ backend so that, when a NIC/GPU
reports a hardware fault, collective operations (starting with AllReduce) continue
producing mathematically-identical results via a *Shadow Ping-Pong* relay:

1. The faulty GPU scatters its tensor to a healthy proxy GPU on the same node over
   NVLink.
2. The proxy GPU pre-aggregates (adds) the faulty GPU's contribution to its own
   tensor.
3. The proxy GPU runs the cross-node NCCL collective on the reduced communicator
   (`proxy_global_comm_`).
4. The proxy GPU scatters the final result back to the faulty GPU.

The watchdog `runLoop` is the central place where NCCL errors are detected and
currently cause an unconditional abort. The new plan intercepts the error at that
point, classifies it as a recoverable hardware fault (NIC failure), and routes it
through the failover path instead of aborting.

---

## Sub-Task 1 — Intercept recoverable errors in the Watchdog `runLoop`

### Intent
Currently, when `work.checkAndSetException()` fires and `work.exception()` is
non-null, the watchdog always aborts and throws. We need to classify
`NCCLFaultToleranceError` (or a NCCL `ncclRemoteError`/`ncclSystemError` coming from
a known faulty NIC) as a *recoverable* event and divert the code to a recovery path
instead of calling `work.abort()` / `pg_->abortComms()`.

The diversion must be safe: if the error is not recoverable (e.g., a genuine CUDA
memory error), the original abort path is preserved.

### Expected Outcomes
- A `work.exception()` caused by a NIC-level NCCL error no longer aborts the
  process.
- The watchdog logs the error as recoverable, clears the exception on the `Work`
  object, and signals the PG to begin the failover negotiation phase.
- The original abort path remains intact for non-recoverable errors.

### Todo List
1. In `Watchdog::runLoop`, inside the `if (work.exception())` block (line ~2451),
   add a branch **before** the existing abort/throw code:
   ```
   if (!pg_->ft_disabled_) {
       // recoverable: clear exception, signal fault to ft_negotiator
       // do NOT call work.abort() or pg_->abortComms()
   } else {
       // original path: abort + throw
   }
   ```
2. In the recoverable branch:
   - Compute `int device_idx = work.device_.index() % pg_->localDeviceCount_`.
   - Call `work.setException(nullptr)` to clear the poison on the `Work` object
     so `WorkNCCLFT::wait()` does not rethrow.
   - Set `pg_->local_hardware_fault_dev_.store(device_idx)` so the existing
     `ft_negotiator_thread_` can pick it up and write the TCPStore proposal
     (re-uses the already-working mechanism).
   - Explicitly remove the faulty work from `workMetaList_` by advancing the
     iterator (`it = pg_->workMetaList_.erase(it)`) so the loop does not
     spin on a poisoned work object that will never complete.
   - Log a `LOG(WARNING)` indicating recovery mode and which device is affected.

### Relevant Context
- Error detection: `work.checkAndSetException()` at `ProcessGroupNCCLFT.cpp` line 2411,
  which calls `checkForNCCLErrorsInternal` (line 2906).
- Current abort block: lines 2451–2499.
- `NCCLFaultToleranceError` is defined in `NCCLFTUtils.hpp` line 26.
- `ft_negotiator_thread_` already handles writing the TCPStore proposal once
  `local_hardware_fault_dev_` is non-negative (see `start_ft_negotiator_thread`,
  line 3842).
- `work.device_` is the `at::Device` of the failed work (member of `WorkNCCLFT`,
  `ProcessGroupNCCLFT.hpp` line 417).

### Status
[ ] pending

---

## Sub-Task 2 — Initialize `local_nvlink_comm_` (intra-node NVLink communicator)

### Intent
Create a long-lived NCCL communicator that covers **only the GPUs on the current
node**, using the raw `ncclCommInitRank` path (not ProcessGroup infrastructure).
This communicator must never depend on cross-node NICs, so it survives any single
NIC failure. It is used in Sub-Tasks 4 & 5 for the scatter/gather relay steps.

### Expected Outcomes
- A `ncclComm_t local_nvlink_comm_` member is initialized once in the
  `ProcessGroupNCCLFT` constructor (or lazily on first use).
- The communicator has `size = localDeviceCount_` and
  `rank = rank_ % localDeviceCount_`.
- No cross-node traffic is involved in creating or using this communicator.

### Todo List
1. Add `ncclComm_t local_nvlink_comm_{nullptr};` to the private section of
   `ProcessGroupNCCLFT` in `ProcessGroupNCCLFT.hpp`.
2. Add a helper `void initLocalNvlinkComm()` declaration in the header and
   implement it in `ProcessGroupNCCLFT.cpp`.
3. In `initLocalNvlinkComm()`:
   - Compute `int node_id = rank_ / localDeviceCount_`.
   - Generate / exchange a `ncclUniqueId` using `globalStore_` with a
     node-scoped key (e.g. `"NCCL_FT_LOCAL_UID_NODE_<node_id>"`). Rank 0 on each
     node generates and writes the UID; all other ranks on the same node read it.
   - Determine `local_rank = rank_ % localDeviceCount_` and
     `local_size = localDeviceCount_`.
   - Call `ncclCommInitRank(&local_nvlink_comm_, local_size, uid, local_rank)`.
4. Call `initLocalNvlinkComm()` from the `ProcessGroupNCCLFT` constructor,
   **after** `store_` is fully available and **only when** `!ft_disabled_`.
5. Add teardown: call `ncclCommDestroy(local_nvlink_comm_)` in the destructor
   (guarded by `local_nvlink_comm_ != nullptr` check).

### Relevant Context
- Constructor: `ProcessGroupNCCLFT.cpp` ~line 979.
- Existing store key pattern used: `"NCCL_FT_EVENT"` (same store, different key).
- `localDeviceCount_` is already initialised at line 1002 from
  `at::cuda::getNumGPUs()`.
- Raw `ncclCommInitRank` is the same API used internally by `NCCLFTComm`; no
  wrapper needed for this prototype.

### Status
[ ] pending

---

## Sub-Task 3 — Implement `rebuild_shadow_ping_pong_topology()` (degrade)

### Intent
Build a new cross-node NCCL communicator `proxy_global_comm_` that excludes the
failed local device index from **every node** (symmetric degradation). Healthy
GPUs join; faulty GPUs skip. After this, the main collective path will use
`proxy_global_comm_` instead of the normal per-device comm.

### Expected Outcomes
- `proxy_global_comm_` is a valid `ncclComm_t` on healthy ranks and `nullptr`
  on faulty ranks.
- `proxy_comm_size_` and `proxy_comm_rank_` reflect the new reduced topology.
- The function is idempotent (calling it twice does not crash).

### Todo List
1. Add to `ProcessGroupNCCLFT.hpp` (private section):
   - `ncclComm_t proxy_global_comm_{nullptr};`
   - `int proxy_comm_rank_{-1};`
   - `int proxy_comm_size_{0};`
   - `int proxy_failed_local_dev_{-1};`
   - `std::atomic<bool> proxy_comm_ready_{false};`
2. Implement `rebuild_shadow_ping_pong_topology()` in
   `ProcessGroupNCCLFT.cpp`:
   - Read `failed_dev_index_.load()` to get the excluded local device index.
   - Determine whether **this** rank is the faulty one:
     `bool is_faulty = (rank_ % localDeviceCount_) == failed_local_dev`.
   - Compute new size: `proxy_comm_size_ = size_ - (size_ / localDeviceCount_)`.
   - Compute new rank for healthy ranks: iterate over all original ranks,
     assign new contiguous indices to those whose `rank % localDeviceCount_ != failed_local_dev`,
     and look up this rank's new index in that mapping.
   - Exchange a `ncclUniqueId` via `globalStore_` key
     `"NCCL_FT_PROXY_UID"` (rank 0 overall writes, others read).
   - If `!is_faulty`: call `ncclCommInitRank(&proxy_global_comm_, proxy_comm_size_, uid, proxy_comm_rank_)`.
   - If `is_faulty`: set `proxy_global_comm_ = nullptr`, skip init.
   - Store `proxy_failed_local_dev_ = failed_local_dev`.
   - Set `proxy_comm_ready_.store(true)` after init completes.
3. Call `rebuild_shadow_ping_pong_topology()` inside the existing fence block in
   `collective()` (around line 4066), just before `this->is_degraded_ = true`.

### Relevant Context
- The fence/brake block: `ProcessGroupNCCLFT.cpp` ~lines 4055–4077.
- The function stub is declared at `ProcessGroupNCCLFT.hpp` line 1092.
- `size_` and `rank_` are `Backend` members.
- The symmetric-degradation assumption (all nodes drop the same device index)
  is a deliberate simplification for this prototype.

### Status
[ ] pending

---

## Sub-Task 4 — Implement AllReduce Shadow Ping-Pong execution path

### Intent
Inside the `collective()` template function, replace the stub `is_degraded_` branch
with the actual four-step relay:
1. **Local Scatter**: faulty GPU sends tensor to proxy GPU via `local_nvlink_comm_`.
2. **Pre-aggregate**: proxy GPU adds the received tensor to its own in-place.
3. **Global AllReduce**: proxy GPU runs `ncclAllReduce` on `proxy_global_comm_`.
4. **Local Gather**: proxy GPU sends result back to faulty GPU via
   `local_nvlink_comm_`.

All four steps are asynchronous CUDA operations enqueued on `ncclStream`.

### Expected Outcomes
- When `is_degraded_ == true` and `opType == ALLREDUCE`, the output tensor on
  every healthy and faulty rank equals the mathematically correct AllReduce
  result (sum of all ranks' inputs divided by world size for AVG, or sum for SUM).
- No rank hangs: every rank either participates in the relay or receives the
  result via NVLink.
- The existing `Work` object lifecycle (timing, watchdog, future) is unchanged.

### Todo List
1. Add a helper `void execute_shadow_allreduce(at::Tensor& input, at::Tensor& output, at::cuda::CUDAStream& stream, ncclRedOp_t op)` (declaration in header, definition in `.cpp`).
2. Implement the helper:
   - Compute `int local_rank = rank_ % localDeviceCount_`.
   - Compute `int proxy_local_rank = (proxy_failed_local_dev_ + 1) % localDeviceCount_`.
   - Determine role: `bool is_faulty = (local_rank == proxy_failed_local_dev_)`.
   - Determine role: `bool is_proxy  = (local_rank == proxy_local_rank)`.
   - If `is_faulty`:
     - `ncclGroupStart()`
     - `ncclSend(input.data_ptr(), input.numel(), dtype, proxy_local_rank, local_nvlink_comm_, stream)`
     - `ncclRecv(output.data_ptr(), output.numel(), dtype, proxy_local_rank, local_nvlink_comm_, stream)`
     - `ncclGroupEnd()`
   - If `is_proxy`:
     - Allocate `proxy_buf` (same shape/dtype as `input`) on this device.
     - `ncclGroupStart()`
     - `ncclRecv(proxy_buf, ..., failed_local_rank, local_nvlink_comm_, stream)`
     - `ncclGroupEnd()`
     - `at::cuda::CUDAStreamGuard guard(stream); output.add_(proxy_buf)`
     - `ncclAllReduce(output.data_ptr(), output.data_ptr(), ..., op, proxy_global_comm_, stream)`
     - `ncclGroupStart()`
     - `ncclSend(output.data_ptr(), ..., failed_local_rank, local_nvlink_comm_, stream)`
     - `ncclGroupEnd()`
   - Otherwise (healthy, non-proxy rank):
     - `ncclAllReduce(input.data_ptr(), output.data_ptr(), ..., op, proxy_global_comm_, stream)`
   - Guard all calls with `C10D_NCCL_FT_CHECK`.
3. In `collective()`, replace the stub body inside `if (C10_UNLIKELY(this->is_degraded_))` (lines 4100–4121) with a call to `execute_shadow_allreduce`, passing the `ncclRedOp_t` extracted from the captured `fn` lambda's reduce op (or add an overload that accepts `ncclRedOp_t` directly).
4. For non-AllReduce ops when `is_degraded_` is true: log a `LOG(WARNING)` and
   fall through to the native NCCL path on `proxy_global_comm_` where possible,
   or assert for unsupported ops.
5. Guard on `proxy_comm_ready_.load()` before using `proxy_global_comm_`.

### Relevant Context
- The stub branch: `ProcessGroupNCCLFT.cpp` lines 4100–4121.
- `C10D_NCCL_FT_CHECK` defined in `NCCLFTUtils.hpp` line 49.
- `ncclGroupStart` / `ncclGroupEnd` wrappers: `ProcessGroupNCCLFT::groupStart/groupEnd`.
- ATen in-place add on CUDA stream: `at::cuda::CUDAStreamGuard` + `output.add_(other)`.

### Status
[ ] pending

---

## Open Questions / Known Risks

1. **Error type classification deferred** — the prototype treats all NCCL errors as
   recoverable network errors (`ft_disabled_` is the only gate). Finer-grained
   classification (e.g., distinguishing `ncclRemoteError` from `ncclInternalError`)
   is explicitly deferred to after the prototype is working.

2. **`proxy_global_comm_` init synchronisation** — `rebuild_shadow_ping_pong_topology()`
   is called from the collective hot path on the main thread. A `proxy_comm_ready_`
   atomic flag (added in Sub-Task 3) guards against using a half-built communicator
   in Sub-Task 4.

3. **Proxy selection for > 2 GPUs per node** — the current plan hard-codes
   `proxy = (failed + 1) % localDeviceCount_`. For an 8-GPU node this is fine for
   the prototype but creates an unbalanced hotspot. A load-balanced multi-proxy
   scheme is out of scope.

4. **Non-AllReduce collectives** — AllGather, ReduceScatter, Broadcast all have
   different semantics. The `is_degraded_` guard currently falls through to the
   native NCCL path which will fail or hang because `proxy_global_comm_` has a
   different rank topology. For the prototype, these should log a warning and
   bypass the shadow path.

5. **`ncclCommRegisterFaultCallback` / `ncclCommBanNic` integration** — these
   custom NCCL APIs are the intended trigger source. The new Sub-Task 1 provides
   a second, independent trigger path (via watchdog error detection) that does not
   rely on the callback API, making the system more robust.

6. **Work object state after recovery** — Sub-Task 1 explicitly erases the faulty
   work from `workMetaList_` immediately after clearing its exception, so the
   watchdog does not spin on a poisoned comm. The training loop resumes with
   subsequent ops going through the degraded path once `is_degraded_` is set.
