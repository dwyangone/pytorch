# ProcessGroupNCCLFT Shadow Ping-Pong Failover — Implementation Plan

## Confirmed Decisions

| Question | Decision |
|---|---|
| Proxy GPU selection | `proxy = (failed_dev + 1) % localDeviceCount_` |
| Degradation strategy | Symmetric: every node drops the same local device index |
| Initial test environment | Multi-node (2 nodes × N GPUs); test script run manually |

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

### Status: ✅ DONE

**What was implemented:**
- In `Watchdog::runLoop`, inside the `if (work.exception())` block, a branch is
  added **before** the existing abort/throw code checking `!pg_->ft_disabled_`.
- Recoverable branch: computes `device_idx`, resets `pg_->error_` to SUCCESS,
  stores `device_idx` into `local_hardware_fault_dev_`, calls
  `work.setException(nullptr)`, erases the poisoned work from `workMetaList_`.
- The original abort path is preserved for `ft_disabled_` mode.

---

## Sub-Task 2 — Initialize `local_nvlink_comm_` (intra-node NVLink communicator)

### Status: ✅ DONE

**What was implemented:**
- `local_nvlink_comm_` declared as `std::shared_ptr<NCCLFTComm>`.
- `initLocalNvlinkComm()` uses `NCCLFTComm::split` (not `ncclCommInitRank`) with
  `color = node_id` so each node gets its own sub-communicator.
- Lazy init: called on the first `collective()` call (not in constructor) because
  `ncclCommSplit` is collective and requires the global comm to already exist.
- Destructor uses `local_nvlink_comm_.reset()` (RAII, no manual `ncclCommDestroy`).

**Why ncclCommSplit instead of ncclCommInitRank:**
`ncclCommInitRank` triggers a full NIC topology scan. In setups where each GPU has
a dedicated NIC only visible from its NUMA context, this crashes with
"Could not find any local path from gpu N to net". `ncclCommSplit` inherits the
already-validated topology from the global comm.

---

## Sub-Task 3 — Implement `rebuild_shadow_ping_pong_topology()` (degrade)

### Status: ✅ DONE (including critical ncclCommSplit fix)

**What was implemented:**
- `proxy_global_comm_` changed from `ncclComm_t` to `std::shared_ptr<NCCLFTComm>`.
- `rebuild_shadow_ping_pong_topology()` now uses `NCCLFTComm::split` with
  `color=1` for healthy ranks and `NCCL_SPLIT_NOCOLOR` for the faulty rank.
  This avoids the NIC rescan crash that `ncclCommInitRank` would trigger.
- All ranks call `ncclCommSplit` simultaneously (collective contract satisfied).
- The faulty rank gets `NCCL_SPLIT_NOCOLOR` → excluded from the sub-communicator
  → `proxy_global_comm_.reset()` after split.
- `proxy_comm_ready_` atomic guards against using a half-built communicator.
- Second-fault safety: `proxy_global_comm_.reset()` before rebuild.

---

## Sub-Task 4 — Implement AllReduce Shadow Ping-Pong execution path

### Status: ✅ DONE (including AVG correctness fix and ReduceOp threading)

**What was implemented:**
- `execute_shadow_allreduce` signature changed to accept `ReduceOp reduceOp` (not
  `ncclRedOp_t op`) so the caller passes the semantic op, not the NCCL wire format.
- `current_shadow_reduce_op_` member added; set by `allreduce_impl` before calling
  `collective()`, read in the degraded branch of `collective()`.
- `collective()` degraded branch now calls:
  ```cpp
  execute_shadow_allreduce(inputs[0], outputs[0], ncclStream, current_shadow_reduce_op_);
  ```
  instead of the old hardcoded `ReduceOp::SUM`.
- **AVG workaround**: when `reduceOp == ReduceOp::AVG`, the proxy and healthy ranks
  use `ncclSum` on `proxy_global_comm_` and then manually call `output.div_(size_)`
  on the CUDA stream. This avoids the wrong divisor (`proxy_comm_size_` instead of
  `size_`) that `ncclAvg` would apply.
- Four-step relay:
  1. FAULTY: `ncclSend` → proxy; `ncclRecv` ← proxy (both in one `ncclGroup`).
  2. PROXY: `ncclRecv` from faulty; `output = input + proxy_buf`; `ncclAllReduce`
     on `proxy_global_comm_`; optional `div`; `ncclSend` back to faulty.
  3. HEALTHY: `ncclAllReduce` on `proxy_global_comm_`; optional `div`.

---

## Sub-Task 5 — Fix 2PC Negotiation (was P1 gap)

### Status: ✅ DONE

**What was implemented:**
Full two-phase commit protocol in `start_ft_negotiator_thread()`:

**Phase 1 (PROPOSE):**
- Any rank's side-car writes `NCCL_FT_EVENT = "PROPOSE:<op>:<node>:<dev>"`.
- Every rank's side-car reads the key, sets `do_not_cross_op_` (local fence),
  sets `failed_dev_index_`, and writes its own ACK key:
  `"NCCL_FT_ACK_<rank>_<target_op>"`.

**Phase 2 (COMMIT):**
- Rank 0's side-car polls until all `size_` ACK keys exist, then writes
  `"NCCL_FT_COMMIT_<target_op>"`.
- Every rank's side-car polls for the COMMIT key; only after it arrives does it
  set `final_commit_op_`, unblocking the main thread.

This ensures all ranks stop at the same OP boundary before any rank rebuilds the
proxy topology and enters degraded mode.

---

## Open Questions / Known Risks

1. **Error type classification deferred** — the prototype treats all NCCL errors as
   recoverable network errors (`ft_disabled_` is the only gate). Finer-grained
   classification (e.g., distinguishing `ncclRemoteError` from `ncclInternalError`)
   is explicitly deferred to after the prototype is working.

2. **Non-AllReduce collectives** — AllGather, ReduceScatter, Broadcast all have
   different semantics. The `is_degraded_` guard currently falls through to the
   native NCCL path on the original (broken) comm. For the prototype, these will
   log a warning. A subsequent work item adds proper Shadow Ping-Pong semantics for
   these ops.

3. **Single failed device** — `proxy_failed_local_dev_` is a single int. A second
   fault after the first triggers recovery again, but the first faulty GPU's
   exclusion is reset. Future work: `std::unordered_set<int> failed_local_devs_`.

4. **`local_nvlink_comm_` key versioning** — uses a fixed key
   `"NCCL_FT_LOCAL_UID_NODE_<N>"` (but now uses `ncclCommSplit`, not a UID key, so
   this risk is eliminated).

5. **TCPStore key cleanup** — `NCCL_FT_ACK_<rank>_<op>` and
   `NCCL_FT_COMMIT_<op>` keys accumulate per-fault. A cleanup pass after each
   successful commit would keep the store tidy; deferred to production hardening.

---

## Remaining Work (Post-Prototype)

| Priority | Item |
|---|---|
| P0 | AllGather degraded path — ZeRO/FSDP `_allgather_base` will fail after NIC fault |
| P0 | ReduceScatter degraded path — ZeRO/FSDP `_reduce_scatter_base` same issue |
| P1 | Multi-fault support — `proxy_failed_local_dev_` is a single int |
| P1 | Error type classification — don't recover from `ncclInternalError` / GPU memory |
| P2 | TCPStore key cleanup — ACK/COMMIT keys accumulate; add post-commit cleanup |
| P2 | Broadcast degraded path |
| P3 | Proxy load balancing — dynamic selection vs fixed `(failed+1) % N` |
| P3 | Tensor shard fan-out — split faulty tensor across all healthy NICs |
