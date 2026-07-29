# ProcessGroupNCCLFT Shadow Ping-Pong Failover — Implementation Plan

## Confirmed Decisions

| Question | Decision |
|---|---|
| Proxy GPU selection | `proxy = (failed_dev + 1) % localDeviceCount_` |
| Degradation strategy | Symmetric: every node drops the same local device index |
| Initial test environment | Multi-node (2 nodes x N GPUs); test script run manually |
| NCCL init strategy | `ncclCommSplit` everywhere — NO second `ncclCommInitRankConfig` ever |
| Callback registration | Lazy: after first collective(), when topology is settled |

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

### Status: DONE

**What was implemented:**
- In `Watchdog::runLoop`, inside the `if (work.exception())` block, a branch is
  added **before** the existing abort/throw code checking `!pg_->ft_disabled_`.
- Recoverable branch: computes `device_idx`, resets `pg_->error_` to SUCCESS,
  stores `device_idx` into `local_hardware_fault_dev_`, calls
  `work.setException(nullptr)`, erases the poisoned work from `workMetaList_`.
- The original abort path is preserved for `ft_disabled_` mode.

---

## Sub-Task 2 — Initialize `local_nvlink_comm_` (intra-node NVLink communicator)

### Status: DONE

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

### Status: DONE (including critical ncclCommSplit fix)

**What was implemented:**
- `proxy_global_comm_` changed from `ncclComm_t` to `std::shared_ptr<NCCLFTComm>`.
- `rebuild_shadow_ping_pong_topology()` now uses `NCCLFTComm::split` with
  `color=1` for healthy ranks and `NCCL_SPLIT_NOCOLOR` for the faulty rank.
  This avoids the NIC rescan crash that `ncclCommInitRank` would trigger.
- All ranks call `ncclCommSplit` simultaneously (collective contract satisfied).
- The faulty rank gets `NCCL_SPLIT_NOCOLOR` -> excluded from the sub-communicator
  -> `proxy_global_comm_.reset()` after split.
- `proxy_comm_ready_` atomic guards against using a half-built communicator.
- Second-fault safety: `proxy_global_comm_.reset()` before rebuild.

---

## Sub-Task 4 — Implement AllReduce Shadow Ping-Pong execution path

### Status: DONE (including AVG correctness fix and ReduceOp threading)

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
  1. FAULTY: `ncclSend` -> proxy; `ncclRecv` <- proxy (both in one `ncclGroup`).
  2. PROXY: `ncclRecv` from faulty; `output = input + proxy_buf`; `ncclAllReduce`
     on `proxy_global_comm_`; optional `div`; `ncclSend` back to faulty.
  3. HEALTHY: `ncclAllReduce` on `proxy_global_comm_`; optional `div`.

---

## Sub-Task 5 — Fix 2PC Negotiation (was P1 gap)

### Status: DONE

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

## ACTIVE BLOCKER — Custom NCCL FT Plugin Bans NIC 0 at Init Time

### Status: ROOT CAUSE IDENTIFIED — Fix required in custom NCCL source

### Evidence from test.log (lines 157-180)

```
[3] NCCL INFO TOPO/NET : Importing network plugins to topology
[3] NCCL INFO NCCL-FT: 物理遮蔽故障網卡 0    <-- fires DURING topology import
[3] NCCL INFO NCCL-FT: 物理遮蔽故障網卡 0    <-- fires TWICE
...
[4] graph/topo.cc:1789 NCCL WARN Could not find any local path from gpu 0 to net.
```

The message `NCCL-FT: 物理遮蔽故障網卡 0` is printed by the **custom NCCL source
code** (not by ProcessGroupNCCLFT). It fires during `TOPO/NET: Importing network
plugins to topology`, which is a phase **inside** `ncclCommInitRankConfig`. This
means the custom NCCL's FT plugin has an initialization hook that runs at topology
construction time and unconditionally bans NIC index 0 for every rank — even
before any actual NIC fault has occurred.

Consequence: every rank then fails `graph/topo.cc:1789` with "Could not find any
local path from gpu 0 to net" because NIC 0 was removed from the topology.

### Identified Trigger: Malformed NCCL_IB_HCA env var

The launch script sets:
```bash
export NCCL_IB_HCA==mlx5_0,=mlx5_1,=mlx5_2,...
```

Note the double `==` at the start and the `=` prefix before every HCA name. In
standard NCCL, `=mlx5_N` means "port-exclusive assignment for rank N". The leading
double `==` is non-standard and may be parsed as an empty first token followed by
`mlx5_0`, or as a "ban" directive.

The custom NCCL's FT plugin reads this string during topology import (as part of
the NET plugin's `getProperties` or `listen` callback) and interprets the `=`
prefix on the first token as a NIC-ban instruction, passing `dev_idx=0` to the
physical-ban function. This fires twice because the NET plugin is imported once
for the IB plugin and once for the GIN plugin (both are loaded per the log).

### Fix required in the custom NCCL source code

**Location to look:** `src/transport/net.cc` or `src/misc/param.cc` in the custom
NCCL — specifically the code that processes `NCCL_IB_HCA` during topology import
and the FT plugin's init callback.

**There are two separate bugs to fix:**

#### Bug 1: FT init hook must NOT run during topology construction

The code that calls the physical NIC ban function must be gated on an explicit
user call to `ncclCommBanNic()` — it must NOT run automatically during
`ncclCommInitRankConfig` / `TOPO/NET: Importing network plugins`. The FT init
hook should only register internal data structures (callback table, fault state
variables) — it should never call the ban function.

Pseudo-fix in NCCL source:
```c
// WRONG (current): ban is called unconditionally at plugin init
static ncclResult_t ncclFTNetInit(struct ncclComm* comm) {
    // ... parses NCCL_IB_HCA ...
    ncclFTBanNic(dev_idx);   // <-- REMOVE THIS from init path
}

// CORRECT: ban is only called via explicit ncclCommBanNic() API
ncclResult_t ncclCommBanNic(int dev_idx) {
    // ... only runs when explicitly called ...
    ncclFTBanNic(dev_idx);   // stays here
}
```

#### Bug 2: Double-fire (fires twice per rank)

The ban function is called twice per rank. This likely means the FT init hook is
registered on two NET plugins (IB and GIN) or the `TOPO/NET: Importing` loop
iterates over the HCA list and fires the hook for every entry it finds. Fix: add
a `bool ft_init_done` guard in the FT init hook so it only runs once per comm.

#### Bug 3: NCCL_IB_HCA format (fix the env var if possible)

The safest fix (in addition to Bug 1) is to also correct the env var format:
```bash
# Wrong (has leading double == and = prefix per NIC):
export NCCL_IB_HCA==mlx5_0,=mlx5_1,...

# Correct (standard NCCL format, one NIC per rank):
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

If the launch script cannot be changed, the PyTorch C++ side can override the
env var from the `ProcessGroupNCCLFT` constructor before `ncclCommInitRankConfig`:

```cpp
// In ProcessGroupNCCLFT constructor, before any NCCL call:
// Each rank uses its own NIC: mlx5_<local_rank>
int local_rank = rank_ % localDeviceCount_;
std::string hca = "mlx5_" + std::to_string(local_rank);
setenv("NCCL_IB_HCA", hca.c_str(), 1 /* overwrite */);
```

This `setenv` approach is a workaround that can be applied in the PyTorch
backend without touching the launch script, but **Bug 1 in the custom NCCL
source must also be fixed** — overriding the env var alone will not prevent
the FT init hook from running and potentially banning a NIC.

### Verification after fix

After fixing Bug 1 in the custom NCCL:

1. The `NCCL-FT: 物理遮蔽故障網卡 0` messages should **not** appear during
   `TOPO/NET: Importing network plugins to topology`.
2. `ncclCommInitRankConfig` should complete without `Could not find any local
   path from gpu N to net` warnings.
3. ProcessGroupNCCLFT's lazy-init block in `collective()` should log:
   `[NCCL-FT] Fault callback registered on comm ...`
   `[NCCL-FT] local_nvlink_comm_ ready: ...`
   without errors.

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

4. **TCPStore key cleanup** — `NCCL_FT_ACK_<rank>_<op>` and
   `NCCL_FT_COMMIT_<op>` keys accumulate per-fault. A cleanup pass after each
   successful commit would keep the store tidy; deferred to production hardening.

---

## NCCL_IB_HCA Env Override (PyTorch-side workaround)

As a belt-and-suspenders measure, even after fixing Bug 1 in the custom NCCL,
add this to the `ProcessGroupNCCLFT` constructor to guarantee the env var is
correct regardless of what the launch script sets:

```cpp
// ProcessGroupNCCLFT constructor — before any NCCL init
if (!ft_disabled_) {
    int local_rank = rank_ % localDeviceCount_;
    std::string hca = "mlx5_" + std::to_string(local_rank);
    if (setenv("NCCL_IB_HCA", hca.c_str(), 1) != 0) {
        LOG(WARNING) << logPrefix()
                     << "[NCCL-FT] Failed to set NCCL_IB_HCA=" << hca;
    } else {
        LOG(INFO) << logPrefix()
                  << "[NCCL-FT] Set NCCL_IB_HCA=" << hca
                  << " for local_rank=" << local_rank;
    }
}
```

This is safe to apply now in `ProcessGroupNCCLFT.cpp` even before the NCCL
source bug is fixed, and it isolates each rank to only its own NIC.

---

## Remaining Work (Post-Prototype)

| Priority | Item |
|---|---|
| P0 | Fix custom NCCL Bug 1: FT init hook must not ban NICs at topology import time |
| P0 | Fix custom NCCL Bug 2: double-fire guard (fires twice per rank) |
| P0 | Fix NCCL_IB_HCA format (either in launch script or via setenv in constructor) |
| P0 | AllGather degraded path — ZeRO/FSDP `_allgather_base` will fail after NIC fault |
| P0 | ReduceScatter degraded path — ZeRO/FSDP `_reduce_scatter_base` same issue |
| P1 | Multi-fault support — `proxy_failed_local_dev_` is a single int |
| P1 | Error type classification — don't recover from `ncclInternalError` / GPU memory |
| P2 | TCPStore key cleanup — ACK/COMMIT keys accumulate; add post-commit cleanup |
| P2 | Broadcast degraded path |
| P3 | Proxy load balancing — dynamic selection vs fixed `(failed+1) % N` |
| P3 | Tensor shard fan-out — split faulty tensor across all healthy NICs |
