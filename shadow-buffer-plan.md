# Shadow Buffer + Rollback for AllReduce — Implementation Plan

## Confirmed Design Decisions

| Question | Decision |
|---|---|
| Tensor size stability | Fixed per model; safe to allocate once and reuse |
| Rollback granularity | Per-op: only the failed AllReduce tensor is restored; prior successful ops untouched |
| Buffer sizing | One shadow buffer per rank, sized to largest tensor seen; grown if a larger tensor arrives |
| Cross-rank sync | TCPStore control channel (dedicated NIC, always healthy) |
| Fault scope | Only AllReduce (DDP). Other ops not covered by shadow buffer. |
| asyncOp mode | DDP default is asyncOp=false (compute stream). asyncOp=true uses separate ncclStream. Both handled. |

---

## Problem Statement

When an IB NIC fails mid-AllReduce, NCCL has already written partial data into the
output tensor (the PyTorch gradient buffer). The content is a mix of partial results
and stale values — neither correct nor consistently wrong across ranks. Without
protection, the optimizer step that follows uses this corrupted gradient, permanently
damaging the model weights.

The goal is:
1. Before any AllReduce, snapshot the input tensor into a shadow buffer (on NCCL stream).
2. If the AllReduce fails, restore the tensor from the snapshot (on NCCL stream, event-fenced).
3. All ranks must agree on what the last clean op number was (via TCPStore).
4. After restoration and 2PC barrier, re-execute the failed op via Shadow Ping-Pong.

---

## Stream Architecture

Understanding which stream is used is critical for correctness.

### asyncOp=false (DDP default, synchronous bucket AllReduce)

```
collective():4532:
  ncclStream = at::cuda::getCurrentCUDAStream(device.index())
                            ^
                            same stream as backward compute
```

Timeline:
```
compute stream: [backward grad compute] -> [shadow copy (pre)] -> [ncclAllReduce] -> [next bucket backward]
```

The shadow copy and the AllReduce are on the same stream, so order is guaranteed
by CUDA stream semantics.

### asyncOp=true (DDP with communication-computation overlap)

```
collective():4532:
  ncclStream = ncclStreams_.at(key)   <- a DEDICATED per-device NCCL stream
  syncStream(device, ncclEvents_[key], ncclStream)  <- waits for alloc stream first
```

Timeline:
```
compute stream: [bucket N backward] -> ........... -> [bucket N+1 backward]
nccl stream:                        -> [shadow copy (pre)] -> [ncclAllReduce for bucket N]
```

The shadow copy is in the `pre` lambda which receives `ncclStream`, so it lands
on the right stream regardless of asyncOp mode.

### Restore path (Watchdog thread)

The Watchdog thread is a CPU thread. It cannot call CUDA APIs on any stream
directly without care. The restore must:
1. Enqueue the copy onto the same stream the failed AllReduce ran on.
2. Record a `shadow_restore_event_` CUDA event on that stream AFTER the copy.
3. The main thread (in the next `collective()` call) calls
   `shadow_restore_event_.block(ncclStream)` before proceeding, ensuring the
   copy is complete before the replay AllReduce begins.

The stream to use for the restore copy is recorded from the work object:
- For asyncOp=false: `work`'s stream == compute stream == `getCurrentCUDAStream`.
- For asyncOp=true: the NCCL stream that the work ran on, accessed via
  `ncclStreams_.at(getKeyFromDevice(work.device_))`.

To avoid adding a stream lookup to the Watchdog (which doesn't have easy access
to `ncclStreams_`), the stream is stored in `shadow_nccl_stream_` at the time
`allreduce_impl` runs the `pre` lambda. This is always the main thread, so no
race.

---

## Sub-Task 1 — Single Shadow Buffer (GPU memory, auto-grow)

### Status: [ ] pending

**Intent:** Pre-allocate one contiguous GPU tensor as the shadow buffer. Sized to the
largest AllReduce tensor seen so far. If a larger tensor arrives, reallocate.
This avoids repeated `at::empty_like` on the hot path.

**Stream involvement:** None. Allocation is a CPU-side operation that returns a
CUDA device tensor. The tensor's storage is pinned to the GPU device. No stream
ordering required at allocation time.

**Expected Outcomes:**
- `std::optional<at::Tensor> shadow_buf_` member in the header.
- `std::mutex shadow_buf_mutex_` protects `shadow_buf_` and `shadow_seq_`:
  the main thread writes them (allreduce_impl), the Watchdog thread reads them.
- `void ensure_shadow_buffer(const at::Tensor& t)` private helper: allocates or
  reallocates `shadow_buf_` only when `t.numel() > shadow_buf_->numel()`.
  Uses `at::empty({t.numel()}, t.options())` for contiguous storage.
- On normal training the function is a branch-not-taken no-op after first call.

**Todo List:**
1. Add to header FT member block:
   - `std::optional<at::Tensor> shadow_buf_`
   - `std::mutex shadow_buf_mutex_`
   - `uint64_t shadow_seq_{0}`
   - `at::cuda::CUDAStream shadow_nccl_stream_` (records stream used for last allreduce)
2. Add private declaration `void ensure_shadow_buffer(const at::Tensor& t)` to header.
3. Implement `ensure_shadow_buffer` in cpp:
   ```cpp
   if (!shadow_buf_.has_value() || t.numel() > shadow_buf_->numel()) {
       shadow_buf_ = at::empty({t.numel()}, t.options());
   }
   ```
4. Call `ensure_shadow_buffer(tensor)` at the top of `allreduce_impl`, before
   `collective()`.

**Relevant Context:**
- `allreduce_impl`: `ProcessGroupNCCLFT.cpp:5362`
- FT member block: `ProcessGroupNCCLFT.hpp:1075`
- `at::empty` with CUDA tensor options allocates contiguous device memory.
- `at::cuda::CUDAStream` has a default constructor (wraps stream 0), safe to
  store as a value member.

---

## Sub-Task 2 — Checkpoint: copy input into shadow buffer in the pre lambda

### Status: [ ] pending

**Intent:** Capture a clean copy of the gradient tensor immediately before the
AllReduce kernel is enqueued on the NCCL stream. Because this copy is on the
SAME stream as the subsequent `ncclAllReduce`, CUDA stream semantics guarantee
the copy completes before the AllReduce reads from `input.data_ptr()`.

**Stream involvement (critical):**

The `pre` lambda in `collective()` is called with the `ncclStream` argument:
```cpp
pre(ncclStream, work);   // ProcessGroupNCCLFT.cpp:4595
```
The copy `shadow_buf_view.copy_(tensor, /*non_blocking=*/true)` issued inside
`pre` is non-blocking from the CPU perspective but is ordered on `ncclStream`.
This means:
- The AllReduce that comes after `pre` is guaranteed to see the ORIGINAL tensor,
  not a partial result.
- If the AllReduce corrupts `tensor`, `shadow_buf_` is unaffected because it was
  written before the AllReduce kernel ran.

For asyncOp=true, the `pre` lambda is called after `syncStream()`, so the
shadow copy also respects the event fence that gates the NCCL stream on the
alloc stream.

**Expected Outcomes:**
- `shadow_buf_` always holds the input value of the most-recently-started AllReduce.
- `shadow_seq_` records the `seqCollective_` value for this checkpoint.
- `shadow_nccl_stream_` records which stream the copy/AllReduce ran on.
- All three writes happen under `shadow_buf_mutex_` held briefly by the main thread.

**Todo List:**
1. In `allreduce_impl`, pass a custom `pre` lambda to `collective()` instead of
   the default no-op. The lambda signature is:
   `[this, &tensor](at::cuda::CUDAStream& stream, c10::intrusive_ptr<Work>&) { ... }`
2. Inside the pre lambda:
   ```cpp
   std::lock_guard<std::mutex> lk(shadow_buf_mutex_);
   int64_t numel = tensor.numel();
   shadow_buf_->narrow(0, 0, numel).copy_(tensor, /*non_blocking=*/true);
   shadow_seq_  = seqCollective_;   // seqCollective_ already bumped by collective()
   shadow_nccl_stream_ = stream;    // save for Watchdog restore
   ```
3. The `copy_` with `non_blocking=true` enqueues a device-to-device copy on
   `stream`. It does NOT block the CPU.
4. Update the `allreduce_impl` call to `collective()` to pass the new `pre`
   lambda (use the 5-argument `collective()` overload that takes pre and post).

**Relevant Context:**
- `collective()` overloads: `ProcessGroupNCCLFT.cpp:5220`.
- `pre` is called at: `ProcessGroupNCCLFT.cpp:4595` (`pre(ncclStream, work)`).
- `seqCollective_` is already bumped to its final value by the time `pre` is called
  (bumped at `ProcessGroupNCCLFT.cpp:4473`).
- `tensor.narrow(0, 0, numel)` gives a view of the first `numel` elements, safe
  even if `shadow_buf_` is larger than the current tensor.

---

## Sub-Task 3 — Restore: copy shadow buffer back on fault, event-fenced

### Status: [ ] pending

**Intent:** When the Watchdog detects `work.exception()` for an AllReduce, restore
the gradient tensor from `shadow_buf_` so the input is clean for the replay.
The restore is enqueued on the NCCL stream (same stream that the failed op used),
then a CUDA event is recorded. The main thread waits on this event before the
replay AllReduce begins.

**Stream involvement (critical):**

The Watchdog is a CPU thread. It calls CUDA APIs to enqueue a copy and record
an event. This is valid because:
1. After a NCCL error, the NCCL op is complete (aborted) — the stream is idle.
2. CUDA allows multiple CPU threads to enqueue work on the same stream; CUDA
   serializes them internally.
3. The main thread is blocked at the 2PC barrier spin-wait (`while final_commit_op_ == 0`)
   and is NOT touching the stream. So there is no concurrent stream use.

Sequence on `shadow_nccl_stream_`:
```
[ncclAllReduce -- ABORTED mid-way]
  <- Watchdog enqueues: output.copy_(shadow_buf_view, non_blocking=true)
  <- Watchdog records: shadow_restore_event_.record(shadow_nccl_stream_)
  ...
[main thread unblocked from 2PC barrier]
  <- main thread: shadow_restore_event_.block(ncclStream) in next collective()
  <- execute_shadow_allreduce reads clean tensor
```

**Expected Outcomes:**
- `shadow_restore_event_` (an `at::cuda::CUDAEvent` member) is recorded after
  the restore copy.
- `shadow_restore_pending_` (a `bool` member, protected by `shadow_buf_mutex_`)
  is set to true by Watchdog and cleared by the main thread after the event wait.
- The restore only runs when `work.seq_ == shadow_seq_` to guard against
  accidentally restoring for the wrong op.

**Todo List:**
1. Add to header FT member block:
   - `at::cuda::CUDAEvent shadow_restore_event_`
   - `bool shadow_restore_pending_{false}`
2. In Watchdog FT recoverable path (`ProcessGroupNCCLFT.cpp:2469`), after
   clearing the exception, add:
   ```cpp
   {
       std::lock_guard<std::mutex> lk(pg_->shadow_buf_mutex_);
       if (pg_->shadow_buf_.has_value() &&
           work.seq_ == pg_->shadow_seq_) {
           // Restore on the stream the failed AllReduce ran on.
           at::cuda::CUDAStreamGuard g(pg_->shadow_nccl_stream_);
           auto& out = (*work.outputs_)[0];
           int64_t numel = out.numel();
           out.copy_(pg_->shadow_buf_->narrow(0, 0, numel), /*non_blocking=*/true);
           pg_->shadow_restore_event_.record(pg_->shadow_nccl_stream_);
           pg_->shadow_restore_pending_ = true;
           LOG(INFO) << pg_->logPrefix()
                     << "[NCCL-FT] Shadow restore enqueued for seq="
                     << work.seq_ << " on stream.";
       }
   }
   ```
3. At the top of `collective()` (before the barrier block), add:
   ```cpp
   if (!ft_disabled_ && shadow_restore_pending_) {
       shadow_restore_event_.block(ncclStream);
       shadow_restore_pending_ = false;
   }
   ```
   This ensures the Watchdog's copy is complete before the replay AllReduce reads
   the tensor.

**Relevant Context:**
- Watchdog FT path: `ProcessGroupNCCLFT.cpp:2469`.
- `work.outputs_`: `shared_ptr<vector<at::Tensor>>` — element 0 is the AllReduce
  output (which equals the input for AllReduce in-place).
- `at::cuda::CUDAStreamGuard` switches the current CUDA stream for the scope,
  making `copy_` enqueue on the correct stream.
- `at::cuda::CUDAEvent::record(stream)` records a marker on the stream.
- `at::cuda::CUDAEvent::block(stream)` inserts a wait on `stream` for the event.

---

## Sub-Task 4 — Cross-rank rollback consensus via TCPStore

### Status: DONE

**Intent:** All ranks must agree that a rollback happened at a specific `seq` number
before any rank starts the replay AllReduce. Without this, a rank that hasn't
yet restored its tensor could start the replay while another rank is still in the
middle of restoration — causing the AllReduce to read different data on different
ranks and produce a mathematically wrong result.

The existing 2PC protocol (PROPOSE / ACK / COMMIT) already synchronizes all ranks
on the fault boundary OP. We extend it by having each rank write its `shadow_seq`
to TCPStore in Phase 1. Rank 0 verifies all ranks agree before writing COMMIT.

**Note on "mismatch" handling (prototype policy):**
In theory, all ranks should have the same `shadow_seq` because DDP AllReduce is
synchronous — every rank checkpoints the same op. A mismatch would indicate a
rare timing edge case. The prototype logs a WARNING and proceeds using the
minimum (most conservative) `shadow_seq`. This is safe because using an older
checkpoint is always correct (it just means the op had already succeeded on
some ranks and we redo it, which is fine for the idempotent AllReduce+SGD path).

**Expected Outcomes:**
- Each rank writes `NCCL_FT_SHADOW_SEQ_<rank>_<target_op> = <shadow_seq>`
  alongside its ACK in Phase 1.
- Rank 0 reads all `NCCL_FT_SHADOW_SEQ_*` keys, verifies equality (logs on
  mismatch), stores the agreed value.
- After COMMIT, all ranks have the same understanding of which op to replay.
- `committed_shadow_seq_` (a new `std::atomic<uint64_t>` member) is set by the
  negotiator thread after COMMIT lands; the main thread reads it at replay time.

**Todo List:**
1. Add `std::atomic<uint64_t> pending_shadow_seq_{0}` and
   `std::atomic<uint64_t> committed_shadow_seq_{0}` to the header.
2. In `trigger_fault_proposal`, also store `shadow_seq_` into
   `pending_shadow_seq_` (read under `shadow_buf_mutex_`, stored atomically).
3. In the negotiator relay-write section, extend the PROPOSE string:
   `"PROPOSE:<target_op>:<node_id>:<dev_idx>:<shadow_seq>"`
   (parse `pending_shadow_seq_`).
4. In the negotiator PROPOSE-handler, write the shadow seq key alongside ACK:
   `NCCL_FT_SHADOW_SEQ_<rank>_<target_op> = <shadow_seq_from_proposal>`
5. In rank 0's ACK-collection loop, also read all
   `NCCL_FT_SHADOW_SEQ_<r>_<target_op>` keys; log WARNING if any differ;
   take `min` as the agreed value; write agreed value to
   `NCCL_FT_SHADOW_SEQ_AGREED_<target_op>` before writing COMMIT.
6. After each rank reads the COMMIT key, also read
   `NCCL_FT_SHADOW_SEQ_AGREED_<target_op>` and store into `committed_shadow_seq_`.

**Relevant Context:**
- Negotiator relay-write: `ProcessGroupNCCLFT.cpp:4241`.
- Negotiator PROPOSE-handler: `ProcessGroupNCCLFT.cpp:4278`.
- Rank 0 ACK-collection loop: `ProcessGroupNCCLFT.cpp:4321`.
- `trigger_fault_proposal`: `ProcessGroupNCCLFT.cpp:4209`.

---

## Sub-Task 5 — Re-execute failed op after rollback (replay path)

### Status: [ ] pending

**Intent:** After the 2PC barrier completes and `rebuild_shadow_ping_pong_topology()`
has run, the main thread must re-execute the AllReduce that was aborted. The
replay uses the clean (restored) tensor as input and writes the correct result
into the output tensor. The DDP bucket reducer's `work.wait()` call must
eventually see a completed work object.

**Stream involvement:**

The replay calls `execute_shadow_allreduce(inputs[0], outputs[0], ncclStream, ...)`.
The `ncclStream` at this point is the stream of the CURRENT `collective()` call
(which is the re-issued AllReduce after the barrier). Before the replay runs,
`shadow_restore_event_.block(ncclStream)` (Sub-Task 3, step 3) has already been
called, so the restore copy is guaranteed complete on this stream.

The replay AllReduce uses `local_nvlink_comm_` (NVLink, intra-node) and
`proxy_global_comm_` (IB, inter-node but on healthy NICs only). Both are
separate from the failed comm — no conflict with the stream that the failed
AllReduce used.

**Expected Outcomes:**
- After the barrier + rebuild, if `shadow_restore_pending_` was set (meaning
  this is the exact op that was rolled back), the replay runs immediately.
- `shadow_replay_pending_` (a `bool` member) marks that a replay is due.
- The replay uses the same `ncclStream` as the current collective call.
- `seqCollective_` is NOT double-incremented; the replay does not go through
  the top of `collective()` again — it is executed inline at the barrier block.
- Log messages clearly indicate "REPLAY after rollback for seq=N".

**Todo List:**
1. Add `bool shadow_replay_pending_{false}` member to the header.
2. In the Watchdog restore path (Sub-Task 3), also set
   `pg_->shadow_replay_pending_ = true` (inside the same mutex lock).
3. In `collective()`, after the barrier block (after `proxy_comm_ready_` is
   confirmed), add:
   ```cpp
   if (!ft_disabled_ && C10_UNLIKELY(shadow_replay_pending_) &&
       opType == OpType::ALLREDUCE &&
       proxy_comm_ready_.load(std::memory_order_acquire)) {
       LOG(INFO) << logPrefix()
                 << "[NCCL-FT] REPLAY after rollback for seq="
                 << committed_shadow_seq_.load();
       execute_shadow_allreduce(
           inputs[0], outputs[0], ncclStream, current_shadow_reduce_op_);
       shadow_replay_pending_ = false;
       // Skip the normal fn() call below.
       goto ft_post;
   }
   ```
4. Add label `ft_post:` just before the `post(ncclStream, work)` call so the
   goto lands correctly. Alternatively, restructure with a boolean flag
   `bool ran_shadow_replay = false` and wrap the normal `fn()` call in
   `if (!ran_shadow_replay)`.
   Using a flag (not goto) is cleaner:
   ```cpp
   bool ran_shadow_replay = false;
   if (shadow_replay_pending_ && is_degraded_ && opType == ALLREDUCE
       && proxy_comm_ready_) {
       execute_shadow_allreduce(...);
       shadow_replay_pending_ = false;
       ran_shadow_replay = true;
   }
   if (!ran_shadow_replay) {
       // normal fn() call (existing degraded + native paths)
   }
   ```

**Relevant Context:**
- Barrier block in `collective()`: `ProcessGroupNCCLFT.cpp:4597`.
- `execute_shadow_allreduce`: `ProcessGroupNCCLFT.cpp:4083`.
- `rebuild_shadow_ping_pong_topology` sets `proxy_comm_ready_ = true` at the end.
- The `post` lambda: `ProcessGroupNCCLFT.cpp:4693` — must always run regardless
  of which path executed the AllReduce.

---

## Open Questions / Known Limitations

1. **asyncOp=true multi-stream case**: For asyncOp=true, `shadow_nccl_stream_`
   is the NCCL stream (not the compute stream). The Watchdog restore enqueues on
   the NCCL stream. The main thread's `shadow_restore_event_.block(ncclStream)`
   also uses the NCCL stream in the next collective. This is consistent. However,
   if the main thread's next collective call uses a DIFFERENT NCCL stream (e.g.
   a different device key), the block would be on the wrong stream. In DDP with
   one device per rank, there is only one NCCL stream, so this does not occur.

2. **Thread safety of `shadow_restore_pending_`**: Written by Watchdog, read by
   main thread. Protected by `shadow_buf_mutex_`. The main thread checks this at
   the top of `collective()` (before the barrier) and after the barrier. Both
   reads are under the mutex. The Watchdog write is also under the mutex (Step 2
   of Sub-Task 3).

3. **asyncOp=false replay stream**: For asyncOp=false, `ncclStream` is
   `getCurrentCUDAStream()` at the time of the REPLAY collective call. This is
   the compute stream of the main thread. The event block ensures the Watchdog's
   restore copy (which ran on the earlier compute stream call) finishes first.
   Since it's the same logical compute stream, this is always correct.

4. **Multiple in-flight AllReduces (asyncOp=true overlap)**: If two AllReduces
   are in flight and BOTH fail, `shadow_buf_` only holds the checkpoint of the
   LAST one started. The first one's checkpoint has been overwritten. This is
   the accepted limitation for the prototype. In practice, NIC failure causes
   both ops to fail with the same error, and only one 2PC round runs.

5. **Second fault during replay**: The replay uses `local_nvlink_comm_` (NVLink)
   and `proxy_global_comm_` (healthy IB NICs only). A second IB fault on a
   different NIC would trigger a new fault callback, a new 2PC, and a new rebuild.
   The in-progress replay would not be interrupted (NVLink is unaffected).

---

## Implementation Order

```
Sub-Task 1 (allocate buffer)
    |
Sub-Task 2 (checkpoint in pre lambda)
    |
Sub-Task 3 (restore in Watchdog + event fence)
    |
Sub-Task 4 (TCPStore shadow_seq consensus)   <- can be done in parallel with 3
    |
Sub-Task 5 (replay after barrier)
```

All five sub-tasks must be complete before end-to-end testing.
