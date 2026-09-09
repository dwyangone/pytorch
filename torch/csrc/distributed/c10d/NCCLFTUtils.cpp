#include <torch/csrc/distributed/c10d/NCCLFTUtils.hpp>
// Include the original NCCLUtils to reuse helper functions (e.g. getNcclVersion, getNcclErrorDetailStr).
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>

#ifdef USE_C10D_NCCL
#include <fmt/format.h>
#include <mutex>
#include <thread>
#include <vector>

namespace c10d {

NCCLFTComm::NCCLFTComm(ncclComm_t ncclComm) : ncclComm_(ncclComm) {}

NCCLFTComm::~NCCLFTComm() noexcept {
  // (kwen2501) Making CUDA/NCCL calls in this destructor can hit CUDA driver
  // shutdown error if CUDA context has exited first. Thus, we are not
  // destroying or aborting NCCL communicators here. We just detect and warn
  // about the risk of memory leak. Normally, a user would have called
  // `destroy_process_group` or `abort_process_group`, and such risk would be
  // avoided.
  LockType lock(mutex_);
  if (ncclComm_ && initialized_ && !aborted_) {
    TORCH_WARN_ONCE(
        "WARNING: NCCL FT communicator hasn't been destroyed. This may cause "
        "memory leaks. To avoid the risk, you can call `destroy_process_group` "
        "during normal exit or `_abort_process_group` when handling failures.")
  }
}

// NOLINTNEXTLINE(*-noexcept-move-*)
NCCLFTComm::NCCLFTComm(NCCLFTComm&& other) {
  // Using other's lock, as it reads other's states
  // Can not use this.mutex_, as this object is being constructed.
  LockType lock(other.mutex_);
  std::swap(ncclComm_, other.ncclComm_);
  std::swap(aborted_, other.aborted_);
  std::swap(ncclAsyncErr_, other.ncclAsyncErr_);
  std::swap(initialized_, other.initialized_);
  std::swap(nonBlocking_, other.nonBlocking_);
  std::swap(deviceIndex_, other.deviceIndex_);
}

void NCCLFTComm::setUniqueHash(ncclUniqueId ncclId) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&ncclId);

  fmt::memory_buffer buf;
  buf.reserve(NCCL_UNIQUE_ID_BYTES * 2); // 2 hex chars per byte
  for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
    fmt::format_to(
        std::back_inserter(buf), "{:02x}", static_cast<int>(bytes[i]));
  }
  this->uniqueHash_ = fmt::to_string(buf);
}

void NCCLFTComm::setUniqueHash(std::string hash) {
  this->uniqueHash_ = std::move(hash);
}

std::string NCCLFTComm::getUniqueHash() {
  return uniqueHash_;
}

std::shared_ptr<NCCLFTComm> NCCLFTComm::create(
    int numRanks,
    int rank,
    ncclUniqueId commId,
    at::DeviceIndex deviceIndex) {
  at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex);
  auto comm = std::make_shared<NCCLFTComm>();
  C10D_NCCL_FT_CHECK(
      ncclCommInitRank(&(comm->ncclComm_), numRanks, commId, rank),
      std::nullopt);
  comm->setUniqueHash(commId);
  comm->rank_ = rank;
  comm->deviceIndex_ = deviceIndex;
  comm->initialized_ = true;
  // Old style comm is always blocking.
  comm->nonBlocking_ = false;
  return comm;
}

#ifdef NCCL_HAS_CONFIG
std::shared_ptr<NCCLFTComm> NCCLFTComm::create(
    int numRanks,
    int rank,
    ncclUniqueId commId,
    at::DeviceIndex deviceIndex,
    ncclConfig_t& config) {
  at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex);
  auto comm = std::make_shared<NCCLFTComm>();
  comm->nonBlocking_ = config.blocking == 0;
  LOG(INFO) << "Rank " << rank << ": creating NCCL FT communicator with mode: "
            << (comm->nonBlocking_ ? "nonblocking" : "blocking");
  C10D_NCCL_FT_CHECK_NONBLOCKING(
      ncclCommInitRankConfig(
          &(comm->ncclComm_), numRanks, commId, rank, &config),
      std::nullopt);
  comm->setUniqueHash(commId);
  comm->rank_ = rank;
  comm->deviceIndex_ = deviceIndex;
  // Under blocking mode, comm is initialized immediately after NCCL init
  // returns; Under nonblocking mode, we check whether comm is initialized the
  // *next* time ncclComm_ is accessed.
  comm->initialized_ = !comm->nonBlocking_;
  return comm;
}
#ifdef NCCL_HAS_INIT_RANK_SCALABLE
std::shared_ptr<NCCLFTComm> NCCLFTComm::create_scalable(
    int numRanks,
    int rank,
    std::vector<ncclUniqueId>& commIds,
    at::DeviceIndex deviceIndex,
    ncclConfig_t& config) {
  at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex);
  auto comm = std::make_shared<NCCLFTComm>();
  comm->nonBlocking_ = config.blocking == 0;
  LOG(INFO) << "Rank " << rank << ": creating NCCL FT communicator with mode: "
            << (comm->nonBlocking_ ? "nonblocking" : "blocking")
            << " with scalable init.";
  C10D_NCCL_FT_CHECK_NONBLOCKING(
      ncclCommInitRankScalable(
          &(comm->ncclComm_),
          numRanks,
          rank,
          commIds.size(),
          commIds.data(),
          &config),
      std::nullopt);
  // Only the first ncclUniqueId will be used to create the
  // communicator hash id, which is used to identify the communicator
  // in the log file and in the replay tool.
  comm->setUniqueHash(commIds[0]);
  comm->rank_ = rank;
  comm->deviceIndex_ = deviceIndex;
  comm->initialized_ = !comm->nonBlocking_;
  return comm;
}
#endif // NCCL_HAS_INIT_RANK_SCALABLE
#endif // NCCL_HAS_CONFIG

ncclComm_t NCCLFTComm::getNcclComm() {
  LockType lock(mutex_);
  if (aborted_) {
    auto commFailureMsg = commFailureReason_ != std::nullopt
        ? c10::str(" Original reason for failure was: ", *commFailureReason_)
        : "";
    TORCH_CHECK_WITH(
        NCCLFaultToleranceError,
        false,
        c10::str(
            "NCCL FT communicator was aborted on rank ",
            rank_,
            ". ",
            commFailureMsg));
  }
  // In non-blocking mode, ensure comm is ready.
  if (nonBlocking_) {
    // Wait with long interval if communicator is being initialized.
    bool longInterval = !initialized_;
    waitReady(longInterval);
    // ncclComm_ should be initialized by now
  }
  if (!initialized_) {
    // TODO: see if we can consolidate other `initialized_` flipping here.
    // Maintaining it elsewhere is some work.
    initialized_ = true;
    LOG(INFO) << "Rank " << rank_ << ": NCCL FT communicator " << repr()
              << " is initialized.";
  }
  return ncclComm_;
}

at::DeviceIndex NCCLFTComm::getDeviceIndex() {
  return deviceIndex_;
}

// Wait for the communicator to be ready. This is a blocking function.
// Arguments:
//   longInterval: if true, wait with sleep of an interval; otherwise, wait
//   with `sched_yield` which is faster (but acquires CPU more frequently).
void NCCLFTComm::waitReady(bool longInterval) {
  LockType lock(mutex_);
  if (aborted_)
    return;
  // If timeout is reached, throw an exception.
  // (RohitRathore1) Note: We already hold the mutex_ lock here, so the direct
  // call to ncclCommGetAsyncError is safe from concurrent access.
  ncclResult_t result = ncclInProgress;
  auto startTimepoint = std::chrono::steady_clock::now();
  auto timeout = nccl_nonblocking_timeout();
  while (result == ncclInProgress) {
    auto currentTime = std::chrono::steady_clock::now();
    auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           currentTime - startTimepoint)
                           .count();
    if (timeElapsed > timeout) {
      std::string err = "NCCL timeout in: " + std::string(__FILE__) + ":" +
          std::to_string(__LINE__);
      TORCH_CHECK_WITH(NCCLFaultToleranceError, false, err);
    }
    if (longInterval) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kCommInitBusyWaitMillis));
    } else {
      sched_yield();
    }
    ncclCommGetAsyncError(ncclComm_, &result);
  }
  if (result != ncclSuccess) {
    std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +
        std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) +
        "\n" + getNcclErrorDetailStr(result, std::nullopt);
    TORCH_CHECK_WITH(NCCLFaultToleranceError, false, err);
  }
}

std::optional<std::string> NCCLFTComm::getNcclCommFailureReason() const {
  LockType lock(mutex_);
  return commFailureReason_;
}

#if defined(NCCL_HAS_COMM_SPLIT)
std::shared_ptr<NCCLFTComm> NCCLFTComm::split(
    NCCLFTComm* source,
    int color_id,
    int rank,
    ncclConfig_t& config) {
  TORCH_CHECK(
      color_id >= NCCL_SPLIT_NOCOLOR,
      "Color must be a non-negative value or NCCL_SPLIT_NOCOLOR (-1)"
      ", but got ",
      color_id);
  LOG(INFO) << "Rank " << source->rank_ << ": split from parent FT comm "
            << source->repr() << " with color_id " << color_id << " and rank "
            << rank;
  at::cuda::OptionalCUDAGuard gpuGuard(source->deviceIndex_);
  auto comm = std::make_shared<NCCLFTComm>();
  // This call will block until the source communicator is initialized
  auto sourceComm = source->getNcclComm();
#ifndef NCCL_HAS_COMM_NONBLOCKING
  C10D_NCCL_FT_CHECK(
      ncclCommSplit(sourceComm, color_id, rank, &(comm->ncclComm_), &config),
      std::nullopt);
#else
  // After calling ncclCommSplit in non-blocking mode, we should wait for the
  // source communicator to be out of ncclInProgress state.
  C10D_NCCL_FT_CHECK_TIMEOUT_SLEEP(
      ncclCommSplit(sourceComm, color_id, rank, &(comm->ncclComm_), &config),
      source, // wait on parent comm
      std::nullopt);
  if (color_id >= 0) {
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = nccl_nonblocking_timeout();
    while (!comm->ncclComm_) {
      C10D_FT_CHECK_TIMEOUT(startTime, timeout);
      C10D_SCHED_SLEEP();
    }
  }
#endif
  ++source->ncclCommSplitCounter_;
  comm->rank_ = rank;
  // Child comm should be on the same device as parent comm
  comm->deviceIndex_ = source->deviceIndex_;
  comm->nonBlocking_ = config.blocking == 0;
  comm->setUniqueHash(
      source->getUniqueHash() + ":" +
      std::to_string(source->ncclCommSplitCounter_));
  LOG(INFO) << "Rank " << source->rank_ << ": created child FT comm "
            << comm->repr() << " with color_id " << color_id;
  return comm;
}
#endif

#ifdef NCCL_HAS_COMM_SHRINK
std::shared_ptr<NCCLFTComm> NCCLFTComm::shrink(
    NCCLFTComm* source,
    std::vector<int>& ranks_to_exclude,
    ncclConfig_t* config,
    int shrinkFlags) {
  // Preconditions are validated in ProcessGroupNCCL::shrink

  LOG(INFO) << "Rank " << source->rank_ << ": shrinking FT comm " << source->repr()
            << " excluding " << ranks_to_exclude.size() << " ranks";

  at::cuda::OptionalCUDAGuard gpuGuard(source->deviceIndex_);
  auto comm = std::make_shared<NCCLFTComm>();

  // This call will block until the source communicator is initialized
  auto sourceComm = source->getNcclComm();

  C10D_NCCL_FT_CHECK_NONBLOCKING(
      ncclCommShrink(
          sourceComm,
          ranks_to_exclude.data(),
          ranks_to_exclude.size(),
          reinterpret_cast<ncclComm_t*>(&(comm->ncclComm_)),
          config,
          shrinkFlags),
      source->getNcclCommFailureReason());

  // Wait for the child communicator to be ready
  source->waitReady(true);
  comm->initialized_ = true;

  // NCCL automatically assigns rank during shrink - query it efficiently
  int assigned_rank;
  try {
    C10D_NCCL_FT_CHECK(
        ncclCommUserRank(comm->ncclComm_, &assigned_rank), std::nullopt);
    comm->rank_ = assigned_rank;
  } catch (const std::exception& e) {
    // Fallback: if ncclCommUserRank fails, we can't determine the rank
    LOG(ERROR) << "Failed to query NCCL-assigned rank: " << e.what();
    throw;
  }

  // Child comm should be on the same device as parent comm
  comm->deviceIndex_ = source->deviceIndex_;
  if (config != nullptr) {
    comm->nonBlocking_ = config->blocking == 0;
  } else {
    // Inherit parent behavior if no config provided
    comm->nonBlocking_ = source->nonBlocking_;
  }

  LOG(INFO) << "Rank " << source->rank_ << ": created shrunken FT comm "
            << comm->repr() << " with NCCL-assigned rank " << assigned_rank;

  return comm;
}
#endif

void NCCLFTComm::finalize() {
  LockType lock(mutex_);
  if (aborted_) {
    LOG(INFO) << "Rank " << rank_
              << ": NCCL FT communicator already Invalidated. Skip finalize.";
    return;
  }
  at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex_);
  auto comm = getNcclComm();
  C10D_NCCL_FT_CHECK_NONBLOCKING(ncclCommFinalize(comm), std::nullopt);
}

void NCCLFTComm::destroy() {
  LockType lock(mutex_);
  if (aborted_) {
    LOG(INFO) << "Rank " << rank_
              << ": NCCL FT communicator already Invalidated. Skip destroy.";
    return;
  }
  at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex_);
  auto comm = getNcclComm();
  C10D_NCCL_FT_CHECK(ncclCommDestroy(comm), std::nullopt);
  // Poison future getNcclComm
  aborted_ = true;
}

void NCCLFTComm::abort(std::optional<std::string> commFailureReason) {
  LockType lock(mutex_);
  at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex_);
#ifdef ENABLE_NCCL_ERROR_CHECKING
  if (aborted_ && !initialized_) {
    // Should not abort twice.
    return;
  }

#ifdef NCCL_HAS_COMM_REGISTER
  // Deregister all registered segments before aborting.
  for (auto& it : registeredSegmentHandles_) {
    void* handle = it.second.first;
    bool is_window = it.second.second;
    if (is_window) {
#ifdef NCCL_HAS_COMM_WINDOW_REGISTER
      C10D_NCCL_CHECK(
          ::ncclCommWindowDeregister(ncclComm_, (ncclWindow_t)handle),
          c10::str(
              "Failed to window deregister segment handle ",
              handle,
              " on ncclComm_ ",
              ncclComm_));
#endif
    } else {
    C10D_NCCL_FT_CHECK(
        ::ncclCommDeregister(ncclComm_, handle),
        c10::str(
            "Failed to deregister segment handle ",
            handle,
            " on ncclComm_ ",
            ncclComm_));
    }
  }
  registeredSegmentHandles_.clear();
#endif

  // Set true failure reason if provided by ProcessGroupNCCL (e.g. work
  // timeout)
  commFailureReason_ = commFailureReason;
  LOG(INFO) << "Aborting ncclComm_ " << ncclComm_ << " with reason: "
            << (commFailureReason ? *commFailureReason
                                  : "No abort reason provided.");
#ifndef NCCL_HAS_COMM_NONBLOCKING
  C10D_NCCL_FT_CHECK(::ncclCommAbort(ncclComm_), commFailureReason_);
#else
  // Note: We already hold the mutex_ lock here, so the direct call to
  // ncclCommGetAsyncError is safe from concurrent access.
  ncclResult_t result = ::ncclCommAbort(ncclComm_);
  auto startTimepoint = std::chrono::steady_clock::now();
  auto timeout = nccl_nonblocking_timeout();
  while (result == ncclInProgress) {
    auto currentTime = std::chrono::steady_clock::now();
    auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           currentTime - startTimepoint)
                           .count();
    if (timeElapsed > timeout) {
      std::string err = "NCCL timeout in: " + std::string(__FILE__) + ":" +
          std::to_string(__LINE__);
      TORCH_CHECK_WITH(NCCLFaultToleranceError, false, err);
    }
    sched_yield();
    ncclCommGetAsyncError(ncclComm_, &result);
  }
  if (result != ncclSuccess) {
    std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +
        std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) +
        "\n" + getNcclErrorDetailStr(result, commFailureReason_);
    TORCH_CHECK_WITH(NCCLFaultToleranceError, false, err);
  }
#endif
  aborted_ = true;
  ncclComm_ = nullptr;

  // Set an appropriate error so that we avoid using the communicator.
  if (ncclAsyncErr_ == ncclSuccess) {
    ncclAsyncErr_ = ncclSystemError;
  }
#else
  // This is a NOOP, if error checks are disabled.
  return;
#endif
}

bool NCCLFTComm::isInitialized() const {
  LockType lock(mutex_);
  return initialized_;
}

bool NCCLFTComm::isAborted() const {
  LockType lock(mutex_);
  return aborted_;
}

uint64_t NCCLFTComm::getCommSplitCounter() const {
  return ncclCommSplitCounter_;
}

ncclResult_t NCCLFTComm::checkForNcclError() {
  LockType lock(mutex_);
#ifdef ENABLE_NCCL_ERROR_CHECKING
  if (ncclAsyncErr_ != ncclSuccess) {
    return ncclAsyncErr_;
  }
  C10D_NCCL_FT_CHECK(
      ncclCommGetAsyncError(ncclComm_, &ncclAsyncErr_), commFailureReason_);
  return ncclAsyncErr_;
#else
  // Always return success, if error checks are disabled.
  return ncclSuccess;
#endif
}

ncclResult_t NCCLFTComm::getAsyncError(ncclResult_t* asyncError) {
  LockType lock(mutex_);
  return ncclCommGetAsyncError(ncclComm_, asyncError);
}

ncclResult_t NCCLFTComm::registerSegment(
    void* ptr,
    size_t size,
    bool errorOnRereg, /*=true*/
    bool window /*=false*/) {
  LockType lock(mutex_);
#ifdef NCCL_HAS_COMM_REGISTER
  // We register only segments from cache allocator
  // which are guaranteed to be with disjoint addr ranges. Thus, a ptr always
  // maps to a unique handle and should not be registered before the current
  // ptr is deregistered and freed.
  if (registeredSegmentHandles_.count(ptr) > 0) {
    TORCH_CHECK(
        !errorOnRereg,
        "Segment with ptr ",
        ptr,
        " has already been registered on ncclComm_ ",
        ncclComm_);
    // Skip below
    return ncclSuccess;
  }

  void* handle = nullptr;
  // Use getNcclComm to make sure comm is ready before calling nccl APIs
  auto comm = getNcclComm();
#ifdef NCCL_HAS_COMM_WINDOW_REGISTER
  if (window) {
    C10D_NCCL_FT_CHECK(
        ncclCommWindowRegister(
            comm, ptr, size, (ncclWindow_t*)&handle, NCCL_WIN_COLL_SYMMETRIC),
        c10::str(
            "Failed to window register segment with ptr ",
            ptr,
            ", size ",
            size,
            " on ncclComm_ ",
            comm));
  } else {
    C10D_NCCL_FT_CHECK(
        ncclCommRegister(comm, ptr, size, &handle),
        c10::str(
            "Failed to register segment with ptr ",
            ptr,
            ", size ",
            size,
            " on ncclComm_ ",
            comm));
  }
#else
  C10D_NCCL_FT_CHECK(
      ncclCommRegister(comm, ptr, size, &handle),
      c10::str(
          "Failed to register segment with ptr ",
          ptr,
          ", size ",
          size,
          " on ncclComm_ ",
          comm));
#endif
  registeredSegmentHandles_[ptr] = {handle, window};
  return ncclSuccess;
#else
  return ncclInvalidUsage;
#endif
}

ncclResult_t NCCLFTComm::deregisterSegment(void* ptr, bool window /*false*/) {
  LockType lock(mutex_);
#ifdef NCCL_HAS_COMM_REGISTER
  TORCH_CHECK(
      registeredSegmentHandles_.count(ptr) == 1,
      "Segment with ptr ",
      ptr,
      " is not registered on ncclComm_ ",
      ncclComm_);

  void* handle = registeredSegmentHandles_[ptr].first;
  // Use getNcclComm to make sure comm is ready before calling nccl APIs
  auto comm = getNcclComm();
#ifdef NCCL_HAS_COMM_WINDOW_REGISTER
  if (window) {
    C10D_NCCL_FT_CHECK(
        ncclCommWindowDeregister(comm, (ncclWindow_t)handle),
        c10::str(
            "Failed to window deregister segment handle ",
            handle,
            ", with ptr ",
            ptr,
            " on ncclComm_ ",
            comm));
  } else {
    C10D_NCCL_FT_CHECK(
        ncclCommDeregister(comm, handle),
        c10::str(
            "Failed to deregister segment handle ",
            handle,
            ", with ptr ",
            ptr,
            " on ncclComm_ ",
            comm));
  }
#else
  C10D_NCCL_FT_CHECK(
      ncclCommDeregister(comm, handle),
      c10::str(
          "Failed to deregister segment handle ",
          handle,
          ", with ptr ",
          ptr,
          " on ncclComm_ ",
          comm));
#endif
  registeredSegmentHandles_.erase(ptr);
  return ncclSuccess;
#else
  return ncclInvalidUsage;
#endif
}

std::string NCCLFTComm::repr() const {
  return c10::str((void*)ncclComm_);
}

void NCCLFTComm::suspend() {
#ifdef NCCL_HAS_COMM_OFFLOAD
  LockType lock(mutex_);
  at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex_);
  auto comm = getNcclComm();
  C10D_NCCL_CHECK(ncclCommSuspend(comm, NCCL_SUSPEND_MEM), std::nullopt);
#else
  TORCH_CHECK(false, "suspend() requires NCCL 2.29.7 or later");
#endif
}

void NCCLFTComm::resume() {
#ifdef NCCL_HAS_COMM_OFFLOAD
  LockType lock(mutex_);
  at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex_);
  auto comm = getNcclComm();
  C10D_NCCL_CHECK(ncclCommResume(comm), std::nullopt);
#else
  TORCH_CHECK(false, "resume() requires NCCL 2.29.7 or later");
#endif
}

std::unordered_map<std::string, uint64_t> NCCLFTComm::getMemoryStats() {
#ifdef NCCL_HAS_COMM_OFFLOAD
  LockType lock(mutex_);
  at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex_);
  auto comm = getNcclComm();
  uint64_t suspend, suspended, persist, total;
  C10D_NCCL_CHECK(
      ncclCommMemStats(comm, ncclStatGpuMemSuspend, &suspend), std::nullopt);
  C10D_NCCL_CHECK(
      ncclCommMemStats(comm, ncclStatGpuMemSuspended, &suspended),
      std::nullopt);
  C10D_NCCL_CHECK(
      ncclCommMemStats(comm, ncclStatGpuMemPersist, &persist), std::nullopt);
  C10D_NCCL_CHECK(
      ncclCommMemStats(comm, ncclStatGpuMemTotal, &total), std::nullopt);
  return {
      {"suspend", suspend},
      {"suspended", suspended},
      {"persist", persist},
      {"total", total},
  };
#else
  TORCH_CHECK(false, "getMemoryStats() requires NCCL 2.29.7 or later");
#endif
}

#if (defined(IS_NCCLX) || defined(USE_ROCM)) && defined(NCCL_COMM_DUMP)
std::unordered_map<std::string, std::string> NCCLFTComm::ncclCommDump() {
  std::unordered_map<std::string, std::string> dump;
  if (isAborted()) {
    LOG(INFO) << "Communicator was aborted before trying to dump its state.";
    return dump;
  }
  C10D_NCCL_FT_CHECK(::ncclCommDump(ncclComm_, dump), std::nullopt);
  return dump;
}
#endif

} // namespace c10d

#endif // USE_C10D_NCCL