#pragma once

#ifdef USE_C10D_NCCL

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <atomic>
#include <chrono>
#include <deque>
#include <future>
#include <iostream>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/NCCLFTUtils.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/cuda/CUDAEventCache.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/intra_node_comm.hpp>

#include <ATen/DynamicLibrary.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/MemPool.h>
#include <c10/core/Stream.h>
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <torch/custom_class.h>

namespace c10d {

// Control broadcasting of NCCL uniqueId
static std::vector<std::string> TORCH_NCCLFT_BCAST_UNIQUEID = {
    "TORCH_NCCLFT_BCAST_UNIQUEID"};

// Control EagerInit P2P serialization warning
static std::vector<std::string>
    TORCH_NCCLFT_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING = {
        "TORCH_NCCLFT_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING"};

// Control whether to always use high priority streams
static std::vector<std::string> TORCH_NCCLFT_HIGH_PRIORITY = {
    "TORCH_NCCLFT_HIGH_PRIORITY"};

// Control whether or not wait() is blocking or non-blocking.
static std::vector<std::string> TORCH_NCCLFT_BLOCKING_WAIT = {
    "TORCH_NCCLFT_BLOCKING_WAIT",
    "NCCLFT_BLOCKING_WAIT"};

// TODO: We want to eventually remove this variable and make users to use
// the default value (3 - SkipCleanUpFT).
// Control whether or not we perform Async Error Handling with NCCL.
static std::vector<std::string> TORCH_NCCLFT_ASYNC_ERROR_HANDLING = {
    "TORCH_NCCLFT_ASYNC_ERROR_HANDLING",
    "NCCLFT_ASYNC_ERROR_HANDLING"};

// Control whether dumping debug info on watchdog
// timeout is enabled. This variable must be set together with
// TORCH_NCCLFT_ENABLE_MONITORING=1 and TORCH_NCCLFT_TRACE_BUFFER_SIZE > 0.
static std::vector<std::string> TORCH_NCCLFT_DUMP_ON_TIMEOUT = {
    "TORCH_NCCLFT_DUMP_ON_TIMEOUT"};

// Control whether to propagate NCCL errors to all ranks through TCPStore.
static std::vector<std::string> TORCH_NCCLFT_PROPAGATE_ERROR = {
    "TORCH_NCCLFT_PROPAGATE_ERROR"};

// Control whether Desync Debug is enabled. This variable must be set
// together with TORCH_NCCLFT_ASYNC_ERROR_HANDLING.
static std::vector<std::string> TORCH_NCCLFT_DESYNC_DEBUG = {
    "TORCH_NCCLFT_DESYNC_DEBUG",
    "NCCLFT_DESYNC_DEBUG"};

// Enable recording start-events for all ProcessGroupNCCLFT collectives, and
// compute accurate collective timing per-collective. (Note: end-events are
// recorded by default. Turn on this flag can increase chances of a watchdog
// hang due to performing a CUDA event query which eventually calls
// cudaEventElapsedTime() API.
static std::vector<std::string> TORCH_NCCLFT_ENABLE_TIMING = {
    "TORCH_NCCLFT_ENABLE_TIMING",
    "NCCLFT_ENABLE_TIMING"};

// Enable monitoring thread which aborts the process when the ProcessGroupNCCLFT
// Watchdog thread gets stuck and no heartbeat is detected after
// TORCH_NCCLFT_HEARTBEAT_TIMEOUT_SEC. This can happen due to calling CUDA/NCCL
// APIs that may hang. It is Useful to prevent jobs being stuck for a prolonged
// time than necessary tying up cluster resources.
static std::vector<std::string> TORCH_NCCLFT_ENABLE_MONITORING = {
    "TORCH_NCCLFT_ENABLE_MONITORING"};

// Control the watchdog heartbeat timeout period after which the monitoring
// thread will abort the process.
static std::vector<std::string> TORCH_NCCLFT_HEARTBEAT_TIMEOUT_SEC = {
    "TORCH_NCCLFT_HEARTBEAT_TIMEOUT_SEC"};

// Whether to rethrow CUDA Errors in the watchdog (default true)
static std::vector<std::string> TORCH_NCCLFT_RETHROW_CUDA_ERRORS = {
    "TORCH_NCCLFT_RETHROW_CUDA_ERRORS"};

// The maximum number of events we store in the flight recorder's ring buffer.
// (One event could be the start or end of a collective, for example).
static std::vector<std::string> TORCH_NCCLFT_TRACE_BUFFER_SIZE = {
    "TORCH_NCCLFT_TRACE_BUFFER_SIZE"};

// Control how much extra time we will wait for dumping the debugging info
// before we exit and throws timeout exception.
static std::vector<std::string> TORCH_NCCLFT_WAIT_TIMEOUT_DUMP_MILSEC = {
    "TORCH_NCCLFT_WAIT_TIMEOUT_DUMP_MILSEC"};

// Control the interval inside the monitoring thread to check the coordinated
// signal from other ranks, e.g. to dump the debugging information.
static std::vector<std::string> TORCH_NCCLFT_COORD_CHECK_MILSEC = {
    "TORCH_NCCLFT_COORD_CHECK_MILSEC"};

// Whether to log C++ stack traces on unclean shutdown (default true)
static std::vector<std::string> TORCH_NCCLFT_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN = {
    "TORCH_NCCLFT_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN"};

// Whether to include only active collectives in the Flight Recorder trace
// (default false)
static std::vector<std::string> TORCH_NCCLFT_EXTRA_DUMP_ON_EXEC = {
    "TORCH_NCCLFT_EXTRA_DUMP_ON_EXEC"};

// Control whether to use CudaEventCache for the collective in watchdog thread.
// We noticed in the past when cuda global lock is held, destroying CudaEvent
// can cause a hang.
static std::vector<std::string> TORCH_NCCLFT_CUDA_EVENT_CACHE = {
    "TORCH_NCCLFT_CUDA_EVENT_CACHE"};

// Control the number of ranks each root can cover during NCCL comm init.
static std::vector<std::string> TORCH_NCCLFT_RANKS_PER_ROOT = {
    "TORCH_NCCLFT_RANKS_PER_ROOT"};

static std::vector<std::string> TORCH_NCCLFT_NAN_CHECK = {"TORCH_NCCLFT_NAN_CHECK"};

constexpr const char* NCCLFT_BACKEND_NAME = "nccl_ft";

constexpr const char* kStoreDumpKey_ft = "exception_dump_ft";

constexpr const char* kStoreErrorSignalKey_ft = "remote_error_ft";

constexpr const int kWorkStatusUpdatePeriodMs_ft = 30 * 1000; // 30 seconds

constexpr auto kProcessGroupNCCLFTDefaultTimeout =
    std::chrono::milliseconds(10 * 60 * 1000);

// NoHandlingFT: do not handle asynchronous NCCL errors
// TearDownFT: tear down process upon error, see `WorkNCCLFT::handleException`
// CleanUpOnlyFT: just clean up collectives and abort communicators without
// tearing down process SkipCleanUpFT: (this is a temporary option and can be
// removed in future) tear down process without cleaning up NCCL communicators.
// This should be used as a last resort in case `ncclCommAbort` itself is
// hanging
enum ErrorHandlingModeFT {
  NoHandlingFT = 0,
  TearDownFT = 1,
  CleanUpOnlyFT = 2,
  SkipCleanUpFT = 3
};

#define SHOULD_CLEAN_UP_FT(a) (a != NoHandlingFT && a != SkipCleanUpFT)

#define SHOULD_TEAR_DOWN_FT(a) (a != NoHandlingFT && a != CleanUpOnlyFT)

#define PRINT_COLLECTIVE_HASH_SIGNATURE_FT(phase, opType, numel, hashValue)      \
  LOG(WARNING) << logPrefix() << "Hash of " << phase << " to NCCLFT " << opType \
               << " with size " << numel << " is " << hashValue;

// If set, ProcessGroupNCCLFT doesn't use recordStream calls to ensure
// caching allocator safety for tensors used on both user-facing and
// internal comm streams.
// Instead, it stashes live references to those tensors until after
// user-facing streams are synced with comm streams.
// See stashed_for_allocator_safety_ below.
static std::vector<std::string> TORCH_NCCLFT_AVOID_RECORD_STREAMS = {
    "TORCH_NCCLFT_AVOID_RECORD_STREAMS"};

// If set, ProcessGroupNCCLFT registers postAlloc and preFree hooks to cuda cache
// allocator so that whenever a tensor is allocated or freed, ProcessGroupNCCLFT
// can register/deregister the tensor on all available NCCL communicators.
static std::vector<std::string> TORCH_NCCLFT_USE_TENSOR_REGISTER_ALLOCATOR_HOOK =
    {"TORCH_NCCLFT_USE_TENSOR_REGISTER_ALLOCATOR_HOOK",
     "NCCLFT_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"};

#if defined(__linux__)
struct DumpPipeFT {
 DumpPipeFT(int rank, const std::string& fileStem, int traceBufferSize) {
    if (fileStem.empty() || traceBufferSize <= 0) {
      return;
    }
    std::string filename = c10::str(fileStem, rank, ".pipe");
    TORCH_CHECK(
        unlink(filename.c_str()) != -1 || errno == ENOENT,
        "Error removing existing named pipe ",
        filename,
        ", Error: ",
        std::strerror(errno));
    TORCH_CHECK(
        mkfifo(filename.c_str(), 0666) != -1,
        "Error creating named pipe ",
        filename,
        ", Error: ",
        std::strerror(errno));
    fd_ = open(filename.c_str(), O_RDONLY | O_NONBLOCK);
    LOG(INFO) << "Pipe file " << filename
              << " has been opened, write to it to trigger NCCL Debug Dump.";
    TORCH_CHECK(fd_ != -1, "Error opening named pipe ", filename);
  }
  bool shouldDump() {
    if (fd_ == -1) {
      return false;
    }
    // NOLINTNEXTLINE(*array*)
    char buf[128]{};
    // non-blocking from O_NONBLOCK above.
    // Ignore EINTR because we already will poll this
    // again later.
    ssize_t bytesRead = read(fd_, &buf, 128);
    return bytesRead > 0;
  }
  ~DumpPipeFT() {
    if (fd_ != -1) {
      close(fd_);
    }
  }

 private:
  int fd_ = -1;
};
#else
struct DumpPipeFT {
  DumpPipeFT(int rank) {}
  bool shouldDump() {
    return false;
  }
};
#endif

// A shelf for stashing tensors between op call and `work.wait()`.
// Used in case of async ops.
class TensorShelfFT {
 public:
  // Stash tensors so that CachingAllocator cannot recycle them prematurely.
  void stash(std::vector<at::Tensor>& tensors);
  // Stash tensors from another shelf.
  void stash(TensorShelfFT& other);
  // Unstage the stashed tensors so that CachingAllocator can recycle them.
  // Same as `clear()`.
  void unstash();
  // Whether shelf is empty.
  bool empty();
  // Clear the shelf.
  void clear();

 protected:
  // Get the inner tensor vector. Use with caution as it is not protected by
  // mutex.
  std::vector<at::Tensor>& get();

 private:
  std::vector<at::Tensor> tVector_;
  // Need a mutex to protect `tVector_` because it can be potentially accessed
  // from both main thread and watchdog thread.
  std::mutex mutex_;
};

// ProcessGroupNCCLFT implements NCCL bindings for c10d.
//
// All functions of the class are expected to be called in the same order
// across all processes in the process group.  This is the only way that we
// can guarantee to match up the same calls among all processes.
//
// All NCCL functions provided by this class are asynchronous functions. More
// specifically, each NCCL call is scheduled on a separate CUDA stream that is
// different from the current CUDA stream. This is for the purpose of
// achieving potentially concurrency and better performance. As a result,
// it is the callers' responsibility to make sure that the CUDA stream their
// code works on needs to wait for the NCCL operation from
// this class.
//
// This can be done by calling:
//
// either WorkNCCLFT::wait() or WorkNCCLFT::synchronize(), both achieves the same
// functionality and are synonyms.
//
// Also note that WorkNCCLFT::finishedGPUExecution() is a helper function only
// provided by ProcessGroupNCCLFT to check if the NCCL operation of WorkNCCLFT has
// finished execution on the GPU (not just scheduled).
//
// Example on using the NCCL process group
//
//   ProcessGroupNCCLFT pg(store, rank, size);
//   std::shared_ptr<WorkNCCLFT> work = pg.allreduce(tensors);
//
//   // At this point, NCCL kernel has already by queued successfully
//   // Now, let current stream wait for the NCCL to finish, this function is
//   // async operation as well
//
//   work->wait()
//
//   // Now continue on other work in the current stream.
class TORCH_API ProcessGroupNCCLFT : public Backend {
 public:
  /* NCCL_FT ======================================================= */
  // 讓全域 C-Callback 可以呼叫
  void trigger_fault_proposal(int dev_idx);
  /* ======================================================= */

  class WorkNCCLFT : public Work, public std::enable_shared_from_this<WorkNCCLFT> {
   public:
    friend struct WorkInfo;

    // Constructor takes a list of CUDA devices
    WorkNCCLFT(
        ProcessGroupNCCLFT* pg,
        std::string pgUID,
        std::string pgDesc,
        at::Device& device,
        int rank,
        OpType opType,
        uint64_t seq,
        bool isP2P = false,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputs = std::nullopt,
        bool enableTiming = false,
        bool cudaEventCacheEnabled = false,
        DebugLevel distDebugLevel = DebugLevel::Off);
    // Copy constructor doing partial copy without outputs_. Cleanup thread
    // monitors and removes finished works. However it will deadlock when
    // destructs outputs_ tensors who are view tensors in autograd graph.
    WorkNCCLFT(const WorkNCCLFT& w);

    ~WorkNCCLFT() override = default;

    // Checks if the NCCL kernel has started to execute.
    bool isStarted();

    // Checks if request has completed. In this specific case of NCCL, it checks
    // if the NCCL operation has completed on the GPU in its own NCCL stream.
    // Non-blocking operation.
    bool isCompleted() override;

    bool isSuccess() const override;

    // Same as calling synchronize() for NCCL work if timeout is not set.
    // Otherwise, it will block the CPU thread until the NCCL work is completed
    // or timed out. If timeout, exception will be thrown.
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    void blockCurrentStream() override {
      synchronize();
    }

    void abort() override;

    // Let current stream wait on the completion of the NCCL work
    // Throws on exceptions.
    void synchronize() override;

    // Synchronize streams by blocking each on the NCCL stream
    void synchronizeStream();

    // Helper function to handle exception (throw if needed).
    void handleException(ErrorHandlingModeFT asyncErrorHandling);

    // Helper function that checks if the NCCL kernels have finished
    // execution on the GPUs
    bool finishedGPUExecution();

    // Get a Future object that will be marked as completed internally.
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    // Get a Future result of each work (e.g. success, different error types).
    // instead of the tensor output.
    c10::intrusive_ptr<c10::ivalue::Future> getFutureResult() override;

    float getDuration() const override;

    uint64_t getSequencenumber() const override;

    std::string logPrefix() const;

    // Helper function that sets an exception_ptr on the WorkNCCLFT object.
    void setException(std::exception_ptr exception_ptr);

    // Helper function that returns True if the WorkNCCLFT object has timed out
    // and False otherwise.
    // In case of timeout, set exception on the WorkNCCLFT object.
    bool checkTimeout(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt);

    // Print the traceback of the collective at call time
    void printTraceback() const;

    std::string getTraceback() const;

    std::vector<at::Tensor> result() override;

   protected:
    // 用來儲存外層 ProcessGroup 的指標
    ProcessGroupNCCLFT* pg_;
    // The process group unique id
    std::string pgUID_;

    // The process group description
    std::string pgDesc_;

    // The cached list of CUDA devices to operate on
    at::Device device_;

    // The start CUDA event of NCCL operator tracking this work item. These
    // start CUDA events are needed by desync debugging if enabled.
    std::shared_ptr<at::cuda::CUDAEvent> ncclStartEvent_;

    // The end CUDA event of NCCL operator tracking this work item.
    std::shared_ptr<at::cuda::CUDAEvent> ncclEndEvent_;

    // The NCCL communicator used for this work item.
    std::shared_ptr<NCCLFTComm> ncclComm_;

    // whether this work is a barrier op
    bool isBarrierOp_{false};

    // Clone of blockingWait_ from ProcessGroupNCCLFT.
    bool blockingWait_{false};

    // Clone of opTimeout_ from ProcessGroupNCCLFT.
    std::chrono::milliseconds opTimeout_{};

    // Ephemeral timeouts are owned by exactly one work,
    // and reset after that work completes.
    // There may be more than one ephemeral timeout active at the same time,
    // and this variable is used to track the ownership of ephemeral timeout.
    std::chrono::milliseconds ownedEphermeralTimeout_ =
        std::chrono::milliseconds(0);

    // Time point representing when the work started.
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

    // Record the sequential number of collective or p2p.
    uint64_t seq_;
    bool isP2P_;

    // Indicates if the nccl start event has been updated to the store trace.
    // This will be used by desync debug.
    bool startTraceUpdated_{false};

    // Record collective sizes for debug. We only record the size on the first
    // device as multi-device per process is deprecated
    size_t numelIn_ = 0;
    size_t numelOut_ = 0;

    // Wrapper method for the static checkForNCCLErrors which can be overridden
    // for tests.
    virtual std::exception_ptr checkForNCCLErrors();

    friend std::ostream& operator<<(
        std::ostream& output,
        const WorkNCCLFT& workNCCLFT);

    // Checks for NCCL errors and sets an appropriate exception_ptr.
    void checkAndSetException();

    // Just checks whether GPU execution has started, without modifying
    // exception_ptr.
    bool startedGPUExecutionInternal() const;

    // Just checks whether GPU execution has completed, without modifying
    // exception_ptr.
    bool finishedGPUExecutionInternal() const;

    // Reference to the store so that we can write aborted communicators
    // to the store.
    c10::intrusive_ptr<Store> store_;

    // Store a reference to NCCL collective's outputs, used by result and to
    // give a more descriptive message when representing the Work as a string.
    std::shared_ptr<std::vector<at::Tensor>> outputs_;

    // TORCH_NCCLFT_AVOID_RECORD_STREAMS implementation helper.
    // Stores references to participating non-output tensors (ie inputs,
    // flattened intermediates).
    // We'll clear this list in synchronizeStream, just after user-facing
    // stream(s) are synced with the nccl work stream(s).
    // By keeping these refs (as well as outputs_) alive until after the
    // collective's work rejoins the user-facing streams, we achieve
    // caching allocator safety without any recordStream calls.
    // For in-place collectives, some refs stashed here may alias outputs_,
    // but that doesn't do any harm.
    std::shared_ptr<TensorShelfFT> stashed_for_allocator_safety_;

    // The future returned by getFuture.
    c10::intrusive_ptr<at::ivalue::Future> future_;

    // the future result (e.g., success or failure) of the work
    c10::intrusive_ptr<at::ivalue::Future> futureWorkResult_;

    bool timingEnabled_;
    // unique id used to tell the trace buffer that this
    // work has completed
    std::optional<uint64_t> trace_id_;
    std::optional<uint64_t> trace_reset_epoch_;
    DebugLevel distDebugLevel_;
    friend class ProcessGroupNCCLFT;
  };

  struct Options : Backend::Options {
    // NOTE: timeout in ProcessGroupNCCLFT::Options denote the timeout for
    // operations. This is only used when blockingWait_ is enabled.
    explicit Options(bool is_high_priority_stream = false);
    Options(const Options&) = default;
    Options(Options&&) noexcept = default;
    Options& operator=(const Options&) = delete;
    Options& operator=(Options&&) noexcept = delete;
    ~Options() override = default;

    // return intrusive_ptr of the object
    static c10::intrusive_ptr<Options> create(
        bool is_high_priority_stream = false) {
      return c10::make_intrusive<Options>(is_high_priority_stream);
    }

    // Schedule NCCL operations on high priority CUDA streams
    bool is_high_priority_stream;

#ifdef NCCL_HAS_CONFIG
    // Configure ranks
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
#endif

    // Optional "parent" backend and color to create communicators from
    // via `ncclCommSplit`
    c10::intrusive_ptr<ProcessGroupNCCLFT> split_from;
    // Color to use for `ncclCommSplit`, values:
    // * Non-negative value: in group;
    // * NCCLFT_SPLIT_NOCOLOR (-1): not in group;
    // * NCCLFT_SPLIT_NOCOLOR - 1: uninitialized.
    // [Note 1]: the type must be `int` instead of `int64_t` because NCCL API
    // accepts int. Otherwise, an implicit conversion may happen at the API call
    // and the value may become negative.
    // [Note 2]: this member is pybinded to Python, the value passed from Python
    // must be within the numerical range of C++ int. Otherwise, Python will
    // raise a RuntimeError saying type is incompatible. See also
    // `_process_group_color` in `distributed_c10d.py`.
#ifdef NCCL_HAS_COMM_SPLIT
    int split_color{NCCL_SPLIT_NOCOLOR - 1};
#else
    // [Note 3]: for older NCCL versions, NCCL_SPLIT_NOCOLOR is not defined. But
    // `split_color` is pybinded to Python, so we need to define it. So we use
    // the int value of `NCCL_SPLIT_NOCOLOR` (-1) instead.
    int split_color{-2};
#endif
  };

  // Helper class related to TORCH_NCCLFT_DESYNC_DEBUG
  class DesyncDebugger {
   public:
    // Initialize and enable DesyncDebugger
    void init(
        int rank,
        int size,
        int globalRank,
        int pgId,
        c10::intrusive_ptr<Store> store);

    // Run desync debug. This function is called by watchdog at time of timeout.
    void run();

    // Log work start to store.
    void logWorkStart(WorkNCCLFT& work);

    // Log work end to store.
    void logWorkEnd(WorkNCCLFT& work);

   private:
    // Whether desync debug is enabled.
    // If false, all functions are no-op.
    bool enabled_{false};

    // From ProcessGroupNCCLFT
    int rank_;
    int size_;
    int globalRank_;
    int pgId_;

    // Reference to the store so that we can log start/end event.
    c10::intrusive_ptr<Store> store_;

    // The store keys to trace the last NCCL collective kernel CUDA events -
    // start event and end event respectively. These are used to do desync root
    // cause analysis.
    std::string traceKeyStart_;
    std::string traceKeyEnd_;
  };

  // Class that runs as a separate thread aside from watchdog
  // thread because we need to check the heartbeat from watchdog thread
  // so that when we get stuck in some NCCL/CUDA calls,
  // we can dump the debugging information and abort the process.
  class HeartbeatMonitor {
   public:
    HeartbeatMonitor(ProcessGroupNCCLFT* pg);
    virtual ~HeartbeatMonitor() = default;

    // Start the heartbeat monitor thread.
    void start();

    // Join the heartbeat monitor thread.
    void join();

    // Run the actual loop to check watchdog heartbeat.
    virtual void runLoop();

    // Set the terminal flag and notify the heartbeat monitor thread to stop.
    void stop();

    // Set the last update time of watchdog thread.
    void setLastWorkListUpdateTime(
        std::chrono::time_point<std::chrono::steady_clock> time);

    int getDumpTimeout() const;

    // Util function to get the timeout error message
    std::string getNCCLWatchdogTimeoutErrorMsg(const std::string& extraMsg);

    // Util function to get the timeout exit message
    std::string getNCCLWatchdogTimeoutExitMsg(const std::string& exitReason);

   protected:
    // We need to keep a reference to the PG instance so that we can access
    // the member functions of the PG instance. We store a raw pointer on
    // purpose because the heartbeat monitor thread now still lives within the
    // lifetime of the PG instance.
    ProcessGroupNCCLFT* pg_;

   private:
    // Whether or not to print C++ stack traces to logs on unclean shutdown.
    bool logCppStackOnUncleanShutdown_;

    // The time interval used for deciding whether there is no watchdog
    // heartbeat.
    int heartbeatTimeoutInSec_;

    // timeout for the dump to finish.
    int waitTimeoutDumpInMilSec_;

    // Interval of check coordinated signals in ProcessGroupNCCLFT from other
    // ranks e.g., trigger the dump of the debugging info for timeout when
    // notified.
    int coordCheckIntervalMilSec_;

    // We gate the heartbeat monitor thread so that we can roll it out
    // gradually.
    bool watchdogHeartbeatMonitorEnabled_;

    // Monitor thread which checks the heartbeat of Watchdog thread.
    // If the monitor thread finds there is no heartbeat, it will dump debug
    // info and then kill the watchdog thread to avoid hang.
    std::thread ncclHeartbeatMonitorThread_;

    // Whether or not we should terminate the heartbeat monitoring threads.
    std::atomic<bool> terminateHeartbeatMonitorThread_{false};

    // Condition Variable for monitor thread to wake up early
    std::condition_variable monitorWakeUpCV_;

    // Whether or not to dump debug info on exception including both watchdog
    // timeout and nccl errors.
    bool dumpOnTimeoutOrEx_;

    // Mutex to Guard monitorWakeUpCV_
    std::mutex monitorMutex_;

    // The last update time of WorkList inside watchdog thread.
    std::chrono::time_point<std::chrono::steady_clock> lastWorkListUpdateTime_;
  };

  // Class that runs as a side thread to check whether the NCCL collective
  // is timed out or errors on the cached NCCL communicators.
  class Watchdog {
   public:
    Watchdog(ProcessGroupNCCLFT* pg);
    virtual ~Watchdog() = default;

    // Start the watchdog thread.
    void start();

    // Join the watchdog thread.
    void join();

    // Function that runs as part of a separate thread and checks for errors on
    // NCCL communicators. We need a separate thread to check for NCCL errors
    // since we can't rely on the user calling certain methods like wait(),
    // isCompleted() etc. to detect and remediate errors. In addition to this,
    // we need a mechanism to safely abort and remove NCCL communicators from
    // our cache. This can be done cleanly by having a thread for the
    // ProcessGroupNCCLFT class. Attempting to modify the communicator cache from
    // the WorkNCCLFT class might run into issues with object lifetime since the
    // ProcessGroupNCCLFT object might get destroyed before the WorkNCCLFT object.
    void run();

    // Watchdog's inside loop.
    // Takes care of cleaning up completed work, and aborting upon failure or
    // timeout.
    void runLoop();

    // Notify the loop inside watchdog.
    void notify();

    void checkAndSetRemoteError();

    // A helper function to get the src rank of a signal from the Store. This is
    // nonblocking function returning -1 if the signal is not available yet.
    int getSignalSrcRank(
        c10::intrusive_ptr<Store>& store,
        const std::string& signal);

    uint64_t getHeartbt() const;

    void setDesyncDebug(bool desyncDebug);

   private:
    std::thread ncclCommWatchdogThread_;

    // We need to keep a reference to the PG instance so that we can access
    // the member functions of the PG instance. We store a raw pointer on
    // purpose because the watchdog thread now still lives within the
    // lifetime of the PG instance.
    ProcessGroupNCCLFT* pg_;

    // Whether the NCCL watchdog should rethrow CUDA errors.
    bool rethrowCUDAErrors_ = false;

    std::exception_ptr watchDogException_ = nullptr;

    // Condition Variable for watchdog thread sleep
    std::condition_variable workMetaListCV_;

    // Heartbeat of watchdog thread.
    std::atomic_uint64_t heartbeat_;

    // Whether or not to propagate detected errors to all ranks in the same PG
    // through TCPStore.
    bool propagatePgError_;

    // Whether or not to enable timeout root cause analysis.
    bool desyncDebug_;

    DesyncDebugger desyncDebugger_;
  };

  // If you wish to create multiple process groups, each with a potentially
  // different rank and size, you can do so by passing a new store instance
  // to each one. If you have only a single store object, you can
  // use the `c10d::PrefixStore` to derive scoped instances.
  // This is also what the Python API in torch.distributed does.
  //
  // The process group instance keeps a reference to the store because
  // it may be used long after the constructor runs. In fact, the constructor
  // doesn't create any NCCL communicators. A single NCCL communicator can
  // only be used on a specific set of devices, and are therefore created
  // on-demand when a collective runs. If another collective is executed later,
  // against a different set of devices, the process group creates another NCCL
  // communicator. These NCCL communicators are cached and reused if possible.
  //
  ProcessGroupNCCLFT(
      c10::intrusive_ptr<Store> store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options = Options::create());

  // This constructor includes the deprecated `groupName` argument.
  // If you have existing code that uses the `groupName`, you can replace
  // it by specifying a `c10d::PrefixStore(groupName, store)` for store.
  C10_DEPRECATED ProcessGroupNCCLFT(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      const std::string& groupName,
      c10::intrusive_ptr<Options> options = Options::create())
      : ProcessGroupNCCLFT(store, rank, size, std::move(options)) {}

  ~ProcessGroupNCCLFT() override;

  // This function returns a local uid for ProcessGroupNCCLFT.
  uint64_t getUid() {
    return static_cast<uint64_t>(local_id_);
  }

  c10::intrusive_ptr<Options> getOptions() {
    return options_;
  }

  c10::intrusive_ptr<Backend::Options> getBackendOptions() override {
    return c10::static_intrusive_pointer_cast<Backend::Options>(options_);
  }

  const std::string getBackendName() const override {
    return std::string(NCCLFT_BACKEND_NAME);
  }

  bool supportsSplitting() const override {
    return true;
  }

  bool supportsCoalescing() const override {
    return true;
  }

  bool supportsTimeEstimation() const override {
#ifdef NCCL_SIM_INFO_INITIALIZER
    return true;
#else
    return false;
#endif
  }

  void setTimeout(std::chrono::milliseconds timeout) override {
    options_->timeout = timeout;
  }

  void startCoalescing() override;

  c10::intrusive_ptr<Work> endCoalescing() override;

  void startTimeEstimate();

  float endTimeEstimate();

  // For specifying a composite optype, such as ALLGATHER and REDUCE_SCATTER
  c10::intrusive_ptr<Work> endCoalescing(OpType optype);

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> _broadcast_oop(
      at::Tensor& outputTensors,
      at::Tensor& inputTensors,
      const BroadcastOptions& opts = BroadcastOptions());

  c10::intrusive_ptr<Work> allreduce_sparse(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work> _reduce_oop(
      at::Tensor& outputTensors,
      at::Tensor& inputTensors,
      const ReduceOptions& opts = ReduceOptions());

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputbuffer,
      at::Tensor& inputbuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  int64_t getCommPtr();

  void groupStart();

  void groupEnd();

  void groupEndNonblocking(const std::shared_ptr<NCCLFTComm>& comm);

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  // Unsupported Ops
  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  // Agrees on an initial sequence number for the whole group by having rank 0
  // create it and broadcast it to other ranks using the store.
  void setSequenceNumberForGroup() override;

  // Retrieves the current sequence number for the whole group, which should be
  // in sync. If the returned number is not consistent across the group, it
  // may indicate that there is some sort of collective desynchronization.
  uint64_t getSequenceNumberForGroup() override;

  // Return the total number of splits the communicators held by this process
  // group have performed.  Counts ncclCommCreateFromRanks() for ncclx v2.21.5+
  uint64_t getCommSplitCounter() const;

  void registerOnCompletionHook(
      std::function<void(std::shared_ptr<WorkInfo>)>&& hook) override;
  void waitForPendingWorks() override;

  void enableCollectivesTiming() override;

  c10::intrusive_ptr<Backend> split(
      const c10::intrusive_ptr<Store>& store,
      const std::vector<int>& ranks,
      const c10::intrusive_ptr<Backend::Options>& opts) override;

  c10::intrusive_ptr<Backend> merge(
      const c10::intrusive_ptr<Store>& store,
      const c10::intrusive_ptr<Backend::Options>& opts,
      const int& rank,
      const int& size) override;

  // Helper function for iteratively aborting communicators in the provided map
  void abortCommsFromMap(
      std::unordered_map<std::string, std::shared_ptr<NCCLFTComm>>& ncclCommsMap,
      const std::optional<std::string>& abortReason);

  c10::intrusive_ptr<intra_node_comm::IntraNodeComm> initIntraNodeComm();

  // Destroy (shutdown) this backend -- normal exit.
  void shutdown() override;

  // Provides an API to abort the ProcessGroup (similar to ncclCommAbort)
  // instead of relying on ProcessGroupNCCLFT destructor.
  void abort() override;

  void eagerConnectSingleDevice(at::Device device) override;

  void performNocolorSplit(at::Device device);

  // If all comms on this PG are fully initialized, return true.
  bool isInitialized();

  ErrorType getError() override;

  bool supportsShrinking() const override {
#ifdef NCCL_HAS_COMM_SHRINK
    return true;
#else
    return false;
#endif
  }

  // Backend-style shrink override that returns a Backend instance.
  c10::intrusive_ptr<Backend> shrink(
      const std::vector<int64_t>& ranks_to_exclude,
      int shrink_flags = 0,
      const c10::intrusive_ptr<Backend::Options>& opts_override =
          nullptr) override;

  std::shared_ptr<c10::Allocator> getMemAllocator() override;

  // Allocate tensor from communication-optimized memory pool
  at::Tensor allocateTensor(long size, at::TensorOptions options = {}) override;

  // Whether tensor allocation from NCCL memory pool is supported
  bool supportsTensorAlloc(c10::DeviceIndex deviceIdx) override;

  // Performs NCCL user buffer registration for all buffers in
  // the given MemPool
  void registerMemPool(at::cuda::MemPool* pool, bool symm = false);

  // Performs NCCL user buffer de-registration for all buffers in
  // the given MemPool
  void deregisterMemPool(at::cuda::MemPool* pool);

  // This method adds a temporary extension for the timeout period,
  // applying to all collectives between the calling of this API and
  // the completion of the first collective on the GPU. While this feature
  // provides flexibility in specific scenarios, it introduces statefulness
  // to timeout setting. Therefore, it is advisable to use this API sparingly
  // and consider alternative approaches, such as directly setting the timeout
  // or utilizing a barrier collective (one can set any timeout to the barrier),
  // whenever feasible.
  void addEphemeralTimeout(const std::chrono::milliseconds& timeout);

  // This function is only intended for testing purposes because we don't
  // want to expose the `WorkNCCLFT` via pybind. It verifies whether the
  // `opTimeout_` of the provided WorkNCCLFT instance is the same as the specified
  // timeout.
  bool verifyWorkTimeoutForTest(
      const c10::intrusive_ptr<Work>& work,
      const std::chrono::milliseconds& timeout);

  void setEnableNanCheck(bool enableNanCheck);

  // APIs related to memory offload (require NCCL 2.29.7+ at runtime)
  void suspend() override;

  void resume() override;

  std::unordered_map<std::string, uint64_t> getMemoryStats() override;

 protected:
  uint64_t getWatchdogHeartbt() const;

  // Instance of the heartbeat monitor thread.
  std::unique_ptr<HeartbeatMonitor> heartbeatMonitor_;

  // Instance of the watchdog thread.
  std::unique_ptr<Watchdog> watchdog_;

  /* ===================================================================== */
  /* --- [NCCL-FT: 零開銷容錯控制面變數] --- */
  std::atomic<uint64_t> do_not_cross_op_{0};
  std::atomic<uint64_t> final_commit_op_{0};
  bool ft_disabled_ = false;

  // Monotonically increasing fault-round counter.  Bumped by the main thread
  // each time it commits a rollback.  Used as a namespace for TCPStore keys
  // so keys from previous rounds do not pollute the current round.
  // Written only by the main thread (under the barrier block); read by the
  // side-car to construct COMMIT key names.  No atomic needed: the main
  // thread writes before unlocking the barrier, side-car reads after COMMIT
  // (there is a TCPStore-mediated happens-before relationship).
  std::atomic<bool> is_degraded_{false};
  // Set to true after the first "Shadow Ping-Pong" log in the current degraded
  // episode; reset to false each time is_degraded_ transitions to true so that
  // the first collective of every new degraded epoch always logs once.
  bool shadow_ping_pong_logged_{false};
  std::atomic<uint64_t> ft_round_{0};

  // Per-rebuild-attempt counter.  Unlike ft_round_, this increments on every
  // entry into rebuild_shadow_ping_pong_topology(), including retries within
  // the same fault round (when execute_shadow_allreduce throws and the
  // side-car catches and retries).  Used to give each rebuild attempt its own
  // unique TCPStore key namespace so stale keys from a failed attempt do not
  // cause later ranks to read wrong data and hang at the barrier wait.
  // Written and read only from recover_and_replay_inflight_ops() (side-car
  // thread); no concurrent writers, so std::atomic with relaxed ordering is
  // sufficient for the counter itself.  The TCPStore set/wait calls provide
  // the necessary cross-rank visibility.
  std::atomic<uint64_t> rebuild_attempt_{0};

  // Number of consecutive rebuild failures within the current fault round.
  // Reset to 0 on each successful recovery.  When it exceeds
  // kMaxConsecutiveRebuildFailures the side-car stops retrying and marks the
  // PG as permanently failed to prevent an unbounded FD-leaking retry loop.
  int consecutive_rebuild_failures_{0};
  static constexpr int kMaxConsecutiveRebuildFailures = 5;

  // ── FT Benchmarking timestamps ─────────────────────────────────────────
  // Disabled when NCCL_FT_NO_TIMING=1 is set in the environment.
  // All use steady_clock; t0 is set on the NCCL callback thread, the rest
  // on the side-car thread.  Values are only meaningful after a completed
  // recovery round — read them from the WARNING summary log.
  bool ft_timing_enabled_ = true;
  struct FTTimings {
    std::chrono::steady_clock::time_point t0_fault_detected;   // NCCL callback fired
    std::chrono::steady_clock::time_point t1_propose_written;  // 2PC PROPOSE sent to TCPStore
    std::chrono::steady_clock::time_point t2_commit_received;  // all ranks committed (consensus)
    std::chrono::steady_clock::time_point t3_recovery_enter;   // recover_and_replay_inflight_ops entered
    std::chrono::steady_clock::time_point t4_topology_rebuilt; // rebuild_shadow_ping_pong_topology done
    std::chrono::steady_clock::time_point t5_replay_done;      // all in-flight buckets replayed
    std::chrono::steady_clock::time_point t6_complete;         // final_commit_op_ cleared, system ready
  };
  FTTimings ft_timings_;

  // Pending fault signal from NCCL callback / Watchdog.
  // Bitmask: bit i is set when local device i has a pending NIC fault.
  // 0 means no pending fault.
  // Written via fetch_or by the callback/watchdog; cleared by the negotiator
  // thread via exchange(0) after it relays the proposal to TCPStore.
  // Using a bitmask (instead of a single int) allows multiple NICs to fault
  // simultaneously without any write silently losing to a failed CAS.
  std::atomic<uint64_t> local_hardware_fault_mask_{0};

  // Persistent set of all local device indices confirmed as faulty by 2PC.
  // Grows monotonically; never shrinks. Protected by faulty_devs_mutex_.
  // The main thread reads it (under lock) only at rebuild time.
  std::unordered_set<int> faulty_local_devs_;
  std::mutex faulty_devs_mutex_;

  // The very first NCCL communicator created for this PG, built via the full
  // ncclCommInitRankConfig path. All subsequent non-P2P comms are derived from
  // this one via ncclCommSplit so that ncclCommInitRankConfig (and its NIC
  // topology scan) is only executed once. ncclCommRegisterFaultCallback is also
  // only registered on this comm.
  std::shared_ptr<NCCLFTComm> ft_root_comm_{nullptr};

  // Intra-node communicator built via ncclCommSplit from the global comm.
  // Inherits the already-validated topology so no NIC topology scan is
  // triggered. Initialised lazily on the first collective() call.
  // Uses NCCLFTComm RAII so no manual ncclCommDestroy is needed.
  std::shared_ptr<NCCLFTComm> local_nvlink_comm_{nullptr};
  // Set to true once the lazy-init block has attempted ncclCommSplit for the
  // NVLink comm.  Prevents the block from firing on every subsequent
  // collective if initLocalNvlinkComm() throws (e.g., NIC failed before the
  // first collective completed).
  bool nvlink_init_attempted_{false};

  // Throttle counter for per-op shadow AllReduce log lines.
  // Logs are emitted on the first call and then every 100 calls thereafter
  // so that degraded-mode steady-state does not flood the log with identical
  // lines differing only in seq number.
  uint64_t shadow_allreduce_log_count_{0};

  // Reduced cross-node communicator built after each fault round is confirmed.
  // Only ranks whose local_rank is NOT in faulty_local_devs_ participate.
  // Rebuilt via ncclCommSplit on every new fault (color=1 for healthy,
  // NCCL_SPLIT_NOCOLOR for faulty) to avoid triggering a NIC topology scan.
  std::shared_ptr<NCCLFTComm> proxy_global_comm_{nullptr};
  int proxy_comm_rank_{-1};
  int proxy_comm_size_{0};
  std::atomic<bool> proxy_comm_ready_{false};

  // The ReduceOp requested by the current allreduce call. Set by allreduce_impl
  // before collective() is called so the degraded path can honour the real op
  // instead of defaulting to SUM.
  ReduceOp current_shadow_reduce_op_{ReduceOp::SUM};

  // -----------------------------------------------------------------------
  // Shadow Buffer: protects AllReduce gradient tensors from partial writes
  // when a NIC fault aborts the op mid-way.
  //
  // Lifecycle:
  //   allreduce_impl (main thread):
  // 
  //     shadow_pre lambda runs on shadow_copy_stream_ (independent of NCCL
  //     stream) to copy GPU gradient -> pinned CPU asynchronously.
  //     shadow_copy_event_ is recorded after the D2H copy so the Watchdog
  //     can verify the checkpoint is complete before doing a restore.
  //
  //   Watchdog (on fault):
  //     waits for shadow_copy_event_ (ensures D2H finished before this fault).
  //     enqueues H2D restore on shadow_copy_stream_: pinned CPU -> GPU output.
  //     records shadow_restore_event_ so the main thread can fence before
  //     starting the replay AllReduce on the NCCL stream.
  //     sets shadow_restore_pending_ and shadow_replay_pending_.
  //
  //   collective() (main thread, next call after 2PC barrier):
  //     shadow_restore_event_.block(ncclStream): inserts a stream-wait so the
  //     NCCL stream does not start the replay AllReduce until the H2D restore
  //     copy on shadow_copy_stream_ has completed.
  //     calls execute_shadow_allreduce with the now-clean tensor.
  //
  // Performance: shadow_copy_stream_ is independent of the NCCL stream.
  // The D2H checkpoint copy runs in parallel with ncclAllReduce on every
  // iteration. The CPU is never blocked; GPU throughput is not reduced.
  //
  // Thread safety: shadow_buf_mutex_ protects shadow_buf_, shadow_seq_,
  // shadow_restore_pending_, and shadow_replay_pending_.
  // The two atomics (pending_shadow_seq_, committed_shadow_seq_) are written
  // by trigger_fault_proposal and the negotiator thread respectively, and
  // read by the negotiator and main threads — no mutex needed for them.
  // -----------------------------------------------------------------------
  // [NCCL-FT] Per-seq Shadow Context: 綁定 CPU 備份、Event 與原始 Tensor 參照
  struct ShadowContext {
      at::Tensor buffer;                                  // Pinned CPU Memory
      std::shared_ptr<at::cuda::CUDAEvent> copy_event;    // D2H 備份完成的 Event
      std::shared_ptr<at::cuda::CUDAEvent> replayed_end_event; // 重播完成的 Event
      std::shared_ptr<at::cuda::CUDAEvent> compute_event; // [新增] 用於同步 compute 與 copy stream
      c10::intrusive_ptr<WorkNCCLFT> work_ptr;            // 綁定原生的 Work，讓系統可以直接檢查它是否成功跑完
      at::Tensor original_input;                          // 用於 H2D 還原的 GPU Tensor
      at::Tensor original_output;
      ReduceOp reduce_op;                                 // 每個 Bucket 可能有不同的 ReduceOp
  };

  std::unordered_map<uint64_t, ShadowContext> in_flight_shadow_bufs_;
  std::vector<ShadowContext> free_shadow_bufs_; 
  std::mutex shadow_buf_mutex_;

  // 用於監控 Pinned Memory 的總用量與區塊數
  std::atomic<size_t> total_pinned_bytes_{0};
  std::atomic<size_t> total_pinned_buffers_{0};

  std::mutex recovery_mutex_; // 保護全域重播中心，確保只有一個 thread 執行重播
  void recover_and_replay_inflight_ops();
  ShadowContext get_or_allocate_shadow_context(const at::Tensor& t);

  std::atomic<uint64_t> shadow_seq_{0};
  at::cuda::CUDAStream shadow_copy_stream_{
      at::cuda::getStreamFromPool(/*isHighPriority=*/false)};

  // shadow_seq_ captured at the moment trigger_fault_proposal fires.
  // Read by the negotiator thread to include in the PROPOSE message.
  // [NCCL-FT Bug 9 fix] Sentinel is UINT64_MAX ("not yet set"), not 0.
  // shadow_seq_ 0 is a valid checkpoint (the very first AllReduce), so 0
  // cannot be used as a sentinel — it would cause the negotiator to wrongly
  // use the proposer's seq instead of this rank's own checkpoint seq.
  std::atomic<uint64_t> pending_shadow_seq_{UINT64_MAX};

  // The shadow_seq value agreed by all ranks via TCPStore consensus.
  // Written by the negotiator thread after COMMIT; read by main thread at replay.
  std::atomic<uint64_t> committed_shadow_seq_{0};
  

  // -----------------------------------------------------------------------


  // 獨立的側車執行緒，專門負責 2PC 協商，絕對不干擾原生 Watchdog
  std::thread ft_negotiator_thread_;
  std::atomic<bool> ft_negotiator_running_{true};

  void start_ft_negotiator_thread();
  uint64_t calculate_safe_buffer();
  // attempt: caller-assigned index that is identical across all ranks for a
  // given invocation.  Lazy-init passes 0; recovery passes (ft_round_ + 1),
  // which is globally agreed via 2PC and therefore consistent on every rank.
  // Using a caller-provided value (instead of a per-instance counter) avoids
  // the mismatch that would occur when some ranks complete lazy-init (counter=1)
  // while others fail and stay at 0, causing them to use different key sets in
  // the subsequent recovery call.
  void initLocalNvlinkComm(uint64_t attempt);
  void rebuild_shadow_ping_pong_topology();
  
  void execute_shadow_allreduce(
      at::Tensor& input,
      at::Tensor& output,
      at::cuda::CUDAStream& stream,
      ReduceOp reduceOp);
  /* ===================================================================== */




  // Helper that broadcasts nccl unique ID to all ranks through the store
  void broadcastUniqueNCCLID(
      ncclUniqueId* ncclID,
      bool isSingleP2POp,
      const std::string& devicesKey,
      int p2pRank);

  // Helper that allgathers nccl unique IDs to all ranks through the store
  void allgatherUniqueNCCLIDs(
      int rootIdx,
      ncclUniqueId* ncclID,
      std::vector<ncclUniqueId>& ncclIDs);

  // Helper that looks up the cached NCCL communicators only
  std::shared_ptr<NCCLFTComm> getNCCLComm(const std::string& deviceKey);

  std::shared_ptr<NCCLFTComm> initNCCLComm(
      const std::string& deviceKey,
      at::Device& device,
      OpType opType,
      int p2pRank = 0,
      bool isSendRecvSelf = false);

  // Initialize device-specific state (comm, stream, event, bookkeeping) for a
  // given communicator on this process group instance.
  void initializeDeviceStateForComm(
      const at::Device& device,
      std::shared_ptr<NCCLFTComm> comm);

  // Wrapper method which can be overridden for tests.
  virtual std::exception_ptr checkForNCCLErrors(
      std::shared_ptr<NCCLFTComm>& ncclComm);

  // Ensure thaht if record is True, the work obj will be enqueued via
  // workEnqueue
  virtual c10::intrusive_ptr<ProcessGroupNCCLFT::WorkNCCLFT> initWork(
      at::Device& device,
      int rank,
      OpType opType,
      bool isP2P,
      const char* profilingTitle = nullptr,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {},
      bool record = false);

  // In the timeout case and we will dump debug info such as the NCCL flight
  // recorder to storage. Down the road, if we have more complicated or blocking
  // operations, we might need to use a side thread to do it.
  bool dumpDebuggingInfo(
      bool includeStackTrace = true,
      bool onlyActive = false);

  void dumpExtraDebuggingInfo();

  // Abort all communicators on this rank.
  bool abortComms(const std::optional<std::string>& abortReason = std::nullopt);

  // A helper function to check if nonblocking API mode should be used.
  // Use this helper instead of directly checking `useNonblocking_` variable.
  bool useNonblocking();

 protected:
  int globalRankStart_{};
  int globalRankStride_{};

 private:
  bool eagerInit_{false};
  bool showSerializationWarning_{true};

  // Helper that encapsulates work shared across all collective communication
  // primitives.  The callbacks have the following signatures:
  //
  //    ncclResult_t fn(at::Tensor& input, at::Tensor& output,
  //                    ncclComm_t, at::cuda::CUDAStream&);
  //    void {pre,post}(std::vector<at::cuda::CUDAStream&>);
  template <typename Fn>
  c10::intrusive_ptr<Work> collective(
      at::Tensor& input,
      at::Tensor& output,
      Fn fn,
      OpType opType,
      bool asyncOp,
      const char* profilingTitle = nullptr,
      bool nanCheck = true);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> collective(
      at::Tensor& input,
      at::Tensor& output,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      OpType opType,
      bool asyncOp,
      const char* profilingTitle = nullptr,
      bool nanCheck = true);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> collective(
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      OpType opType,
      bool asyncOp,
      const char* profilingTitle = nullptr,
      bool nanCheck = true);

  template <typename Fn>
  c10::intrusive_ptr<Work> collectiveCoalesced(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      OpType opType,
      bool asyncOp,
      const char* profilingTitle = nullptr);

  // Helper that encapsulates work shared across point-to-point communication
  // primitives. It is the same structure as the helper used for collective
  // communication primitives.
  template <typename Fn>
  c10::intrusive_ptr<Work> pointToPoint(
      at::Tensor& tensor,
      Fn fn,
      int peer,
      OpType opType,
      const char* profilingTitle = nullptr);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> pointToPoint(
      at::Tensor& tensor,
      Fn fn,
      int peer,
      OpType opType,
      PreProcess pre,
      PostProcess post,
      const char* profilingTitle);

  c10::intrusive_ptr<Work> allreduce_impl(
      at::Tensor& tensor,
      const char* profilingTitle = "nccl:all_reduce",
      const AllreduceOptions& opts = AllreduceOptions());

  // Checks for NCCL errors on each of the communicators and returns an
  // appropriate exception_ptr (nullptr if no errors).
  static std::exception_ptr checkForNCCLErrorsInternal(
      std::shared_ptr<NCCLFTComm>& ncclComm);

  // Return the CUDA device most likely associated with this backend.
  // If we aren't bound to a specific device, there is no strict
  // guarantee that this heuristic is the correct assignment of ranks
  // to GPUs that Python layers use, but in practice it tends to be.
  // Fortunately we don't rely on this for correctness of any tensor
  // operations, just for ancillary uses like barriers.
  at::Device guessDeviceForRank() const;

  // Destroys initialized NCCL communicators in devNCCLComMap_ given by input
  // key. Throws if there are no communicators to destroy. Also removes
  // communicators from the cache and clears used device indices.
  void destroyNCCLComms(const std::string& devNCCLCommMapKey);

  void runHookLoop();

  // Generates a prefix that is unique to this process group and rank, for
  // disambiguating logs
  std::string createLogPrefix() const;

  // Returns the unique prefix created in createLogPrefix
  const std::string& logPrefix() const;

  // Returns the global rank of the device. This function assumes that users
  // always create a default global process group(PG) which includes all
  // devices. It is called in the constructor of ProcessGroupNCCLFT, so it always
  // return the rank_ of the very first PG created, aka, default global PG.
  const int& globalRank() const;

  const c10::intrusive_ptr<Store>& globalStore() const;

  // Returns the global ranks of a PG.
  const std::vector<uint64_t>& groupRanks() const;

  // Util function to assign timeout to each work.
  void assignTimeoutToWork(
      const c10::intrusive_ptr<ProcessGroupNCCLFT::WorkNCCLFT>& work,
      const c10::intrusive_ptr<Options>& option);

  // Broadcast flight-recorder dump signal
  void broadcastDumpSignal();

  // A helper function to broadcast a signal (key) from a src rank to all other
  // ranks using the specified store.
  void broadcastSignal(
      c10::intrusive_ptr<Store>& store,
      const std::string& signal,
      int srcRank);

 protected:
  // Function that directly trigger std::abort so that the whole process
  // gets terminated.
  virtual void terminateProcess(const std::string& errMsg);

  // A helper function to wait for a future to complete or timeout.
  // Returns true if the future completes before timeout, false otherwise.
  bool waitForFutureOrTimeout(
      std::future<bool>& fut,
      const std::chrono::milliseconds& timeOutMilSec,
      const std::string& futDescription,
      ::c10d::C10dLoggingData& debugLog,
      bool throwException = false);

  // A helper function to guess the device id of the current rank, based on
  // bounded device or used device. Do not use this function if you already know
  // the device id to operate on.
  c10::DeviceIndex guessDeviceId() const;

  static const int64_t kWatchdogThreadSleepMillis;

  // The store is used to broadcast the NCCL unique ID of rank 0. This store
  // comes with prefix and it is different across ProcessGroup NCCL instances
  // (aka, different ProcessGroups).
  c10::intrusive_ptr<Store> store_;

  // Reference to the store without prefix so that keys are same across all
  // ProcessGroup NCCL instances and (key, value) pairs written to the store are
  // global.
  c10::intrusive_ptr<Store> globalStore_;

  // The lock which protects the write/read of
  // ephemeralTimeoutActive_/ephemeralTimeoutInflight_.
  // TODO(fduwjj): We need to have an audit on all mutexes we are adding here.
  // And consolidate them if possible.
  std::mutex mtxTimeoutExtension_;

  // The ephemeral timeout added on top of existing timeout for works issued
  // before first work finishes.
  std::chrono::milliseconds ephemeralTimeoutActive_ =
      std::chrono::milliseconds(0);

  // The ephemeral timeout addition which has been already applied to work.
  std::chrono::milliseconds ephemeralTimeoutInflight_ =
      std::chrono::milliseconds(0);

  const c10::intrusive_ptr<Options> options_;

  // The number of NCCL communicators that have been created during
  // the lifetime of this process group. This sequence number is
  // used to scope keys used in the store.
  uint64_t ncclCommCounter_{0};

  // The NCCL communicator that the process group has cached.
  //
  // For collective operations:
  // The key is a list of GPU devices that an operation is operating on
  // The GPU devices are stored in a device sequence and the cache NCCL
  // communicator is associated with this GPU device sequence
  //
  // e.g. If the process group op only uses device 0, then the value of
  // the used device string stored (value of the hashmap) would be "0".
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 1, 2, 3, 4, 5, 6, 7 separately,
  //      then the value of the used device string (key) stored would be
  //      "0,1,2,3,4,5,6,7"
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 4, 5, 6, 7, 1, 2, 3 separately,
  //      then the value of the used device string stored would be
  //      "0,4,5,6,7,1,2,3"
  //
  //      Note that the order of the device for the tensor list matters.
  //
  // For point-to-point operations:
  // The key is a string of my current rank and the peer process rank.
  // e.g. If process 1 and process 2 are involved in a point-to-point
  // communication, the key will be "1:2" on both processes. Note: this is for
  // the scenario where there is only 1 GPU per process. When it comes to
  // multiple GPUs per process, this part may need to redesigned.
  // TODO: we probably need a separate map for P2P comms
  std::unordered_map<std::string, std::shared_ptr<NCCLFTComm>> devNCCLCommMap_;

  // The NCCL communicators currently in process of being initialized.
  std::unordered_map<std::string, std::shared_ptr<NCCLFTComm>>
      inInitializationCommMap_;

  // Mutex to guard maps like devNCCLCommMap_.
  std::mutex mutex_;

  // Size of ring buffer where we store NCCL Traces for debugging.
  int traceBufferSize_;

  // Stores TORCH_NCCL_DEBUG_INFO_PIPE_FILE
  std::string debugInfoPipeFile_;

  // We gate the cudaEventCache so that we can roll it out gradually.
  std::atomic<bool> cudaEventCacheEnabled_;

  std::thread onCompletionHookThread_;

  // Whether or not we should terminate the watchdog and workCleanup threads.
  std::atomic<bool> terminateProcessGroup_;

  // Whether there are hooks pending to be fired
  std::atomic<bool> hasPendingHooks_;

  // This is the signal from watchdog threads to indicate whether the monitor
  // thread should dump. Making it static so that it is accessible from all the
  // PGs. With this flag, monitor thread would dump debug info under any one of
  // the three conditions:
  //
  // 1: watchdog thread of any PG detects a collective timeout.
  // 2: timeout signal is received from other ranks through tcpstore.
  // 3: current PG's watchdog heartbeat timeout occurs.
  //
  // Note that only the monitor thread from PG0 will dump the debug info for
  // case one and two so that the debug info is only dumped once.
  static std::atomic<bool> shouldDump_;

  // Mutex to Guard workMetaList_
  std::mutex workMetaListMutex_;

  bool writeDebugInfo_ = false;

  // Vector to store WorkNCCLFT pointers
  std::list<ProcessGroupNCCLFT::WorkNCCLFT> workMetaList_;

  // Mutex to Guard workMetaList_
  std::mutex completedWorkListMutex_;

  // Condition Variable for watchdog thread sleep
  std::condition_variable completedWorkListCV_;

  std::list<ProcessGroupNCCLFT::WorkNCCLFT> completedWorkList_;

  // Add Work Pointer to workVector
  void workEnqueue(
      const c10::intrusive_ptr<ProcessGroupNCCLFT::WorkNCCLFT>& /*work*/);

  // The CUDA streams used by NCCL kernels
  std::unordered_map<std::string, at::cuda::CUDAStream> ncclStreams_;

  // The CUDA events used to sync NCCL streams
  std::unordered_map<std::string, at::cuda::CUDAEvent> ncclEvents_;

  // Device Indexes used for all collectives in this group
  std::set<c10::DeviceIndex> usedDeviceIdxs_;

  // Flag to denote if a coalescing groupStart/groupEnd block is active
  int coalescing_state_ = 0;

  // Stores device indexes for all collectives run inside a coalescing block
  at::Device coalescedDevice_ = at::Device("cuda");

  // Stores communicators for all collectives run inside a coalescing block
  std::shared_ptr<NCCLFTComm> coalescedComm_ = nullptr;

  // Whether the coalesced calls are sync or async.
  bool coalescedAsync_{};

  // keeps track of input and output tensors when coalescing is in flight.  Will
  // hand over these tensors to WorkNCCLFT's stash when coalescing is ended.
  TensorShelfFT coalescedTensors_;

  // Some ops may have completed, but user still hasn't called `work.wait()`.
  // When watchdog detects this, it transfers the TensorShelfFT from `work` to
  // this `shelves` structure. Next time we execute ProcessGroupNCCLFT's methods
  // on main thread, we clear the `shelves` in one shot. This is mainly because
  // watchdog (a side thread) unstashing the shelf directly seems to cause some
  // problem.
  std::vector<std::shared_ptr<TensorShelfFT>> shelvesToUnstash_;
  std::mutex shelvesMutex_;

  // Whether or not wait() and synchronize() are blocking operations that wait
  // for the operation to complete.
  bool blockingWait_ = false;

  // Whether or not the workCleanupThread is used to perform async error
  // handling.
  ErrorHandlingModeFT asyncErrorHandling_ = NoHandlingFT;

  ErrorType error_ = ErrorType::SUCCESS;

  std::mutex errorMutex_;

  // Whether or not to sleep after an exception is thrown in the watchdog.
  bool sleepAfterException_{};

  // Whether or not to enable nan check for input tensors to collectives.
  bool enableNanCheck_;

  // Whether or not to create start CUDAEvent and enable timing for start
  // and end events. Note that enableTiming_ is always true if desyncDebug_
  // is set to true.
  std::atomic<bool> enableTiming_;

  // Flag to enable the print of hash value of input/output of collectives for
  // verification.
  std::atomic<bool> enableCollectiveHashDebug_;

  // Whether or not TORCH_NCCLFT_AVOID_RECORD_STREAMS was set
  bool avoidRecordStreams_ = false;

  // The number of active ncclGroupStart() calls. This counter will be increased
  // by 1 when ncclGroupStart() is called and decreased by 1 when ncclGroupEnd()
  // is called.
  static thread_local uint64_t ncclActiveGroupCounter_;

  // Counting for the sequential number of NCCL collective call.
  // (specifically, how many actual kernels we launched, which differs from
  // op_id_ when coalescing is enabled)
  uint64_t seqCollective_{0};

  // Counting for the sequential number of NCCL P2P calls.
  uint64_t seqP2P_{0};

  // Incrementing counter for logical operations (collective or p2p) issued on
  // the ProcessGroup
  uint64_t op_id_{0};

  // The number of ProcessGroupNCCLFT created on the current rank.
  size_t local_id_;

  std::string logPrefix_;

  c10::intrusive_ptr<intra_node_comm::IntraNodeComm> intraNodeComm_;

  // Number of devices on this node.
  int localDeviceCount_{0};

  std::shared_ptr<ProcessGroupStatus> pgStatus_ =
      std::make_shared<ProcessGroupStatus>();

  // Internal cached value: use NCCL non-blocking API mode or not.
  // Use `useNonblocking()` method instead of accessing this variable directly.
  std::optional<bool> useNonblocking_{std::nullopt};

  // Communication-optimized memory pool associated with this PG
  std::unique_ptr<at::cuda::MemPool> memPool_ = nullptr;
};

// Reset the flighrecorder recordings for the current rank.
TORCH_API void reset_nccl_ft_trace();

// Dumps the NCCL comm traces and additional information about the Process
// Group.
TORCH_API std::string dump_nccl_ft_trace(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive);

// Dumps the NCCL comm traces and additional information about the Process
// Group in JSON formatted string.
// We don't include stack traces in JSON format as it is far too much data.
TORCH_API std::string dump_nccl_ft_trace_json(
    bool includeCollectives,
    bool onlyActive);

// // Gets a mutable reference to a global optional function.Heartbeat Monitor
// // will use this function to dump traces, if available. Inside fbcode, we
// // store a function here that uses an internal tool for process tracing
TORCH_API std::optional<
    std::function<void(std::function<void(const std::string&)>)>>&
get_cpp_trace_dumper();

// Similar to get_cpp_trace_dumper, this stores a function defined in
// torch-python layer that lets us check whether the GIL can be acquired,
// helpful for instrumenting in cases where a hang was observed.
typedef bool (*gil_checker_t)();

TORCH_API gil_checker_t& get_gil_checker();
} // namespace c10d

#endif // USE_C10D_NCCL
