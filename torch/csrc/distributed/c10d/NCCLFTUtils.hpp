#pragma once

#ifdef USE_C10D_NCCL

#include <sched.h>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/util/Exception.h>
#include <nccl.h>
#include <torch/csrc/cuda/nccl.h>

// 引入原始的 NCCLUtils
// 藉此繼承所有 NCCL_HAS_XXX 版本巨集、輔助函式 (getNcclVersion等) 及 C10D_SCHED_SLEEP 定義
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <c10/util/Exception.h>

namespace c10 {
class C10_EXPORT NCCLFaultToleranceError : public DistBackendError {
 public:
  using DistBackendError::DistBackendError;
};
} // namespace c10


namespace c10d {

// 確保 C++ 找得到這兩個全域函數的宣告 (它們的實作在原版 ProcessGroupNCCL.cpp 中)
typedef bool (*gil_checker_t)();
TORCH_API gil_checker_t& get_gil_checker();

TORCH_API std::optional<
    std::function<void(std::function<void(const std::string&)>)>>&
get_cpp_trace_dumper();



// =========================================================================
// FT 專用巨集區塊 (繞過 TORCH_CHECK_WITH 的限制，手動 throw ::c10d::NCCLFaultToleranceError)
// =========================================================================

#define C10D_NCCL_FT_CHECK(cmd, failureReason)                                \
  do {                                                                        \
    ncclResult_t result = cmd;                                                \
    if (result != ncclSuccess) {                                              \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +     \
          std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) + \
          "\n" + getNcclErrorDetailStr(result, failureReason);                \
      throw ::c10::NCCLFaultToleranceError(                                  \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, err);        \
    }                                                                         \
  } while (0)

#define C10D_NCCL_FT_CHECK_NONBLOCKING(cmd, failureReason)                    \
  do {                                                                        \
    ncclResult_t result = cmd;                                                \
    if (result != ncclSuccess && result != ncclInProgress) {                  \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +     \
          std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) + \
          "\n" + getNcclErrorDetailStr(result, failureReason);                \
      throw ::c10::NCCLFaultToleranceError(                                  \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, err);        \
    }                                                                         \
  } while (0)

#define C10D_FT_CHECK_TIMEOUT(startTime, timeout)                           \
  do {                                                                      \
    auto currentTime = std::chrono::steady_clock::now();                    \
    auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(    \
                           currentTime - startTime)                         \
                           .count();                                        \
    if (timeElapsed > timeout) {                                            \
      std::string err = "NCCL timeout in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__);                                         \
      throw ::c10::NCCLFaultToleranceError(                                \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, err);      \
    }                                                                       \
  } while (0)

#define C10D_NCCL_FT_CHECK_TIMEOUT_BASE(                                      \
    cmd, commWrapper, failureReason, yield_fn)                                \
  do {                                                                        \
    ncclResult_t result = cmd;                                                \
    auto startTimepoint = std::chrono::steady_clock::now();                   \
    auto timeout = nccl_nonblocking_timeout();                                \
    while (result == ncclInProgress) {                                        \
      C10D_FT_CHECK_TIMEOUT(startTimepoint, timeout);                         \
      yield_fn;                                                               \
      commWrapper->getAsyncError(&result);                                    \
    }                                                                         \
    if (result != ncclSuccess) {                                              \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +     \
          std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) + \
          "\n" + getNcclErrorDetailStr(result, failureReason);                \
      throw ::c10::NCCLFaultToleranceError(                                  \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, err);        \
    }                                                                         \
  } while (0)

#define C10D_NCCL_FT_CHECK_TIMEOUT(cmd, commWrapper, failureReason) \
  C10D_NCCL_FT_CHECK_TIMEOUT_BASE(cmd, commWrapper, failureReason, sched_yield())

#define C10D_NCCL_FT_CHECK_TIMEOUT_SLEEP(cmd, commWrapper, failureReason) \
  C10D_NCCL_FT_CHECK_TIMEOUT_BASE(                                        \
      cmd, commWrapper, failureReason, C10D_SCHED_SLEEP())

#define C10D_NCCL_FT_CHECK_TIMEOUT_GROUPEND(cmd, comm, failureReason)        \
  do {                                                                       \
    ncclResult_t state = cmd;                                                \
    auto startTimepoint = std::chrono::steady_clock::now();                  \
    auto timeout = nccl_nonblocking_timeout();                               \
    if (state == ncclInProgress) {                                           \
      do {                                                                   \
        C10D_FT_CHECK_TIMEOUT(startTimepoint, timeout);                      \
        sched_yield();                                                       \
        comm->getAsyncError(&state);                                         \
      } while (state == ncclInProgress);                                     \
    }                                                                        \
    if (state != ncclSuccess) {                                              \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +    \
          std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(state) + \
          "\n" + getNcclErrorDetailStr(state, failureReason);                \
      throw ::c10::NCCLFaultToleranceError(                                 \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, err);       \
    }                                                                        \
  } while (0)

#define C10D_NCCL_FT_ASSERT(cmd)                             \
  do {                                                       \
    ncclResult_t result = cmd;                               \
    if (result != ncclSuccess) {                             \
      std::string err = ncclGetErrorWithVersion(result);     \
      fprintf(                                               \
          stderr,                                            \
          "NCCL error in: %s:%d, %s\n",                      \
          __FILE__,                                          \
          __LINE__,                                          \
          err.c_str());                                      \
      abort();                                               \
    }                                                        \
  } while (0)



// 接下來保留原本的 class NCCLFTComm 定義...
// =========================================================================
// NCCLFTComm 類別宣告
// =========================================================================

// RAII wrapper for NCCL FT communicator
class NCCLFTComm {
  using MutexType = std::recursive_mutex;
  using LockType = std::unique_lock<MutexType>;

 public:
  explicit NCCLFTComm(ncclComm_t ncclComm);

  NCCLFTComm() = default;

  ~NCCLFTComm() noexcept;

  void setUniqueHash(ncclUniqueId ncclId);
  void setUniqueHash(std::string hash);
  std::string getUniqueHash();

  static std::shared_ptr<NCCLFTComm> create(
      int numRanks,
      int rank,
      ncclUniqueId commId,
      at::DeviceIndex deviceIndex);

#ifdef NCCL_HAS_CONFIG
  static std::shared_ptr<NCCLFTComm> create(
      int numRanks,
      int rank,
      ncclUniqueId commId,
      at::DeviceIndex deviceIndex,
      ncclConfig_t& config);
#ifdef NCCL_HAS_INIT_RANK_SCALABLE
  static std::shared_ptr<NCCLFTComm> create_scalable(
      int numRanks,
      int rank,
      std::vector<ncclUniqueId>& commIds,
      at::DeviceIndex deviceIndex,
      ncclConfig_t& config);
#endif // NCCL_HAS_INIT_RANK_SCALABLE
#endif // NCCL_HAS_CONFIG

#ifdef NCCL_HAS_COMM_SPLIT
  static std::shared_ptr<NCCLFTComm> split(
      NCCLFTComm* source,
      int color_id,
      int rank,
      ncclConfig_t& config);
#endif // NCCL_HAS_COMM_SPLIT

#ifdef NCCL_HAS_COMM_SHRINK
  static std::shared_ptr<NCCLFTComm> shrink(
      NCCLFTComm* source,
      std::vector<int>& ranks_to_exclude,
      ncclConfig_t* config,
      int shrinkFlags = 0);
#endif // NCCL_HAS_COMM_SHRINK

#if (defined(IS_NCCLX) || defined(USE_ROCM)) && defined(NCCL_COMM_DUMP)
  std::unordered_map<std::string, std::string> ncclCommDump();
#endif

  at::DeviceIndex getDeviceIndex();

  // Must not be copyable
  NCCLFTComm(const NCCLFTComm&) = delete;
  NCCLFTComm& operator=(const NCCLFTComm&) = delete;

  // Do not support move assignment as there is no valid use case
  NCCLFTComm& operator=(NCCLFTComm&& other) = delete;

  // Move constructable
  // NOLINTNEXTLINE(*-noexcept-move-*)
  NCCLFTComm(NCCLFTComm&& other);

  ncclComm_t getNcclComm();

  void waitReady(bool longInterval);

  std::optional<std::string> getNcclCommFailureReason() const;

  void abort(std::optional<std::string> commFailureReason = std::nullopt);

  void finalize();

  void destroy();

  bool isInitialized() const;

  bool isAborted() const;

  uint64_t getCommSplitCounter() const;

  ncclResult_t checkForNcclError();

  ncclResult_t getAsyncError(ncclResult_t* asyncError);

  ncclResult_t registerSegment(
      void* ptr,
      size_t size,
      bool errorOnRereg = true,
      bool window = false);

  ncclResult_t deregisterSegment(void* ptr, bool window = false);

  std::string repr() const;

  friend class ProcessGroupNCCLFT;
 
 protected:
  // Unique hash for this communicator.
  std::string uniqueHash_;
  bool aborted_{false};
  uint64_t ncclCommSplitCounter_{0};
  ncclResult_t ncclAsyncErr_{ncclSuccess};
  mutable MutexType mutex_;
  // Rank that this communicator corresponds to.
  int rank_{};
  // Optional reason for communicator failure
  std::optional<std::string> commFailureReason_;
  bool initialized_{false};
  bool nonBlocking_{true};
  // Device index for which the NCCL comm is created
  at::DeviceIndex deviceIndex_{-1};
#ifdef NCCL_HAS_COMM_REGISTER
  // Stores handlers for tensors registered by NCCL
  std::unordered_map<void*, void*> registeredSegmentHandles_;
#endif // NCCL_HAS_COMM_REGISTER

 private:
  ncclComm_t ncclComm_{nullptr};
};

} // namespace c10d

#endif // USE_C10D_NCCL