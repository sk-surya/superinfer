#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

#include <cuda_runtime_api.h>

#include <superinfer/base/result.hpp>

namespace superinfer::sm120::cuda_runtime {

inline base::Status cuda_status(cudaError_t error, std::string_view context) {
  if (error == cudaSuccess) return {};
  base::Status status = error == cudaErrorMemoryAllocation
                            ? base::Status::resource_exhausted(cudaGetErrorString(error))
                            : (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver
                                   ? base::Status::unavailable(cudaGetErrorString(error))
                                   : base::Status::internal(cudaGetErrorString(error)));
  status.with_context(context);
  return status;
}

/** Move-only owner for an allocation on the selected CUDA device. */
class DeviceBuffer final {
 public:
  static base::Result<DeviceBuffer> allocate(std::uint64_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max()) {
      return base::Status::overflow("CUDA allocation does not fit host size_t");
    }
    if (bytes == 0) return DeviceBuffer{};
    void* pointer = nullptr;
    const cudaError_t error = cudaMalloc(&pointer, static_cast<std::size_t>(bytes));
    if (error != cudaSuccess) return cuda_status(error, "cudaMalloc");
    return DeviceBuffer{pointer, bytes};
  }

  DeviceBuffer() = default;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : pointer_(other.pointer_), bytes_(other.bytes_) {
    other.pointer_ = nullptr;
    other.bytes_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) return *this;
    (void)reset();
    pointer_ = other.pointer_;
    bytes_ = other.bytes_;
    other.pointer_ = nullptr;
    other.bytes_ = 0;
    return *this;
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { (void)reset(); }

  base::Status reset() noexcept {
    if (pointer_ == nullptr) return {};
    const cudaError_t error = cudaFree(pointer_);
    if (error != cudaSuccess) return cuda_status(error, "cudaFree");
    pointer_ = nullptr;
    bytes_ = 0;
    return {};
  }

  [[nodiscard]] void* data() noexcept { return pointer_; }
  [[nodiscard]] const void* data() const noexcept { return pointer_; }
  [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

 private:
  DeviceBuffer(void* pointer, std::uint64_t bytes) : pointer_(pointer), bytes_(bytes) {}

  void* pointer_{nullptr};
  std::uint64_t bytes_{0};
};

/** Move-only owner for one non-blocking CUDA stream. */
class StreamOwner final {
 public:
  static base::Result<StreamOwner> create() {
    cudaStream_t stream = nullptr;
    const cudaError_t error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (error != cudaSuccess) return cuda_status(error, "cudaStreamCreateWithFlags");
    return StreamOwner{stream};
  }

  StreamOwner(StreamOwner&& other) noexcept : stream_(other.stream_) { other.stream_ = nullptr; }
  StreamOwner& operator=(StreamOwner&& other) noexcept {
    if (this == &other) return *this;
    (void)reset();
    stream_ = other.stream_;
    other.stream_ = nullptr;
    return *this;
  }
  StreamOwner(const StreamOwner&) = delete;
  StreamOwner& operator=(const StreamOwner&) = delete;
  ~StreamOwner() { (void)reset(); }

  base::Status reset() noexcept {
    if (stream_ == nullptr) return {};
    const cudaError_t error = cudaStreamDestroy(stream_);
    if (error != cudaSuccess) return cuda_status(error, "cudaStreamDestroy");
    stream_ = nullptr;
    return {};
  }

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  explicit StreamOwner(cudaStream_t stream) : stream_(stream) {}
  cudaStream_t stream_{nullptr};
};

/** Move-only owner for one timing-independent CUDA dependency event. */
class EventOwner final {
 public:
  static base::Result<EventOwner> create() {
    cudaEvent_t event = nullptr;
    const cudaError_t error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (error != cudaSuccess) return cuda_status(error, "cudaEventCreateWithFlags");
    return EventOwner{event};
  }

  EventOwner(EventOwner&& other) noexcept : event_(other.event_) { other.event_ = nullptr; }
  EventOwner& operator=(EventOwner&& other) noexcept {
    if (this == &other) return *this;
    (void)reset();
    event_ = other.event_;
    other.event_ = nullptr;
    return *this;
  }
  EventOwner(const EventOwner&) = delete;
  EventOwner& operator=(const EventOwner&) = delete;
  ~EventOwner() { (void)reset(); }

  base::Status reset() noexcept {
    if (event_ == nullptr) return {};
    const cudaError_t error = cudaEventDestroy(event_);
    if (error != cudaSuccess) return cuda_status(error, "cudaEventDestroy");
    event_ = nullptr;
    return {};
  }

  [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

 private:
  explicit EventOwner(cudaEvent_t event) : event_(event) {}
  cudaEvent_t event_{nullptr};
};

}  // namespace superinfer::sm120::cuda_runtime
