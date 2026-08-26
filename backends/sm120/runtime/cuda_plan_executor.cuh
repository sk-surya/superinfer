#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include <sm120/runtime/cuda_ownership.cuh>
#include <superinfer/base/views.hpp>
#include <superinfer/ir/physical_plan.hpp>

namespace superinfer::sm120::cuda_runtime {

struct CudaExecutionTrace final {
  std::uint64_t commands_executed{0};
  std::uint64_t launches{0};
};

namespace detail {

__global__ inline void baseline_command_stub() {}

__global__ inline void residual_f32(const float* left, const float* right, float* output,
                                    std::size_t elements) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += blockDim.x * gridDim.x) {
    output[index] = left[index] + right[index];
  }
}

__global__ inline void rms_norm_f32(const float* input, float* output, std::size_t elements) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  float sum_squares = 0.0F;
  for (std::size_t index = 0; index < elements; ++index) sum_squares += input[index] * input[index];
  const float denominator = sqrtf(sum_squares / static_cast<float>(elements) + 1.0e-5F);
  for (std::size_t index = 0; index < elements; ++index) output[index] = input[index] / denominator;
}

using LaunchFunction = cudaError_t (*)(const ir::physical::CommandDescriptor&,
                                       const ir::physical::Plan&, void*, void*, cudaStream_t);

inline void* buffer_pointer(const ir::physical::Plan& plan, void* arena,
                            ir::physical::BufferId id) {
  return static_cast<std::byte*>(arena) + static_cast<std::size_t>(plan.buffers()[id.value()].offset);
}

inline cudaError_t launch_stub(const ir::physical::CommandDescriptor&, const ir::physical::Plan&,
                               void*, void*, cudaStream_t stream) {
  baseline_command_stub<<<1, 1, 0, stream>>>();
  return cudaGetLastError();
}

inline cudaError_t launch_copy(const ir::physical::CommandDescriptor& command,
                               const ir::physical::Plan& plan, void* arena, void*,
                               cudaStream_t stream) {
  if (command.buffers.size() < 2) return cudaErrorInvalidValue;
  const auto& source = plan.buffers()[command.buffers[0].value()];
  const auto& destination = plan.buffers()[command.buffers[1].value()];
  const std::uint64_t bytes = std::min(source.size, destination.size);
  return cudaMemcpyAsync(buffer_pointer(plan, arena, destination.id),
                         buffer_pointer(plan, arena, source.id), static_cast<std::size_t>(bytes),
                         cudaMemcpyDeviceToDevice, stream);
}

inline cudaError_t launch_residual(const ir::physical::CommandDescriptor& command,
                                   const ir::physical::Plan& plan, void* arena, void*,
                                   cudaStream_t stream) {
  if (command.buffers.size() < 3) return cudaErrorInvalidValue;
  const auto& left = plan.buffers()[command.buffers[0].value()];
  const auto& right = plan.buffers()[command.buffers[1].value()];
  const auto& output = plan.buffers()[command.buffers[2].value()];
  const std::uint64_t bytes = std::min({left.size, right.size, output.size});
  if (bytes % sizeof(float) != 0) return cudaErrorInvalidValue;
  residual_f32<<<1, 256, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, left.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, right.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)),
      static_cast<std::size_t>(bytes / sizeof(float)));
  return cudaGetLastError();
}

inline cudaError_t launch_rms_norm(const ir::physical::CommandDescriptor& command,
                                   const ir::physical::Plan& plan, void* arena, void*,
                                   cudaStream_t stream) {
  if (command.buffers.size() < 2) return cudaErrorInvalidValue;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& output = plan.buffers()[command.buffers[1].value()];
  const std::uint64_t bytes = std::min(input.size, output.size);
  if (bytes == 0 || bytes % sizeof(float) != 0) return cudaErrorInvalidValue;
  rms_norm_f32<<<1, 1, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)),
      static_cast<std::size_t>(bytes / sizeof(float)));
  return cudaGetLastError();
}

inline LaunchFunction resolve(std::uint64_t kernel_id) {
  switch (kernel_id) {
    case 1: return &launch_copy;
    case 4: return &launch_residual;
    case 5: return &launch_rms_norm;
    case 2:
    case 3:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
    case 13: return &launch_stub;
    default: return nullptr;
  }
}

inline base::Status contextual(const base::Status& source, std::string_view context) {
  base::Status copy = source;
  copy.with_context(context);
  return copy;
}

}  // namespace detail

/**
 * CUDA-backed Physical Plan shell for lifecycle and scheduling qualification.
 *
 * Construction validates and binds every resource. execute() launches only prebound command
 * stubs in Physical Plan dependency order; it performs no allocation or device-wide sync. The
 * stubs deliberately provide scheduling evidence, not numerical model execution.
 */
class CudaPlanSession final {
 public:
  static base::Result<CudaPlanSession> create(const ir::physical::Plan& plan,
                                              std::uint32_t target_capability,
                                              std::string_view kernel_catalog) {
    base::Status plan_status = plan.verify();
    if (!plan_status.ok()) return detail::contextual(plan_status, "CUDA physical plan");
    if (target_capability != 120 || plan.capability().target_capability != target_capability ||
        plan.capability().kernel_catalog != kernel_catalog) {
      return base::Status::unsupported("CUDA physical plan capability does not match target");
    }

    CudaPlanSession session{plan};
    session.commands_plan_ = plan.commands();
    session.launchers_.reserve(plan.commands().size());
    for (const ir::physical::CommandDescriptor& command : plan.commands()) {
      if (command.kernel.value() == 0) {
        return base::Status::failed_precondition("CUDA command has no stable kernel ID");
      }
      const detail::LaunchFunction launcher = detail::resolve(command.kernel.value());
      if (launcher == nullptr) return base::Status::unsupported("CUDA kernel ID is not registered");
      session.launchers_.push_back(launcher);
    }
    auto device_arena = DeviceBuffer::allocate(plan.resources().arena_bytes);
    if (!device_arena.has_value()) return detail::contextual(device_arena.error(), "CUDA device arena");
    auto workspace = DeviceBuffer::allocate(plan.resources().workspace_bytes);
    if (!workspace.has_value()) return detail::contextual(workspace.error(), "CUDA workspace arena");

    std::uint32_t stream_count = 0;
    for (const ir::physical::CommandDescriptor& command : plan.commands()) {
      if (command.kernel.value() == 0) {
        return base::Status::failed_precondition("CUDA command has no stable kernel ID");
      }
      if (command.stream == std::numeric_limits<std::uint32_t>::max()) {
        return base::Status::resource_exhausted("CUDA stream ordinal cannot be incremented");
      }
      stream_count = std::max(stream_count, command.stream + 1);
    }

    session.streams_.reserve(stream_count);
    for (std::uint32_t index = 0; index < stream_count; ++index) {
      auto stream = StreamOwner::create();
      if (!stream.has_value()) return detail::contextual(stream.error(), "CUDA stream creation");
      session.streams_.push_back(std::move(stream).value());
    }
    session.events_.reserve(plan.commands().size());
    for (std::size_t index = 0; index < plan.commands().size(); ++index) {
      auto event = EventOwner::create();
      if (!event.has_value()) return detail::contextual(event.error(), "CUDA event creation");
      session.events_.push_back(std::move(event).value());
    }

    std::vector<bool> emitted(plan.commands().size(), false);
    session.command_order_.reserve(plan.commands().size());
    for (std::size_t rank = 0; rank < plan.commands().size(); ++rank) {
      bool found = false;
      for (std::size_t index = 0; index < plan.commands().size(); ++index) {
        if (emitted[index]) continue;
        bool dependencies_emitted = true;
        for (const ir::physical::CommandId dependency : plan.commands()[index].dependencies) {
          if (!emitted[dependency.value()]) {
            dependencies_emitted = false;
            break;
          }
        }
        if (!dependencies_emitted) continue;
        emitted[index] = true;
        session.command_order_.push_back(index);
        found = true;
        break;
      }
      if (!found) return base::Status::failed_precondition("CUDA command schedule is not executable");
    }
    session.device_arena_ = std::move(device_arena).value();
    session.workspace_ = std::move(workspace).value();
    return session;
  }

  CudaPlanSession(CudaPlanSession&&) noexcept = default;
  CudaPlanSession& operator=(CudaPlanSession&&) noexcept = default;
  CudaPlanSession(const CudaPlanSession&) = delete;
  CudaPlanSession& operator=(const CudaPlanSession&) = delete;

  base::Status execute() noexcept {
    if (poisoned_) return base::Status::failed_precondition("CUDA session is poisoned");
    for (const std::size_t command_index : command_order_) {
      const ir::physical::CommandDescriptor& command = commands_plan_[command_index];
      cudaStream_t stream = streams_[command.stream].get();
      for (const ir::physical::CommandId dependency : command.dependencies) {
        const cudaError_t wait_error = cudaStreamWaitEvent(stream, events_[dependency.value()].get(), 0);
        if (wait_error != cudaSuccess) return poison(wait_error, "cudaStreamWaitEvent");
      }
      const cudaError_t launch_error = launchers_[command_index](
          command, plan_, device_arena_.data(), workspace_.data(), stream);
      if (launch_error != cudaSuccess) return poison(launch_error, "baseline command launch");
      const cudaError_t record_error = cudaEventRecord(events_[command_index].get(), stream);
      if (record_error != cudaSuccess) return poison(record_error, "cudaEventRecord");
      ++trace_.commands_executed;
      ++trace_.launches;
    }
    return {};
  }

  /** Explicit test/profiling synchronization; never called by execute(). */
  base::Status synchronize_for_test() noexcept {
    return cuda_status(cudaDeviceSynchronize(), "explicit test synchronization");
  }

  base::Status copy_to_device(ir::physical::BufferId id, base::ConstByteView source) noexcept {
    const auto validation = validate_copy(id, source.size());
    if (!validation.ok()) return validation;
    const auto& buffer = plan_.buffers()[id.value()];
    return cuda_status(cudaMemcpy(detail::buffer_pointer(plan_, device_arena_.data(), buffer.id), source.data(),
                                  source.size(), cudaMemcpyHostToDevice),
                       "host-to-device copy");
  }

  base::Status copy_from_device(ir::physical::BufferId id, base::ByteView destination) noexcept {
    const auto validation = validate_copy(id, destination.size());
    if (!validation.ok()) return validation;
    const auto& buffer = plan_.buffers()[id.value()];
    return cuda_status(cudaMemcpy(destination.data(),
                                  detail::buffer_pointer(plan_, device_arena_.data(), buffer.id),
                                  destination.size(), cudaMemcpyDeviceToHost),
                       "device-to-host copy");
  }

  [[nodiscard]] const CudaExecutionTrace& trace() const noexcept { return trace_; }
  [[nodiscard]] std::uint64_t device_arena_bytes() const noexcept { return device_arena_.bytes(); }
  [[nodiscard]] std::uint64_t workspace_bytes() const noexcept { return workspace_.bytes(); }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }

 private:
  explicit CudaPlanSession(const ir::physical::Plan& plan) : plan_(plan) {}

  base::Status validate_copy(ir::physical::BufferId id, std::size_t bytes) const noexcept {
    if (id.value() >= plan_.buffers().size()) return base::Status::out_of_range("CUDA buffer is undefined");
    if (bytes > plan_.buffers()[id.value()].size) {
      return base::Status::out_of_range("host copy exceeds physical buffer");
    }
    return {};
  }

  base::Status poison(cudaError_t error, std::string_view context) noexcept {
    poisoned_ = true;
    return cuda_status(error, context);
  }

  ir::physical::Plan plan_;
  DeviceBuffer device_arena_;
  DeviceBuffer workspace_;
  std::vector<StreamOwner> streams_;
  std::vector<EventOwner> events_;
  std::vector<std::size_t> command_order_;
  std::vector<ir::physical::CommandDescriptor> commands_plan_;
  std::vector<detail::LaunchFunction> launchers_;
  CudaExecutionTrace trace_{};
  bool poisoned_{false};
};

}  // namespace superinfer::sm120::cuda_runtime
