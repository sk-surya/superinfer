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

__global__ inline void residual_f32(const float* left, const float* right, float* output,
                                    std::size_t elements) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += blockDim.x * gridDim.x) {
    output[index] = left[index] + right[index];
  }
}

__global__ inline void rms_norm_f32(const float* input, const float* scale, float* output,
                                    std::size_t elements, float epsilon) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  float sum_squares = 0.0F;
  for (std::size_t index = 0; index < elements; ++index) sum_squares += input[index] * input[index];
  const float denominator = sqrtf(sum_squares / static_cast<float>(elements) + epsilon);
  for (std::size_t index = 0; index < elements; ++index) {
    output[index] = input[index] / denominator * scale[index];
  }
}

__global__ inline void layer_norm_f32(const float* input, const float* scale, const float* bias,
                                      float* output, std::size_t elements, float epsilon) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  float mean = 0.0F;
  for (std::size_t index = 0; index < elements; ++index) mean += input[index];
  mean /= static_cast<float>(elements);
  float variance = 0.0F;
  for (std::size_t index = 0; index < elements; ++index) {
    const float centered = input[index] - mean;
    variance += centered * centered;
  }
  const float denominator = sqrtf(variance / static_cast<float>(elements) + epsilon);
  for (std::size_t index = 0; index < elements; ++index) {
    output[index] = (input[index] - mean) / denominator * scale[index] + bias[index];
  }
}

using LaunchFunction = cudaError_t (*)(const ir::physical::CommandDescriptor&,
                                       const ir::physical::Plan&, void*, void*, cudaStream_t);

inline void* buffer_pointer(const ir::physical::Plan& plan, void* arena,
                            ir::physical::BufferId id) {
  return static_cast<std::byte*>(arena) + static_cast<std::size_t>(plan.buffers()[id.value()].offset);
}

inline cudaError_t launch_copy(const ir::physical::CommandDescriptor& command,
                               const ir::physical::Plan& plan, void* arena, void*,
                               cudaStream_t stream) {
  if (command.buffers.size() != 2) return cudaErrorInvalidValue;
  const auto& source = plan.buffers()[command.buffers[0].value()];
  const auto& destination = plan.buffers()[command.buffers[1].value()];
  const std::uint64_t bytes = source.size;
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
  const std::uint64_t bytes = left.size;
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
  if (command.buffers.size() < 3) return cudaErrorInvalidValue;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& output = plan.buffers()[command.buffers[1].value()];
  const auto& scale = plan.buffers()[command.buffers[2].value()];
  const std::uint64_t bytes = input.size;
  if (bytes == 0 || bytes % sizeof(float) != 0) return cudaErrorInvalidValue;
  rms_norm_f32<<<1, 1, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, scale.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)),
      static_cast<std::size_t>(bytes / sizeof(float)), command.epsilon);
  return cudaGetLastError();
}

inline cudaError_t launch_layer_norm(const ir::physical::CommandDescriptor& command,
                                     const ir::physical::Plan& plan, void* arena, void*,
                                     cudaStream_t stream) {
  if (command.buffers.size() < 4) return cudaErrorInvalidValue;
  const auto& input = plan.buffers()[command.buffers[0].value()];
  const auto& output = plan.buffers()[command.buffers[1].value()];
  const auto& scale = plan.buffers()[command.buffers[2].value()];
  const auto& bias = plan.buffers()[command.buffers[3].value()];
  const std::uint64_t bytes = input.size;
  if (bytes == 0 || bytes % sizeof(float) != 0) return cudaErrorInvalidValue;
  layer_norm_f32<<<1, 1, 0, stream>>>(
      static_cast<const float*>(buffer_pointer(plan, arena, input.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, scale.id)),
      static_cast<const float*>(buffer_pointer(plan, arena, bias.id)),
      static_cast<float*>(buffer_pointer(plan, arena, output.id)),
      static_cast<std::size_t>(bytes / sizeof(float)), command.epsilon);
  return cudaGetLastError();
}

inline base::Status validate_command(const ir::physical::CommandDescriptor& command,
                                     const ir::physical::Plan& plan) {
  const auto exact_buffers = [&](std::size_t count) {
    return command.buffers.size() == count;
  };
  const auto same_sizes = [&](std::size_t count) {
    if (!exact_buffers(count)) return false;
    const std::uint64_t expected = plan.buffers()[command.buffers.front().value()].size;
    for (std::size_t index = 1; index < count; ++index) {
      if (plan.buffers()[command.buffers[index].value()].size != expected) return false;
    }
    return expected != 0 && expected % sizeof(float) == 0;
  };
  switch (command.kernel.value()) {
    case 1:
      if (!exact_buffers(2) ||
          plan.buffers()[command.buffers[0].value()].size !=
              plan.buffers()[command.buffers[1].value()].size) {
        return base::Status::invalid_argument("CUDA copy requires two equal-sized buffers");
      }
      return {};
    case 4:
      if (!same_sizes(3)) return base::Status::invalid_argument("CUDA residual requires three equal-sized f32 buffers");
      return {};
    case 5:
      if (!same_sizes(3)) return base::Status::invalid_argument("CUDA RMSNorm requires input, output, and scale buffers");
      return {};
    case 6:
      if (!same_sizes(4)) return base::Status::invalid_argument("CUDA LayerNorm requires input, output, scale, and bias buffers");
      return {};
    default:
      return base::Status::unsupported("CUDA kernel ID is not registered");
  }
}

inline LaunchFunction resolve(std::uint64_t kernel_id) {
  switch (kernel_id) {
    case 1: return &launch_copy;
    case 4: return &launch_residual;
    case 5: return &launch_rms_norm;
    case 6: return &launch_layer_norm;
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
 * Construction validates and binds every resource. execute() launches only prebound baseline
 * functions in Physical Plan dependency order; it performs no allocation or device-wide sync.
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
    if (kernel_catalog != "baseline-v1") {
      return base::Status::unsupported("CUDA kernel catalog is not registered");
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
      return base::Status::unavailable("CUDA active device is unavailable");
    }
    cudaDeviceProp properties{};
    const cudaError_t property_error = cudaGetDeviceProperties(&properties, device);
    if (property_error != cudaSuccess) {
      return detail::contextual(cuda_status(property_error, "cudaGetDeviceProperties"),
                                "CUDA target probe");
    }
    if (properties.major != 12 || properties.minor != 0) {
      return base::Status::unsupported("active CUDA device is not sm_120a");
    }
    if (plan.resources().arena_bytes > properties.totalGlobalMem -
                                           std::min<std::size_t>(properties.totalGlobalMem,
                                                                 plan.resources().workspace_bytes)) {
      return base::Status::resource_exhausted("physical plan exceeds active CUDA device memory");
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
      if (command.workspace_size != 0) {
        return base::Status::unsupported("CUDA baseline has no command workspace contract");
      }
      const base::Status command_status = detail::validate_command(command, plan);
      if (!command_status.ok()) return command_status;
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
    if (poisoned_) return base::Status::failed_precondition("CUDA session is poisoned");
    const cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) return poison(error, "explicit test synchronization");
    return {};
  }

  base::Status copy_to_device(ir::physical::BufferId id, base::ConstByteView source) noexcept {
    if (poisoned_) return base::Status::failed_precondition("CUDA session is poisoned");
    const auto validation = validate_copy(id, source.size());
    if (!validation.ok()) return validation;
    const cudaError_t sync_error = cudaDeviceSynchronize();
    if (sync_error != cudaSuccess) return poison(sync_error, "host-to-device copy boundary");
    const auto& buffer = plan_.buffers()[id.value()];
    const cudaError_t copy_error = cudaMemcpy(detail::buffer_pointer(plan_, device_arena_.data(), buffer.id),
                                              source.data(), source.size(), cudaMemcpyHostToDevice);
    if (copy_error != cudaSuccess) return poison(copy_error, "host-to-device copy");
    return {};
  }

  base::Status copy_from_device(ir::physical::BufferId id, base::ByteView destination) noexcept {
    if (poisoned_) return base::Status::failed_precondition("CUDA session is poisoned");
    const auto validation = validate_copy(id, destination.size());
    if (!validation.ok()) return validation;
    const cudaError_t sync_error = cudaDeviceSynchronize();
    if (sync_error != cudaSuccess) return poison(sync_error, "device-to-host copy boundary");
    const auto& buffer = plan_.buffers()[id.value()];
    const cudaError_t copy_error = cudaMemcpy(destination.data(),
                                              detail::buffer_pointer(plan_, device_arena_.data(), buffer.id),
                                              destination.size(), cudaMemcpyDeviceToHost);
    if (copy_error != cudaSuccess) return poison(copy_error, "device-to-host copy");
    return {};
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
