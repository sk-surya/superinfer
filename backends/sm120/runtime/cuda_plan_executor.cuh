#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include <sm120/runtime/cuda_ownership.cuh>
#include <superinfer/ir/physical_plan.hpp>

namespace superinfer::sm120::cuda_runtime {

struct CudaExecutionTrace final {
  std::uint64_t commands_executed{0};
  std::uint64_t launches{0};
};

namespace detail {

__global__ inline void baseline_command_stub() {}

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

    CudaPlanSession session;
    session.commands_plan_ = plan.commands();
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
      detail::baseline_command_stub<<<1, 1, 0, stream>>>();
      const cudaError_t launch_error = cudaGetLastError();
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

  [[nodiscard]] const CudaExecutionTrace& trace() const noexcept { return trace_; }
  [[nodiscard]] std::uint64_t device_arena_bytes() const noexcept { return device_arena_.bytes(); }
  [[nodiscard]] std::uint64_t workspace_bytes() const noexcept { return workspace_.bytes(); }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }

 private:
  CudaPlanSession() = default;

  base::Status poison(cudaError_t error, std::string_view context) noexcept {
    poisoned_ = true;
    return cuda_status(error, context);
  }

  DeviceBuffer device_arena_;
  DeviceBuffer workspace_;
  std::vector<StreamOwner> streams_;
  std::vector<EventOwner> events_;
  std::vector<std::size_t> command_order_;
  std::vector<ir::physical::CommandDescriptor> commands_plan_;
  CudaExecutionTrace trace_{};
  bool poisoned_{false};
};

}  // namespace superinfer::sm120::cuda_runtime
