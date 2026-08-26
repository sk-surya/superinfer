#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <superinfer/base/memory_space.hpp>
#include <superinfer/base/result.hpp>
#include <superinfer/base/views.hpp>
#include <superinfer/runtime/plan_binding.hpp>

namespace superinfer::runtime {

enum class BackendKind { host_reference, cuda };

struct SessionOptions final {
  std::uint32_t target_capability{0};
  std::string_view kernel_catalog;
  BackendKind backend{BackendKind::host_reference};
};

/** Move-only owner for one runtime arena allocation.
 *
 * The host-reference backend owns host bytes so CPU CI can exercise lifetime and bounds rules.
 * CUDA materialization is a separate backend and returns unavailable until a CUDA target is built.
 */
class DeviceBuffer final {
 public:
  static base::Result<DeviceBuffer> allocate(std::uint64_t bytes, base::MemorySpace space) {
    if (space != base::MemorySpace::host) {
      return base::Status::unavailable("requested memory space has no compiled runtime backend");
    }
    if (bytes > std::numeric_limits<std::size_t>::max()) {
      return base::Status::overflow("runtime arena size does not fit host size_t");
    }
    std::unique_ptr<std::byte[]> storage;
    if (bytes != 0) {
      storage.reset(new (std::nothrow) std::byte[static_cast<std::size_t>(bytes)]);
      if (!storage) return base::Status::resource_exhausted("runtime arena allocation failed");
    }
    return DeviceBuffer{space, bytes, std::move(storage)};
  }

  DeviceBuffer(DeviceBuffer&&) noexcept = default;
  DeviceBuffer& operator=(DeviceBuffer&&) noexcept = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] base::MemorySpace memory_space() const noexcept { return space_; }
  [[nodiscard]] std::byte* data() noexcept { return storage_.get(); }
  [[nodiscard]] const std::byte* data() const noexcept { return storage_.get(); }

 private:
  DeviceBuffer(base::MemorySpace space, std::uint64_t bytes, std::unique_ptr<std::byte[]> storage)
      : space_(space), bytes_(bytes), storage_(std::move(storage)) {}

  base::MemorySpace space_{base::MemorySpace::host};
  std::uint64_t bytes_{0};
  std::unique_ptr<std::byte[]> storage_;
};

/** Move-only owner for a logical execution stream established during session construction. */
class StreamOwner final {
 public:
  static base::Result<StreamOwner> create(std::uint32_t ordinal) {
    return StreamOwner{ordinal};
  }

  StreamOwner(StreamOwner&&) noexcept = default;
  StreamOwner& operator=(StreamOwner&&) noexcept = default;
  StreamOwner(const StreamOwner&) = delete;
  StreamOwner& operator=(const StreamOwner&) = delete;

  [[nodiscard]] std::uint32_t ordinal() const noexcept { return ordinal_; }

 private:
  explicit StreamOwner(std::uint32_t ordinal) : ordinal_(ordinal) {}
  std::uint32_t ordinal_{0};
};

/** Move-only owner for one explicit dependency event. */
class EventOwner final {
 public:
  static base::Result<EventOwner> create(std::uint32_t ordinal) {
    return EventOwner{ordinal};
  }

  EventOwner(EventOwner&&) noexcept = default;
  EventOwner& operator=(EventOwner&&) noexcept = default;
  EventOwner(const EventOwner&) = delete;
  EventOwner& operator=(const EventOwner&) = delete;

  [[nodiscard]] std::uint32_t ordinal() const noexcept { return ordinal_; }

 private:
  explicit EventOwner(std::uint32_t ordinal) : ordinal_(ordinal) {}
  std::uint32_t ordinal_{0};
};

/** Immutable runtime catalog binding produced before the first execution. */
class ModuleRegistry final {
 public:
  static base::Result<ModuleRegistry> create(std::string_view catalog) {
    if (catalog.empty()) return base::Status::invalid_argument("runtime kernel catalog is empty");
    return ModuleRegistry{catalog};
  }

  ModuleRegistry(ModuleRegistry&&) noexcept = default;
  ModuleRegistry& operator=(ModuleRegistry&&) noexcept = default;
  ModuleRegistry(const ModuleRegistry&) = delete;
  ModuleRegistry& operator=(const ModuleRegistry&) = delete;

  [[nodiscard]] std::string_view catalog() const noexcept { return catalog_; }

 private:
  explicit ModuleRegistry(std::string_view catalog) : catalog_(catalog) {}
  std::string catalog_;
};

/** Stable kernel-ID registry; selection has already happened in the Physical Plan. */
class KernelRegistry final {
 public:
  static base::Result<KernelRegistry> bind(const ir::physical::Plan& plan) {
    KernelRegistry registry;
    registry.kernel_ids_.reserve(plan.commands().size());
    for (const ir::physical::CommandDescriptor& command : plan.commands()) {
      if (command.kernel.value() == 0) {
        return base::Status::failed_precondition("physical command has no stable kernel ID");
      }
      registry.kernel_ids_.push_back(command.kernel);
    }
    return registry;
  }

  KernelRegistry(KernelRegistry&&) noexcept = default;
  KernelRegistry& operator=(KernelRegistry&&) noexcept = default;
  KernelRegistry(const KernelRegistry&) = delete;
  KernelRegistry& operator=(const KernelRegistry&) = delete;

  [[nodiscard]] std::size_t size() const noexcept { return kernel_ids_.size(); }

 private:
  KernelRegistry() = default;
  std::vector<base::KernelId> kernel_ids_;
};

/**
 * Owns all resources required by one validated Physical Plan.
 *
 * Construction performs validation, catalog/kernel binding, and every arena/stream/event
 * allocation. execute() only forwards to the prebound command sequence and never selects policy,
 * reads model metadata, touches the filesystem, or allocates.
 */
class RuntimeSession final {
 public:
  static base::Result<RuntimeSession> create(const ir::physical::Plan& plan,
                                             SessionOptions options) {
    if (options.backend == BackendKind::cuda) {
      return base::Status::unavailable("CUDA runtime backend is not compiled in this build");
    }
    base::Status plan_status = plan.verify();
    if (!plan_status.ok()) return plan_status.with_context("runtime session physical plan");
    if (plan.capability().target_capability != options.target_capability ||
        plan.capability().kernel_catalog != options.kernel_catalog) {
      return base::Status::unsupported("physical plan capability does not match session options");
    }

    auto device_arena = DeviceBuffer::allocate(plan.resources().arena_bytes, base::MemorySpace::host);
    if (!device_arena.has_value()) return contextual(device_arena.error(), "device arena");
    auto workspace = DeviceBuffer::allocate(plan.resources().workspace_bytes, base::MemorySpace::host);
    if (!workspace.has_value()) return contextual(workspace.error(), "workspace arena");
    auto module = ModuleRegistry::create(options.kernel_catalog);
    if (!module.has_value()) return contextual(module.error(), "runtime module registry");
    auto kernels = KernelRegistry::bind(plan);
    if (!kernels.has_value()) return contextual(kernels.error(), "runtime kernel registry");
    auto binding = PlanBinding::create(plan, options.target_capability, options.kernel_catalog);
    if (!binding.has_value()) return contextual(binding.error(), "runtime plan binding");

    std::uint32_t stream_count = 0;
    for (const ir::physical::CommandDescriptor& command : plan.commands()) {
      if (command.stream == std::numeric_limits<std::uint32_t>::max()) {
        return base::Status::resource_exhausted("physical stream ordinal cannot be incremented");
      }
      stream_count = std::max(stream_count, command.stream + 1);
    }
    std::vector<StreamOwner> streams;
    streams.reserve(stream_count);
    for (std::uint32_t ordinal = 0; ordinal < stream_count; ++ordinal) {
      auto stream = StreamOwner::create(ordinal);
      if (!stream.has_value()) return contextual(stream.error(), "runtime stream registry");
      streams.push_back(std::move(stream).value());
    }

    std::vector<EventOwner> events;
    events.reserve(plan.commands().size());
    for (const ir::physical::CommandDescriptor& command : plan.commands()) {
      if (command.dependencies.empty()) continue;
      auto event = EventOwner::create(static_cast<std::uint32_t>(events.size()));
      if (!event.has_value()) return contextual(event.error(), "runtime event registry");
      events.push_back(std::move(event).value());
    }
    return RuntimeSession{std::move(device_arena).value(), std::move(workspace).value(),
                          std::move(module).value(), std::move(kernels).value(),
                          std::move(binding).value(), std::move(streams), std::move(events)};
  }

  RuntimeSession(RuntimeSession&&) noexcept = default;
  RuntimeSession& operator=(RuntimeSession&&) noexcept = default;
  RuntimeSession(const RuntimeSession&) = delete;
  RuntimeSession& operator=(const RuntimeSession&) = delete;

  /** Executes the immutable plan; all failure state is retained for safe teardown. */
  base::Status execute() noexcept {
    if (poisoned_) return base::Status::failed_precondition("runtime session is poisoned");
    const base::Status status = binding_.execute();
    if (!status.ok()) poisoned_ = true;
    return status;
  }

  [[nodiscard]] std::uint64_t device_arena_bytes() const noexcept { return device_arena_.bytes(); }
  [[nodiscard]] std::uint64_t workspace_bytes() const noexcept { return workspace_.bytes(); }
  [[nodiscard]] std::size_t stream_count() const noexcept { return streams_.size(); }
  [[nodiscard]] std::size_t event_count() const noexcept { return events_.size(); }
  [[nodiscard]] const ExecutionTrace& trace() const noexcept { return binding_.trace(); }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_ || binding_.poisoned(); }

 private:
  static base::Status contextual(const base::Status& source, std::string_view context) {
    base::Status copy = source;
    copy.with_context(context);
    return copy;
  }

  RuntimeSession(DeviceBuffer device_arena, DeviceBuffer workspace, ModuleRegistry module,
                 KernelRegistry kernels, PlanBinding binding, std::vector<StreamOwner> streams,
                 std::vector<EventOwner> events)
      : device_arena_(std::move(device_arena)),
        workspace_(std::move(workspace)),
        module_(std::move(module)),
        kernels_(std::move(kernels)),
        binding_(std::move(binding)),
        streams_(std::move(streams)),
        events_(std::move(events)) {}

  DeviceBuffer device_arena_;
  DeviceBuffer workspace_;
  ModuleRegistry module_;
  KernelRegistry kernels_;
  PlanBinding binding_;
  std::vector<StreamOwner> streams_;
  std::vector<EventOwner> events_;
  bool poisoned_{false};
};

}  // namespace superinfer::runtime
