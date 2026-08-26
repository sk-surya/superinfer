#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <superinfer/base/ids.hpp>
#include <superinfer/base/result.hpp>

namespace superinfer::ir::physical {

struct BufferIdTag;
struct CommandIdTag;
using BufferId = base::StrongId<BufferIdTag>;
using CommandId = base::StrongId<CommandIdTag>;

struct ResourceBounds final {
  std::uint64_t arena_bytes;
  std::uint64_t workspace_bytes;
  std::uint32_t max_commands;
};

struct CapabilityFingerprint final {
  std::uint32_t target_capability;
  std::string kernel_catalog;
};

struct BufferDescriptor final {
  BufferId id;
  std::uint64_t offset;
  std::uint64_t size;
  std::uint64_t alignment;
};

struct CommandDescriptor final {
  CommandId id;
  base::KernelId kernel;
  std::vector<BufferId> buffers;
  std::vector<CommandId> dependencies;
  std::uint32_t stream;
  std::uint64_t workspace_offset;
  std::uint64_t workspace_size;
  float epsilon{1.0e-5F};
  float scalar{1.0F};
};

class PlanBuilder;

/**
 * Immutable, validated physical execution contract. Runtime consumers only receive const views;
 * allocation, kernel resolution, and command construction belong to compilation/materialization.
 */
class Plan final {
 public:
  static Plan empty() noexcept { return {}; }
  [[nodiscard]] std::uint32_t version() const noexcept { return 1; }
  [[nodiscard]] const CapabilityFingerprint& capability() const noexcept { return capability_; }
  [[nodiscard]] const ResourceBounds& resources() const noexcept { return resources_; }
  [[nodiscard]] const std::vector<BufferDescriptor>& buffers() const noexcept { return buffers_; }
  [[nodiscard]] const std::vector<CommandDescriptor>& commands() const noexcept { return commands_; }

  /** Rechecks all references and resource bounds without allocating or touching a device. */
  [[nodiscard]] base::Status verify() const {
    if ((!buffers_.empty() || !commands_.empty()) &&
        (capability_.target_capability == 0 || capability_.kernel_catalog.empty())) {
      return base::Status::failed_precondition("physical plan capability fingerprint is incomplete");
    }
    if (commands_.size() > resources_.max_commands && resources_.max_commands != 0) {
      return base::Status::resource_exhausted("physical command count exceeds resource bound");
    }
    for (std::size_t index = 0; index < buffers_.size(); ++index) {
      const BufferDescriptor& buffer = buffers_[index];
      if (buffer.id.value() != index || buffer.alignment == 0 || buffer.offset % buffer.alignment != 0) {
        return base::Status::invalid_argument("physical buffer has invalid identity or alignment");
      }
      if (buffer.offset > resources_.arena_bytes ||
          buffer.size > resources_.arena_bytes - buffer.offset) {
        return base::Status::out_of_range("physical buffer exceeds arena bounds");
      }
      for (std::size_t prior = 0; prior < index; ++prior) {
        const BufferDescriptor& other = buffers_[prior];
        const bool buffer_end_overflow = buffer.offset > UINT64_MAX - buffer.size;
        const bool other_end_overflow = other.offset > UINT64_MAX - other.size;
        if (buffer_end_overflow || other_end_overflow) {
          return base::Status::overflow("physical buffer end overflows");
        }
        const bool overlap = buffer.size != 0 && other.size != 0 &&
                             buffer.offset < other.offset + other.size &&
                             other.offset < buffer.offset + buffer.size;
        if (overlap) return base::Status::failed_precondition("physical buffers overlap");
      }
    }
    for (std::size_t index = 0; index < commands_.size(); ++index) {
      const CommandDescriptor& command = commands_[index];
      if (command.id.value() != index || command.workspace_size > resources_.workspace_bytes ||
          command.workspace_offset > resources_.workspace_bytes - command.workspace_size ||
          !std::isfinite(command.epsilon) || command.epsilon <= 0.0F ||
          !std::isfinite(command.scalar)) {
        return base::Status::out_of_range("physical command workspace exceeds bounds");
      }
      for (BufferId buffer : command.buffers) {
        if (buffer.value() >= buffers_.size()) {
          return base::Status::out_of_range("physical command references undefined buffer");
        }
      }
      for (CommandId dependency : command.dependencies) {
        if (dependency.value() >= commands_.size() || dependency.value() == command.id.value()) {
          return base::Status::failed_precondition("physical command dependency is invalid");
        }
      }
    }
    std::vector<std::uint8_t> marks(commands_.size(), 0);
    std::function<bool(std::size_t)> acyclic = [&](std::size_t index) {
      if (marks[index] == 1) return false;
      if (marks[index] == 2) return true;
      marks[index] = 1;
      for (CommandId dependency : commands_[index].dependencies) {
        if (!acyclic(dependency.value())) return false;
      }
      marks[index] = 2;
      return true;
    };
    for (std::size_t index = 0; index < commands_.size(); ++index) {
      if (!acyclic(index)) return base::Status::failed_precondition("physical command dependency cycle");
    }
    return {};
  }

  [[nodiscard]] std::string dump() const {
    std::ostringstream output;
    output << "physical-plan:v1 capability=" << capability_.target_capability
           << " catalog=" << capability_.kernel_catalog << "\n";
    output << "resources arena=" << resources_.arena_bytes << " workspace="
           << resources_.workspace_bytes << " max_commands=" << resources_.max_commands << "\n";
    for (const BufferDescriptor& buffer : buffers_) {
      output << "buffer id=" << buffer.id.value() << " offset=" << buffer.offset
             << " size=" << buffer.size << " alignment=" << buffer.alignment << "\n";
    }
    for (const CommandDescriptor& command : commands_) {
      output << "command id=" << command.id.value() << " kernel=" << command.kernel.value()
             << " stream=" << command.stream << " workspace=" << command.workspace_offset << "+"
             << command.workspace_size << "\n";
    }
    return output.str();
  }

 private:
  friend class PlanBuilder;
  Plan(CapabilityFingerprint capability, ResourceBounds resources,
       std::vector<BufferDescriptor> buffers, std::vector<CommandDescriptor> commands)
      : capability_(std::move(capability)),
        resources_(resources),
        buffers_(std::move(buffers)),
        commands_(std::move(commands)) {}
  Plan() = default;

  CapabilityFingerprint capability_{};
  ResourceBounds resources_{};
  std::vector<BufferDescriptor> buffers_;
  std::vector<CommandDescriptor> commands_;
};

/** Checked builder that is the only construction path for non-empty Physical Plans. */
class PlanBuilder final {
 public:
  void set_resource_bounds(ResourceBounds resources) noexcept { resources_ = resources; }

  base::Result<BufferId> add_buffer(std::uint64_t offset, std::uint64_t size,
                                    std::uint64_t alignment) {
    buffers_.push_back({BufferId{buffers_.size()}, offset, size, alignment});
    return buffers_.back().id;
  }

  base::Result<CommandId> add_command(base::KernelId kernel, std::vector<BufferId> buffers,
                                      std::vector<CommandId> dependencies,
                                      std::uint32_t stream, std::uint64_t workspace_offset,
                                      std::uint64_t workspace_size, float epsilon = 1.0e-5F,
                                      float scalar = 1.0F) {
    commands_.push_back({CommandId{commands_.size()}, kernel, std::move(buffers),
                         std::move(dependencies), stream, workspace_offset, workspace_size,
                         epsilon, scalar});
    return commands_.back().id;
  }

  [[nodiscard]] base::Result<Plan> finalize(CapabilityFingerprint capability) && {
    Plan plan{std::move(capability), resources_, std::move(buffers_), std::move(commands_)};
    base::Status status = plan.verify();
    if (!status.ok()) return status.with_context("physical-plan builder");
    return base::Result<Plan>(std::move(plan));
  }

 private:
  ResourceBounds resources_{};
  std::vector<BufferDescriptor> buffers_;
  std::vector<CommandDescriptor> commands_;
};

}  // namespace superinfer::ir::physical
