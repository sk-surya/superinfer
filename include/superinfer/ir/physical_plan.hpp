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

enum class PhysicalDType { unknown, f32, f16, bf16, int8, int32, u8 };
enum class PhysicalLayout { row_major, column_major, blocked };
enum class StorageEncoding { none, nvfp4_packed, fp8_e4m3_group_scale };

/** Fully typed physical view metadata used to validate kernel operands. */
struct PhysicalTensorDescriptor final {
  PhysicalDType dtype{PhysicalDType::unknown};
  std::vector<std::uint64_t> shape;
  PhysicalLayout layout{PhysicalLayout::row_major};
  std::uint64_t alignment{0};
  StorageEncoding encoding{StorageEncoding::none};
};

/** A command operand binds a typed logical view to one physical allocation. */
struct PhysicalOperandDescriptor final {
  BufferId buffer;
  PhysicalDType dtype{PhysicalDType::unknown};
  std::vector<std::uint64_t> shape;
  PhysicalLayout layout{PhysicalLayout::row_major};
  std::uint64_t alignment{0};
  StorageEncoding encoding{StorageEncoding::none};
};

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
  PhysicalTensorDescriptor tensor;
};

/** Compile-time dimensions required by physical attention commands. Zero means not applicable. */
struct AttentionDimensions final {
  std::uint32_t query_heads{0};
  std::uint32_t key_value_heads{0};
  std::uint32_t head_dimension{0};
  std::uint32_t positions{0};
  std::uint32_t value_heads{0};
  std::uint32_t value_dimension{0};
};

struct CommandDescriptor final {
  CommandId id;
  base::KernelId kernel;
  std::vector<BufferId> buffers;
  std::vector<PhysicalOperandDescriptor> operands;
  std::vector<CommandId> dependencies;
  std::uint32_t stream;
  std::uint64_t workspace_offset;
  std::uint64_t workspace_size;
  float epsilon{1.0e-5F};
  float scalar{1.0F};
  AttentionDimensions attention{};
  bool add_one_to_scale{false};
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
      if (buffer.tensor.dtype == PhysicalDType::unknown || buffer.tensor.shape.empty() ||
          buffer.tensor.alignment == 0 || buffer.tensor.alignment != buffer.alignment) {
        return base::Status::invalid_argument("physical buffer tensor descriptor is incomplete");
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
      if (command.operands.size() != command.buffers.size()) {
        return base::Status::invalid_argument("physical command operand descriptors do not match buffers");
      }
      for (BufferId buffer : command.buffers) {
        if (buffer.value() >= buffers_.size()) {
          return base::Status::out_of_range("physical command references undefined buffer");
        }
      }
      for (std::size_t operand_index = 0; operand_index < command.operands.size(); ++operand_index) {
        const PhysicalOperandDescriptor& operand = command.operands[operand_index];
        const BufferDescriptor& buffer = buffers_[operand.buffer.value()];
        if (operand.buffer != command.buffers[operand_index] || operand.dtype == PhysicalDType::unknown ||
            operand.shape.empty() || operand.alignment == 0 || operand.alignment != buffer.alignment ||
            operand.dtype != buffer.tensor.dtype || operand.shape != buffer.tensor.shape ||
            operand.layout != buffer.tensor.layout || operand.encoding != buffer.tensor.encoding) {
          return base::Status::failed_precondition("physical operand type does not match its buffer");
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
             << command.workspace_size;
      if (command.attention.query_heads != 0 || command.attention.key_value_heads != 0 ||
          command.attention.head_dimension != 0 || command.attention.positions != 0) {
        output << " attention=" << command.attention.query_heads << "x"
               << command.attention.key_value_heads << "x" << command.attention.head_dimension
               << "@" << command.attention.positions;
      }
      output << "\n";
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
                                    std::uint64_t alignment,
                                    PhysicalTensorDescriptor tensor = {}) {
    if (tensor.dtype == PhysicalDType::unknown) {
      tensor.dtype = size % 4 == 0 ? PhysicalDType::f32 : PhysicalDType::u8;
      tensor.shape = {size / (tensor.dtype == PhysicalDType::f32 ? 4U : 1U)};
    }
    if (tensor.shape.empty()) tensor.shape = {size};
    if (tensor.alignment == 0) tensor.alignment = alignment;
    buffers_.push_back({BufferId{buffers_.size()}, offset, size, alignment, std::move(tensor)});
    return buffers_.back().id;
  }

  base::Result<CommandId> add_command(base::KernelId kernel, std::vector<BufferId> buffers,
                                      std::vector<CommandId> dependencies,
                                      std::uint32_t stream, std::uint64_t workspace_offset,
                                      std::uint64_t workspace_size, float epsilon = 1.0e-5F,
                                      float scalar = 1.0F,
                                      AttentionDimensions attention = {},
                                      bool add_one_to_scale = false,
                                      std::vector<PhysicalOperandDescriptor> operands = {}) {
    if (operands.empty()) {
      operands.reserve(buffers.size());
      for (const BufferId buffer : buffers) {
        if (buffer.value() >= buffers_.size()) {
          operands.push_back(PhysicalOperandDescriptor{buffer, PhysicalDType::unknown, {},
                                                       PhysicalLayout::row_major, 0,
                                                       StorageEncoding::none});
          continue;
        }
        const PhysicalTensorDescriptor& tensor = buffers_[buffer.value()].tensor;
        operands.push_back({buffer, tensor.dtype, tensor.shape, tensor.layout, tensor.alignment,
                            tensor.encoding});
      }
    }
    commands_.push_back({CommandId{commands_.size()}, kernel, std::move(buffers),
                         std::move(operands),
                         std::move(dependencies), stream, workspace_offset, workspace_size,
                         epsilon, scalar, attention, add_one_to_scale});
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
