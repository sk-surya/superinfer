#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include <superinfer/base/checked_math.hpp>
#include <superinfer/compiler/memory_planner.h>
#include <superinfer/compiler/target.h>
#include <superinfer/ir/lowered/module.hpp>
#include <superinfer/ir/physical_plan.hpp>
#include <superinfer/kernels/kernel_provider.hpp>

namespace superinfer::sm120 {

struct CompileOptions final {
  compiler::TargetProfile target;
  std::uint64_t max_device_memory_bytes{0};
  std::uint32_t max_commands{0};
};

struct SpecializationResult final {
  compiler::MemoryPlan memory;
  ir::physical::Plan plan;
};

/**
 * Converts verified Lowered IR into a deterministic, resource-checked Physical Plan.
 *
 * This compiler owns no device resources and is safe to run offline. It selects only stable
 * capability IDs from the baseline catalog; model frontends and names never enter this boundary.
 */
class Specializer final {
 public:
  [[nodiscard]] base::Result<SpecializationResult> compile(
      const ir::lowered::Module& lowered, const CompileOptions& options,
      const kernels::KernelProvider& provider) const {
    base::Status target_status = options.target.validate();
    if (!target_status.ok()) return target_status.with_context("sm120 target");
    base::Status lowered_status = lowered.verify();
    if (!lowered_status.ok()) return lowered_status.with_context("lowered module");
    if (options.max_commands != 0 && lowered.kernel_requirements().size() > options.max_commands) {
      return base::Status::resource_exhausted("lowered kernel command count exceeds target limit");
    }

    const std::uint64_t memory_budget =
        options.max_device_memory_bytes == 0
            ? options.target.device_memory_bytes
            : std::min(options.target.device_memory_bytes, options.max_device_memory_bytes);
    compiler::MemoryPlanner planner{memory_budget, 0};
    std::vector<compiler::AllocationRequest> requests;
    requests.reserve(lowered.tensors().size());
    for (const ir::lowered::Tensor& tensor : lowered.tensors()) {
      const auto bytes = tensor_bytes(tensor);
      if (!bytes.has_value()) {
        base::Status error = bytes.error();
        return error.with_context("lowered tensor bytes");
      }
      requests.push_back({tensor.id.value(), "tensor_" + std::to_string(tensor.id.value()),
                          compiler::ArenaKind::device, bytes.value(),
                          std::max(options.target.required_alignment, tensor.alignment),
                          {0, std::max<std::uint64_t>(1, lowered.kernel_requirements().size())},
                          allocation_class(tensor.role)});
    }
    const auto memory = planner.plan(requests);
    if (!memory.has_value()) {
      base::Status error = memory.error();
      return error.with_context("sm120 memory plan");
    }

    ir::physical::PlanBuilder plan_builder;
    plan_builder.set_resource_bounds(
        {memory.value().device_arena_bytes, memory.value().workspace_bytes, options.max_commands});
    std::vector<ir::physical::BufferId> buffer_for_tensor(lowered.tensors().size(),
                                                           ir::physical::BufferId{std::numeric_limits<std::uint64_t>::max()});
    for (const compiler::Allocation& allocation : memory.value().allocations) {
      const auto buffer = plan_builder.add_buffer(allocation.offset, allocation.bytes, allocation.alignment);
      if (!buffer.has_value()) {
        base::Status error = buffer.error();
        return error.with_context("physical buffer");
      }
      if (allocation.id >= buffer_for_tensor.size()) {
        return base::Status::out_of_range("memory plan allocation is not a lowered tensor");
      }
      buffer_for_tensor[allocation.id] = buffer.value();
    }

    std::vector<ir::physical::CommandId> dependencies;
    std::uint64_t workspace_bytes = 0;
    for (const ir::lowered::KernelRequirement& requirement : lowered.kernel_requirements()) {
      if (requirement.target_capability != options.target.compute_capability) {
        return base::Status::unsupported("lowered kernel requirement targets an incompatible capability");
      }
      const auto candidate = select_candidate(provider, requirement, lowered);
      if (!candidate.has_value()) {
        base::Status error = candidate.error();
        return error.with_context(requirement.operation);
      }
      if (requirement.operands.empty()) {
        return base::Status::failed_precondition(
            "kernel requirement must declare explicit tensor operands");
      }
      std::vector<ir::physical::BufferId> operands;
      operands.reserve(requirement.operands.size());
      for (const ir::lowered::LoweredTensorId operand : requirement.operands) {
        if (operand.value() >= buffer_for_tensor.size() ||
            buffer_for_tensor[operand.value()].value() == std::numeric_limits<std::uint64_t>::max()) {
          return base::Status::failed_precondition("kernel operand has no physical allocation");
        }
        operands.push_back(buffer_for_tensor[operand.value()]);
      }
      if (requirement.operation == "rms_norm" && operands.size() == 3) {
        std::swap(operands[1], operands[2]);
      } else if (requirement.operation == "layer_norm" && operands.size() == 4) {
        std::swap(operands[1], operands[3]);
        std::swap(operands[2], operands[3]);
      }
      ir::physical::AttentionDimensions attention{};
      if (requirement.operation == "attention") {
        if (requirement.operands.size() < 4 || requirement.attributes.num_heads == 0 ||
            requirement.attributes.num_kv_heads == 0 || requirement.attributes.head_dimension == 0) {
          return base::Status::invalid_argument("attention requirement lacks authored dimensions");
        }
        const auto& key_tensor = lowered.tensors()[requirement.operands[1].value()];
        if (key_tensor.physical_shape.size() != 3 || key_tensor.physical_shape[0] == 0 ||
            key_tensor.physical_shape[0] > std::numeric_limits<std::uint32_t>::max()) {
          return base::Status::invalid_argument("attention key tensor lacks a static position dimension");
        }
        attention = {requirement.attributes.num_heads, requirement.attributes.num_kv_heads,
                     requirement.attributes.head_dimension,
                     static_cast<std::uint32_t>(key_tensor.physical_shape[0])};
      }
      workspace_bytes = std::max(workspace_bytes, candidate.value().workspace_bytes);
      const auto command = plan_builder.add_command(
          candidate.value().id, std::move(operands), dependencies, 0, 0,
          candidate.value().workspace_bytes, 1.0e-5F, 1.0F, attention);
      if (!command.has_value()) {
        base::Status error = command.error();
        return error.with_context("physical command");
      }
      dependencies = {command.value()};
    }
    plan_builder.set_resource_bounds(
        {memory.value().device_arena_bytes, workspace_bytes, options.max_commands});
    const auto plan = std::move(plan_builder).finalize(
        {options.target.compute_capability, options.target.kernel_catalog});
    if (!plan.has_value()) {
      base::Status error = plan.error();
      return error.with_context("sm120 physical plan");
    }
    return SpecializationResult{std::move(memory).value(), std::move(plan).value()};
  }

 private:
  static compiler::AllocationClass allocation_class(ir::semantic::TensorRole role) noexcept {
    switch (role) {
      case ir::semantic::TensorRole::weight: return compiler::AllocationClass::persistent_weight;
      case ir::semantic::TensorRole::kv_cache: return compiler::AllocationClass::kv_state;
      case ir::semantic::TensorRole::decode_state: return compiler::AllocationClass::decode_state;
      case ir::semantic::TensorRole::activation:
      case ir::semantic::TensorRole::logits: return compiler::AllocationClass::activation;
    }
    return compiler::AllocationClass::activation;
  }

  static base::Result<std::uint64_t> tensor_bytes(const ir::lowered::Tensor& tensor) {
    std::uint64_t elements = 1;
    for (const std::uint64_t dimension : tensor.physical_shape) {
      const auto product = base::checked_mul(elements, dimension);
      if (!product.has_value()) return product.error();
      elements = product.value();
    }
    switch (tensor.storage_dtype) {
      case ir::semantic::DType::f32:
        return base::checked_mul(elements, 4);
      case ir::semantic::DType::f16:
      case ir::semantic::DType::bf16:
        return base::checked_mul(elements, 2);
      case ir::semantic::DType::int8:
        return elements;
      case ir::semantic::DType::int32:
        return base::checked_mul(elements, 4);
      case ir::semantic::DType::int4:
        {
          const auto rounded = base::checked_add(elements, 1);
          if (!rounded.has_value()) return rounded.error();
          return rounded.value() / 2;
        }
    }
    return base::Status::unsupported("lowered tensor dtype is unsupported by sm120 baseline");
  }

  struct SelectedKernel final {
    base::KernelId id;
    std::uint64_t workspace_bytes;
  };

  static base::Result<SelectedKernel> select_candidate(
      const kernels::KernelProvider& provider, const ir::lowered::KernelRequirement& requirement,
      const ir::lowered::Module& lowered) {
    std::string_view storage_dtype = "f32";
    if ((requirement.operation == "embedding" || requirement.operation == "lm_head" ||
         requirement.operation == "gated_dense_ffn" || requirement.operation == "rms_norm" ||
         requirement.operation == "layer_norm") && requirement.operands.size() >= 2) {
      storage_dtype = dtype_name(lowered.tensors()[requirement.operands[1].value()].storage_dtype);
    }
    const auto candidates = provider.enumerate(
        {requirement.operation, requirement.target_capability, storage_dtype});
    if (!candidates.has_value()) return candidates.error();
    if (candidates.value().empty()) {
      return base::Status::unsupported("kernel provider returned no candidates");
    }
    const kernels::KernelCandidate* selected = nullptr;
    for (const kernels::KernelCandidate& candidate : candidates.value()) {
      if (candidate.id.value() == 0 || !candidate.deterministic) continue;
      if (selected == nullptr || candidate.workspace_bytes < selected->workspace_bytes ||
          (candidate.workspace_bytes == selected->workspace_bytes &&
           candidate.id.value() < selected->id.value())) {
        selected = &candidate;
      }
    }
    if (selected == nullptr) {
      return base::Status::failed_precondition("kernel provider has no deterministic usable candidate");
    }
    return SelectedKernel{selected->id, selected->workspace_bytes};
  }

  static std::string_view dtype_name(ir::semantic::DType dtype) noexcept {
    switch (dtype) {
      case ir::semantic::DType::f32: return "f32";
      case ir::semantic::DType::f16: return "f16";
      case ir::semantic::DType::bf16: return "bf16";
      case ir::semantic::DType::int8: return "int8";
      case ir::semantic::DType::int32: return "int32";
      case ir::semantic::DType::int4: return "int4";
    }
    return "unknown";
  }
};

}  // namespace superinfer::sm120
