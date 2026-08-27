#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
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
    const std::uint64_t no_alias = std::numeric_limits<std::uint64_t>::max();
    std::vector<std::uint64_t> state_alias(lowered.tensors().size(), no_alias);
    for (const ir::lowered::StateSlot& slot : lowered.state_slots()) {
      if (slot.input.value() >= lowered.tensors().size() ||
          slot.output.value() >= lowered.tensors().size()) {
        return base::Status::out_of_range("state slot tensor is undefined during physical planning");
      }
      const ir::lowered::Tensor& input = lowered.tensors()[slot.input.value()];
      const ir::lowered::Tensor& output = lowered.tensors()[slot.output.value()];
      if (input.physical_shape != output.physical_shape || input.storage_dtype != output.storage_dtype ||
          input.layout != output.layout || input.memory_space != output.memory_space) {
        return base::Status::failed_precondition(
            "state slot input and output tensors have incompatible physical contracts");
      }
      if (state_alias[slot.output.value()] != no_alias &&
          state_alias[slot.output.value()] != slot.input.value()) {
        return base::Status::failed_precondition("state output has multiple physical aliases");
      }
      state_alias[slot.output.value()] = slot.input.value();
    }
    std::vector<compiler::AllocationRequest> requests;
    requests.reserve(lowered.tensors().size());
    for (const ir::lowered::Tensor& tensor : lowered.tensors()) {
      if (state_alias[tensor.id.value()] != no_alias) continue;
      const auto bytes = tensor_bytes(tensor);
      if (!bytes.has_value()) {
        base::Status error = bytes.error();
        return error.with_context("lowered tensor bytes");
      }
      requests.push_back({tensor.id.value(), "tensor_" + std::to_string(tensor.id.value()),
                          compiler::ArenaKind::device, bytes.value(),
                          std::max(options.target.required_alignment, tensor.alignment),
                          lifetime_for(tensor, lowered),
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
      if (allocation.id >= lowered.tensors().size()) {
        return base::Status::out_of_range("memory plan allocation is not a lowered tensor");
      }
      const ir::lowered::Tensor& tensor = lowered.tensors()[allocation.id];
      const auto buffer = plan_builder.add_buffer(
          allocation.offset, allocation.bytes, allocation.alignment,
          physical_tensor_descriptor(tensor, allocation.alignment),
          {allocation.lifetime.first, allocation.lifetime.last});
      if (!buffer.has_value()) {
        base::Status error = buffer.error();
        return error.with_context("physical buffer");
      }
      buffer_for_tensor[allocation.id] = buffer.value();
    }
    for (const ir::lowered::StateSlot& slot : lowered.state_slots()) {
      if (buffer_for_tensor[slot.input.value()].value() == no_alias) {
        return base::Status::failed_precondition("state input has no physical allocation");
      }
      if (buffer_for_tensor[slot.output.value()].value() == no_alias) {
        buffer_for_tensor[slot.output.value()] = buffer_for_tensor[slot.input.value()];
      }
    }

    for (const ir::lowered::EntryPoint& entry : lowered.entry_points()) {
      std::vector<ir::physical::BufferId> inputs;
      std::vector<ir::physical::BufferId> outputs;
      inputs.reserve(entry.inputs.size());
      outputs.reserve(entry.outputs.size());
      for (const ir::lowered::LoweredTensorId id : entry.inputs) {
        if (id.value() >= buffer_for_tensor.size() ||
            buffer_for_tensor[id.value()].value() == std::numeric_limits<std::uint64_t>::max()) {
          return base::Status::failed_precondition("entry input has no physical allocation");
        }
        inputs.push_back(buffer_for_tensor[id.value()]);
      }
      for (const ir::lowered::LoweredTensorId id : entry.outputs) {
        if (id.value() >= buffer_for_tensor.size() ||
            buffer_for_tensor[id.value()].value() == std::numeric_limits<std::uint64_t>::max()) {
          return base::Status::failed_precondition("entry output has no physical allocation");
        }
        outputs.push_back(buffer_for_tensor[id.value()]);
      }
      base::Status binding = plan_builder.add_entry_point(entry.name, std::move(inputs), std::move(outputs));
      if (!binding.ok()) return binding.with_context("physical entry point");
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
      ir::physical::RopeDimensions rope{};
      ir::physical::CacheAppendDimensions cache_append{};
      ir::physical::SplitDimensions split{};
      if (requirement.operation == "attention" ||
          requirement.operation == "attention_bf16_cache") {
        if (requirement.operands.size() < 4 || requirement.attributes.num_heads == 0 ||
            requirement.attributes.num_kv_heads == 0 || requirement.attributes.head_dimension == 0) {
          return base::Status::invalid_argument("attention requirement lacks authored dimensions");
        }
        std::uint64_t positions = 0;
        if (requirement.operation == "attention") {
          const auto& key_tensor = lowered.tensors()[requirement.operands[1].value()];
          if (key_tensor.physical_shape.size() != 3 || key_tensor.physical_shape[0] == 0 ||
              key_tensor.physical_shape[0] > std::numeric_limits<std::uint32_t>::max()) {
            return base::Status::invalid_argument("attention key tensor lacks a static position dimension");
          }
          positions = key_tensor.physical_shape[0];
        } else {
          if (requirement.attributes.attention_position == std::numeric_limits<std::uint32_t>::max()) {
            return base::Status::invalid_argument("cached attention position overflows active length");
          }
          positions = static_cast<std::uint64_t>(requirement.attributes.attention_position) + 1U;
        }
        attention = {requirement.attributes.num_heads, requirement.attributes.num_kv_heads,
                     requirement.attributes.head_dimension,
                     static_cast<std::uint32_t>(positions)};
      }
      if (requirement.operation == "cache_append") {
        if (requirement.operands.size() != 4 || requirement.attributes.num_kv_heads == 0 ||
            requirement.attributes.head_dimension == 0) {
          return base::Status::invalid_argument("cache append lacks authored dimensions");
        }
        const auto& key_cache = lowered.tensors()[requirement.operands[2].value()];
        if (key_cache.physical_shape.size() != 3 || key_cache.physical_shape[0] == 0 ||
            key_cache.physical_shape[1] != requirement.attributes.num_kv_heads ||
            key_cache.physical_shape[2] != requirement.attributes.head_dimension ||
            key_cache.physical_shape[0] > std::numeric_limits<std::uint32_t>::max()) {
          return base::Status::invalid_argument("cache append state lacks authored cache dimensions");
        }
        cache_append = {requirement.attributes.num_kv_heads, requirement.attributes.head_dimension,
                        requirement.attributes.attention_position,
                        static_cast<std::uint32_t>(key_cache.physical_shape[0])};
      }
      if (requirement.operation == "gated_delta_attention") {
        if (requirement.attributes.num_kv_heads == 0 || requirement.attributes.head_dimension == 0 ||
            requirement.attributes.value_head_count == 0 ||
            requirement.attributes.value_head_dimension == 0) {
          return base::Status::invalid_argument("gated delta requirement lacks authored dimensions");
        }
        attention = {0, requirement.attributes.num_kv_heads, requirement.attributes.head_dimension,
                     1, requirement.attributes.value_head_count,
                     requirement.attributes.value_head_dimension};
      }
      ir::physical::ConvolutionDimensions convolution{};
      if (requirement.operation == "causal_conv_silu") {
        if (requirement.attributes.convolution_kernel_dimension == 0 || requirement.operands.empty()) {
          return base::Status::invalid_argument("causal convolution lacks authored dimensions");
        }
        std::uint64_t channels = 1;
        for (const std::uint64_t dimension :
             lowered.tensors()[requirement.operands.front().value()].physical_shape) {
          const auto product = base::checked_mul(channels, dimension);
          if (!product.has_value() || product.value() == 0 ||
              product.value() > std::numeric_limits<std::uint32_t>::max()) {
            return base::Status::invalid_argument("causal convolution input has invalid channel count");
          }
          channels = product.value();
        }
        convolution = {static_cast<std::uint32_t>(channels),
                       requirement.attributes.convolution_kernel_dimension};
      }
      if (requirement.operation == "rope") {
        if (requirement.attributes.num_heads == 0 || requirement.attributes.head_dimension == 0 ||
            requirement.attributes.rope_dimension == 0 ||
            requirement.attributes.rope_dimension > requirement.attributes.head_dimension) {
          return base::Status::invalid_argument("rope requirement lacks valid authored dimensions");
        }
        rope = {requirement.attributes.num_heads, requirement.attributes.head_dimension,
                requirement.attributes.rope_dimension, requirement.attributes.rope_position};
      }
      if (requirement.operation == "split_last") {
        if (requirement.operands.size() != 3 || requirement.attributes.num_heads == 0 ||
            requirement.attributes.head_dimension == 0) {
          return base::Status::invalid_argument("last-dimension split lacks authored dimensions");
        }
        const auto& first = lowered.tensors()[requirement.operands[1].value()];
        const auto& second = lowered.tensors()[requirement.operands[2].value()];
        const std::uint64_t first_elements = first.physical_shape.empty()
                                                 ? 0
                                                 : first.physical_shape.back();
        const std::uint64_t second_elements = second.physical_shape.empty()
                                                  ? 0
                                                  : second.physical_shape.back();
        if (first_elements == 0 || second_elements == 0 || first_elements >
                std::numeric_limits<std::uint32_t>::max() || second_elements >
                std::numeric_limits<std::uint32_t>::max()) {
          return base::Status::invalid_argument("last-dimension split output shape is invalid");
        }
        split = {requirement.attributes.num_heads, static_cast<std::uint32_t>(first_elements),
                 static_cast<std::uint32_t>(second_elements)};
      }
      workspace_bytes = std::max(workspace_bytes, candidate.value().workspace_bytes);
      const float epsilon = (requirement.operation == "rms_norm" ||
                             requirement.operation == "layer_norm")
                                ? requirement.attributes.epsilon
                                : 1.0e-5F;
      const bool add_one_to_scale =
          requirement.operation == "rms_norm" &&
          requirement.attributes.norm_scale_convention ==
              ir::semantic::NormScaleConvention::one_plus_weight;
      const auto command = plan_builder.add_command(
          candidate.value().id, std::move(operands), dependencies, 0, 0,
          candidate.value().workspace_bytes, epsilon, 1.0F, attention, add_one_to_scale, {}, rope,
          cache_append, convolution, split);
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
  static ir::physical::PhysicalTensorDescriptor physical_tensor_descriptor(
      const ir::lowered::Tensor& tensor, std::uint64_t alignment) {
    ir::physical::PhysicalTensorDescriptor descriptor;
    descriptor.dtype = physical_dtype(tensor.storage_dtype);
    descriptor.shape = tensor.physical_shape;
    descriptor.layout = physical_layout(tensor.layout);
    descriptor.alignment = alignment;
    descriptor.encoding = tensor.storage_dtype == ir::semantic::DType::int4
                              ? ir::physical::StorageEncoding::nvfp4_packed
                              : (tensor.storage_dtype == ir::semantic::DType::int8
                                     ? ir::physical::StorageEncoding::fp8_e4m3_group_scale
                                     : ir::physical::StorageEncoding::none);
    if (tensor.role == ir::semantic::TensorRole::weight && !tensor.name.empty()) {
      constexpr std::string_view prefix = "weight/";
      descriptor.artifact_name = tensor.name.starts_with(prefix)
                                     ? tensor.name.substr(prefix.size())
                                     : tensor.name;
    }
    return descriptor;
  }

  static ir::physical::PhysicalDType physical_dtype(ir::semantic::DType dtype) noexcept {
    switch (dtype) {
      case ir::semantic::DType::f32: return ir::physical::PhysicalDType::f32;
      case ir::semantic::DType::f16: return ir::physical::PhysicalDType::f16;
      case ir::semantic::DType::bf16: return ir::physical::PhysicalDType::bf16;
      case ir::semantic::DType::int8: return ir::physical::PhysicalDType::u8;
      case ir::semantic::DType::int32: return ir::physical::PhysicalDType::int32;
      case ir::semantic::DType::int4: return ir::physical::PhysicalDType::u8;
    }
    return ir::physical::PhysicalDType::unknown;
  }

  static ir::physical::PhysicalLayout physical_layout(ir::lowered::LayoutKind layout) noexcept {
    switch (layout) {
      case ir::lowered::LayoutKind::row_major: return ir::physical::PhysicalLayout::row_major;
      case ir::lowered::LayoutKind::column_major: return ir::physical::PhysicalLayout::column_major;
      case ir::lowered::LayoutKind::blocked: return ir::physical::PhysicalLayout::blocked;
    }
    return ir::physical::PhysicalLayout::row_major;
  }

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

  static compiler::Lifetime lifetime_for(const ir::lowered::Tensor& tensor,
                                         const ir::lowered::Module& lowered) noexcept {
    const std::uint64_t command_count =
        std::max<std::uint64_t>(1, lowered.kernel_requirements().size());
    const compiler::AllocationClass kind = allocation_class(tensor.role);
    if (kind == compiler::AllocationClass::persistent_weight ||
        kind == compiler::AllocationClass::kv_state ||
        kind == compiler::AllocationClass::decode_state) {
      return {0, command_count};
    }

    // Lowered kernel operands contain both inputs and outputs.  Therefore the first
    // occurrence is the definition/use boundary and the last occurrence is the final
    // consumer boundary for the lowered value.  The physical planner uses half-open
    // lifetimes, so a value produced by command N and consumed by command N+1 remains
    // live through boundary N+1, while a value used only by command N can be reused by
    // command N+1.
    std::uint64_t first = command_count;
    std::uint64_t last = 0;
    for (std::uint64_t command = 0; command < lowered.kernel_requirements().size(); ++command) {
      const auto& requirement = lowered.kernel_requirements()[command];
      const bool present = std::any_of(
          requirement.operands.begin(), requirement.operands.end(),
          [&](const ir::lowered::LoweredTensorId operand) { return operand == tensor.id; });
      if (!present) continue;
      first = std::min(first, command);
      last = command + 1;
    }
    if (first == command_count) {
      // Entry-bound tensors that are not consumed by a command still need a valid
      // allocation for plan binding.  Keep their minimal lifetime deterministic.
      return {0, 1};
    }
    return {first, last};
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
    std::vector<std::string_view> operand_dtypes;
    operand_dtypes.reserve(requirement.operands.size());
    for (const ir::lowered::LoweredTensorId operand : requirement.operands) {
      if (operand.value() >= lowered.tensors().size()) {
        return base::Status::out_of_range("kernel operand dtype is undefined");
      }
      operand_dtypes.push_back(dtype_name(lowered.tensors()[operand.value()].storage_dtype));
    }
    const auto candidates = provider.enumerate(
        {requirement.operation, requirement.target_capability, storage_dtype,
         requirement.operands.size(), std::move(operand_dtypes),
         requirement.attributes.attention_output_gate !=
             ir::semantic::AttentionOutputGate::none});
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
