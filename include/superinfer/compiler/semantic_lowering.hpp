#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <string_view>
#include <vector>

#include <superinfer/base/memory_space.hpp>
#include <superinfer/base/result.hpp>
#include <superinfer/ir/lowered/module.hpp>
#include <superinfer/ir/semantic/module.hpp>

namespace superinfer::compiler {

struct SemanticLoweringOptions final {
  std::uint32_t target_capability{0};
  std::uint64_t required_alignment{1};
};

/**
 * Generic shape/origin lowering for canonical semantic graphs.
 *
 * This pass does not inspect model identity or select a physical kernel. It carries each static
 * semantic tensor into Lowered IR and records operation capability requirements with explicit
 * lowered tensor operands. Target-specific layout and provider selection remain later boundaries.
 */
class SemanticLowering final {
 public:
  [[nodiscard]] base::Result<ir::lowered::Module> lower(
      const ir::semantic::Module& semantic, SemanticLoweringOptions options) const {
    base::Status semantic_status = semantic.verify();
    if (!semantic_status.ok()) return semantic_status.with_context("semantic lowering input");
    if (options.target_capability == 0 || options.required_alignment == 0) {
      return base::Status::invalid_argument("semantic lowering target/alignment is incomplete");
    }

    ir::lowered::ModuleBuilder builder;
    std::vector<ir::lowered::LoweredTensorId> lowered_tensors;
    lowered_tensors.reserve(semantic.tensors().size());
    for (const ir::semantic::Tensor& tensor : semantic.tensors()) {
      std::vector<std::uint64_t> shape;
      shape.reserve(tensor.spec.shape.size());
      for (const ir::semantic::Dimension& dimension : tensor.spec.shape) {
        if (dimension.is_symbolic) {
          return base::Status::unsupported("semantic lowering requires static tensor dimensions");
        }
        shape.push_back(dimension.value);
      }
      const auto lowered = builder.add_tensor(
          tensor.id, std::move(shape), ir::lowered::LayoutKind::row_major,
          base::MemorySpace::device, options.required_alignment, tensor.spec.dtype,
          ir::semantic::DType::f32, tensor.spec.role, tensor.name);
      if (!lowered.has_value()) {
        base::Status error = lowered.error();
        return error.with_context("semantic tensor lowering");
      }
      lowered_tensors.push_back(lowered.value());
    }

    const auto add_f32_scratch = [&](ir::semantic::TensorId origin)
        -> base::Result<ir::lowered::LoweredTensorId> {
      const ir::semantic::Tensor& source = semantic.tensors()[origin.value()];
      std::vector<std::uint64_t> shape;
      shape.reserve(source.spec.shape.size());
      for (const ir::semantic::Dimension& dimension : source.spec.shape) {
        if (dimension.is_symbolic) {
          return base::Status::unsupported("target lowering scratch tensor requires static dimensions");
        }
        shape.push_back(dimension.value);
      }
      return builder.add_tensor(origin, std::move(shape), ir::lowered::LayoutKind::row_major,
                                base::MemorySpace::device, options.required_alignment,
                                ir::semantic::DType::f32, ir::semantic::DType::f32,
                                source.spec.role, source.name + "$fp32");
    };

    const auto emit_cast = [&](ir::semantic::DType source_dtype,
                               ir::semantic::DType destination_dtype,
                               ir::lowered::LoweredTensorId source,
                               ir::lowered::LoweredTensorId destination) -> base::Status {
      if (source_dtype == destination_dtype) return {};
      if (!((source_dtype == ir::semantic::DType::bf16 &&
             destination_dtype == ir::semantic::DType::f32) ||
            (source_dtype == ir::semantic::DType::f32 &&
             destination_dtype == ir::semantic::DType::bf16))) {
        return base::Status::unsupported("target lowering lacks an explicit dtype conversion");
      }
      return builder.add_kernel_requirement("cast", options.target_capability,
                                            {source, destination});
    };

    const auto add_nvfp4_sidecars = [&](ir::semantic::TensorId weight)
        -> base::Result<std::pair<ir::lowered::LoweredTensorId,
                                  ir::lowered::LoweredTensorId>> {
      const ir::semantic::Tensor& source = semantic.tensors()[weight.value()];
      if (source.spec.shape.size() != 2 || source.spec.shape[0].is_symbolic ||
          source.spec.shape[1].is_symbolic || source.spec.shape[0].value == 0 ||
          source.spec.shape[1].value == 0 || source.spec.shape[1].value % 16U != 0) {
        return base::Status::invalid_argument(
            "NVFP4 projection weight requires two static dimensions divisible by group size");
      }
      const std::string_view suffix = ".weight";
      if (!source.name.ends_with(suffix)) {
        return base::Status::invalid_argument("NVFP4 projection weight name lacks .weight suffix");
      }
      const std::string prefix = source.name.substr(0, source.name.size() - suffix.size());
      const auto block_scale = builder.add_tensor(
          weight, {source.spec.shape[0].value, source.spec.shape[1].value / 16U},
          ir::lowered::LayoutKind::row_major, base::MemorySpace::device,
          options.required_alignment, ir::semantic::DType::int8, ir::semantic::DType::f32,
          ir::semantic::TensorRole::weight, prefix + ".weight_scale");
      if (!block_scale.has_value()) return block_scale.error();
      const auto tensor_scale = builder.add_tensor(
          weight, {1}, ir::lowered::LayoutKind::row_major, base::MemorySpace::device,
          options.required_alignment, ir::semantic::DType::f32, ir::semantic::DType::f32,
          ir::semantic::TensorRole::weight, prefix + ".weight_scale_2");
      if (!tensor_scale.has_value()) return tensor_scale.error();
      return std::make_pair(block_scale.value(), tensor_scale.value());
    };

    for (const ir::semantic::Operation& operation : semantic.operations()) {
      const std::string_view capability = capability_name(operation.kind);
      if (capability.empty()) {
        return base::Status::unsupported("semantic operation has no generic lowering capability");
      }
      std::vector<ir::lowered::LoweredTensorId> inputs;
      inputs.reserve(operation.inputs.size());
      for (const ir::semantic::TensorId input : operation.inputs) {
        inputs.push_back(lowered_tensors[input.value()]);
      }
      std::vector<ir::lowered::LoweredTensorId> outputs;
      outputs.reserve(operation.outputs.size());
      for (const ir::semantic::TensorId output : operation.outputs) {
        outputs.push_back(lowered_tensors[output.value()]);
      }

      // The authored Qwen graph uses BF16 activations, while the correctness-first SM120
      // providers intentionally consume FP32 activations. Materialize that boundary explicitly;
      // never let a provider reinterpret BF16 storage as float memory.
      const bool fp32_activation_contract = operation.kind == ir::semantic::OperationKind::embedding ||
                                             operation.kind == ir::semantic::OperationKind::rms_norm ||
                                             operation.kind == ir::semantic::OperationKind::residual ||
                                             operation.kind == ir::semantic::OperationKind::gated_dense_ffn ||
                                             operation.kind == ir::semantic::OperationKind::lm_head;
      const std::size_t converted_inputs =
          operation.kind == ir::semantic::OperationKind::residual ? inputs.size() :
          std::min<std::size_t>(inputs.size(), 1);
      if (fp32_activation_contract) {
        for (std::size_t index = 0; index < converted_inputs; ++index) {
          if (semantic.tensors()[operation.inputs[index].value()].spec.dtype !=
              ir::semantic::DType::bf16) {
            continue;
          }
          const auto scratch = add_f32_scratch(operation.inputs[index]);
          if (!scratch.has_value()) {
            base::Status error = scratch.error();
            return error.with_context("activation scratch lowering");
          }
          base::Status cast = emit_cast(ir::semantic::DType::bf16, ir::semantic::DType::f32,
                                        inputs[index], scratch.value());
          if (!cast.ok()) return cast.with_context("activation input lowering");
          inputs[index] = scratch.value();
        }
      }

      ir::semantic::DType output_dtype = ir::semantic::DType::f32;
      if (!operation.outputs.empty()) {
        output_dtype = semantic.tensors()[operation.outputs.front().value()].spec.dtype;
      }
      ir::lowered::LoweredTensorId output_target{};
      bool cast_output = fp32_activation_contract && !outputs.empty() &&
                         output_dtype == ir::semantic::DType::bf16;
      if (cast_output) {
        const auto scratch = add_f32_scratch(operation.outputs.front());
        if (!scratch.has_value()) {
          base::Status error = scratch.error();
          return error.with_context("activation output lowering");
        }
        output_target = scratch.value();
        outputs.front() = output_target;
      }

      std::string lowered_operation{capability};
      std::vector<ir::lowered::LoweredTensorId> operands;
      if (operation.kind == ir::semantic::OperationKind::lm_head && operation.inputs.size() == 2 &&
          semantic.tensors()[operation.inputs[1].value()].spec.dtype == ir::semantic::DType::int4) {
        const auto sidecars = add_nvfp4_sidecars(operation.inputs[1]);
        if (!sidecars.has_value()) {
          base::Status error = sidecars.error();
          return error.with_context("NVFP4 projection sidecar lowering");
        }
        lowered_operation = "nvfp4_linear";
        operands = {inputs[0], inputs[1], sidecars.value().first, sidecars.value().second,
                    outputs.front()};
      } else {
        operands.reserve(inputs.size() + outputs.size());
        operands.insert(operands.end(), inputs.begin(), inputs.end());
        operands.insert(operands.end(), outputs.begin(), outputs.end());
      }
      base::Status requirement = builder.add_kernel_requirement(
          std::move(lowered_operation), options.target_capability, std::move(operands),
          operation.attributes);
      if (!requirement.ok()) return requirement.with_context("semantic operation lowering");
      if (cast_output) {
        base::Status cast = emit_cast(ir::semantic::DType::f32, ir::semantic::DType::bf16,
                                      output_target, lowered_tensors[operation.outputs.front().value()]);
        if (!cast.ok()) return cast.with_context("activation output lowering");
      }
    }
    for (const ir::semantic::StateEdge& edge : semantic.state_edges()) {
      const auto slot = builder.add_state_slot(
          edge.name, lowered_tensors[edge.source.value()], lowered_tensors[edge.destination.value()]);
      if (!slot.has_value()) {
        base::Status error = slot.error();
        return error.with_context("semantic state lowering");
      }
      for (const ir::lowered::StateAction action : {ir::lowered::StateAction::read,
                                                     ir::lowered::StateAction::write,
                                                     ir::lowered::StateAction::commit}) {
        base::Status transition = builder.add_state_transition(slot.value(), action);
        if (!transition.ok()) return transition.with_context("semantic state transition lowering");
      }
    }
    for (const ir::semantic::EntryPoint& entry : semantic.entry_points()) {
      std::vector<ir::lowered::LoweredTensorId> inputs;
      std::vector<ir::lowered::LoweredTensorId> outputs;
      inputs.reserve(entry.inputs.size());
      outputs.reserve(entry.outputs.size());
      for (const ir::semantic::TensorId id : entry.inputs) inputs.push_back(lowered_tensors[id.value()]);
      for (const ir::semantic::TensorId id : entry.outputs) outputs.push_back(lowered_tensors[id.value()]);
      base::Status binding = builder.add_entry_point(entry.name, std::move(inputs), std::move(outputs));
      if (!binding.ok()) return binding.with_context("semantic entry point lowering");
    }
    return std::move(builder).build();
  }

 private:
  static std::string_view capability_name(ir::semantic::OperationKind kind) noexcept {
    using ir::semantic::OperationKind;
    switch (kind) {
      case OperationKind::embedding: return "embedding";
      case OperationKind::cast: return "cast";
      case OperationKind::rms_norm: return "rms_norm";
      case OperationKind::layer_norm: return "layer_norm";
      case OperationKind::rope: return "rope";
      case OperationKind::qkv_projection: return "qkv_projection";
      case OperationKind::gated_delta_attention: return "gated_delta_attention";
      case OperationKind::multi_head_attention: return "attention";
      case OperationKind::grouped_query_attention: return "attention";
      case OperationKind::gated_grouped_query_attention: return "attention";
      case OperationKind::local_attention: return "attention";
      case OperationKind::residual: return "residual";
      case OperationKind::gated_dense_ffn: return "gated_dense_ffn";
      case OperationKind::moe_route: return "moe_route";
      case OperationKind::moe_top_k: return "moe_top_k";
      case OperationKind::moe_expert: return "moe_expert";
      case OperationKind::moe_combine: return "moe_combine";
      case OperationKind::lm_head: return "lm_head";
      case OperationKind::decode_logits: return "decode_logits";
      case OperationKind::sampling_inputs: return "sampling_inputs";
    }
    return {};
  }
};

}  // namespace superinfer::compiler
