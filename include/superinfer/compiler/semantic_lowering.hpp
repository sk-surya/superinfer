#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
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

    const auto add_named_f32_scratch = [&](ir::semantic::TensorId origin,
                                           std::vector<std::uint64_t> shape,
                                           std::string name)
        -> base::Result<ir::lowered::LoweredTensorId> {
      return builder.add_tensor(origin, std::move(shape), ir::lowered::LayoutKind::row_major,
                                base::MemorySpace::device, options.required_alignment,
                                ir::semantic::DType::f32, ir::semantic::DType::f32,
                                ir::semantic::TensorRole::activation, std::move(name));
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
                                             operation.kind == ir::semantic::OperationKind::gated_grouped_query_attention ||
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
      const bool quantized_ffn =
          operation.kind == ir::semantic::OperationKind::gated_dense_ffn &&
          operation.inputs.size() == 4 &&
          semantic.tensors()[operation.inputs[1].value()].spec.dtype == ir::semantic::DType::int4 &&
          semantic.tensors()[operation.inputs[2].value()].spec.dtype == ir::semantic::DType::int4 &&
          semantic.tensors()[operation.inputs[3].value()].spec.dtype == ir::semantic::DType::int4;
      if (quantized_ffn) {
        const auto& gate_weight = semantic.tensors()[operation.inputs[1].value()];
        const auto& up_weight = semantic.tensors()[operation.inputs[2].value()];
        const auto& down_weight = semantic.tensors()[operation.inputs[3].value()];
        if (gate_weight.spec.shape.size() != 2 || up_weight.spec.shape.size() != 2 ||
            down_weight.spec.shape.size() != 2 || gate_weight.spec.shape[0].is_symbolic ||
            gate_weight.spec.shape[1].is_symbolic || up_weight.spec.shape[0].is_symbolic ||
            up_weight.spec.shape[1].is_symbolic || down_weight.spec.shape[0].is_symbolic ||
            down_weight.spec.shape[1].is_symbolic ||
            up_weight.spec.shape[0].value != gate_weight.spec.shape[0].value ||
            up_weight.spec.shape[1].value != gate_weight.spec.shape[1].value ||
            down_weight.spec.shape[1].value != gate_weight.spec.shape[0].value ||
            gate_weight.spec.shape[1].value % 16U != 0 ||
            down_weight.spec.shape[1].value % 16U != 0) {
          return base::Status::invalid_argument(
              "quantized gated FFN projection shapes are incompatible");
        }
        const std::uint64_t batch = semantic.tensors()[operation.inputs[0].value()].spec.shape[0].value;
        const std::uint64_t intermediate = gate_weight.spec.shape[0].value;
        const auto gate_projection = add_named_f32_scratch(
            operation.outputs.front(), {batch, intermediate},
            semantic.tensors()[operation.inputs[1].value()].name + "$projection");
        const auto up_projection = add_named_f32_scratch(
            operation.outputs.front(), {batch, intermediate},
            semantic.tensors()[operation.inputs[2].value()].name + "$projection");
        const auto gated_projection = add_named_f32_scratch(
            operation.outputs.front(), {batch, intermediate},
            semantic.tensors()[operation.inputs[3].value()].name + "$gated");
        if (!gate_projection.has_value() || !up_projection.has_value() ||
            !gated_projection.has_value()) {
          return base::Status::resource_exhausted("quantized FFN scratch lowering failed");
        }
        const auto gate_sidecars = add_nvfp4_sidecars(operation.inputs[1]);
        const auto up_sidecars = add_nvfp4_sidecars(operation.inputs[2]);
        const auto down_sidecars = add_nvfp4_sidecars(operation.inputs[3]);
        if (!gate_sidecars.has_value() || !up_sidecars.has_value() || !down_sidecars.has_value()) {
          return base::Status::invalid_argument("quantized FFN sidecar lowering failed");
        }
        const auto emit_projection = [&](ir::semantic::TensorId weight,
                                         const std::pair<ir::lowered::LoweredTensorId,
                                                         ir::lowered::LoweredTensorId>& sidecars,
                                         ir::lowered::LoweredTensorId destination) -> base::Status {
          return builder.add_kernel_requirement(
              "nvfp4_linear", options.target_capability,
              {inputs[0], lowered_tensors[weight.value()], sidecars.first, sidecars.second,
               destination});
        };
        base::Status gate_status = emit_projection(operation.inputs[1], gate_sidecars.value(),
                                                   gate_projection.value());
        base::Status up_status = emit_projection(operation.inputs[2], up_sidecars.value(),
                                                 up_projection.value());
        if (!gate_status.ok() || !up_status.ok()) {
          return (!gate_status.ok() ? gate_status : up_status).with_context(
              "quantized FFN projection lowering");
        }
        base::Status activation_status = builder.add_kernel_requirement(
            "silu_mul", options.target_capability,
            {gate_projection.value(), up_projection.value(), gated_projection.value()});
        if (!activation_status.ok()) return activation_status.with_context("quantized FFN activation lowering");
        base::Status down_status = emit_projection(operation.inputs[3], down_sidecars.value(),
                                                   outputs.front());
        if (!down_status.ok()) return down_status.with_context("quantized FFN down projection lowering");
        if (cast_output) {
          base::Status cast = emit_cast(ir::semantic::DType::f32, ir::semantic::DType::bf16,
                                        output_target, lowered_tensors[operation.outputs.front().value()]);
          if (!cast.ok()) return cast.with_context("activation output lowering");
        }
        continue;
      }
      if (operation.kind == ir::semantic::OperationKind::gated_grouped_query_attention) {
        // Qwen's full-attention node is semantically gated: q_proj contains a query and a
        // separate gate, while K/V are appended to persistent BF16 state before grouped
        // attention. Keep every boundary explicit so no provider has to infer model meaning
        // from an unusual projection shape.
        if (operation.inputs.size() != 9 || operation.outputs.size() != 3 ||
            operation.attributes.attention_output_gate != ir::semantic::AttentionOutputGate::sigmoid) {
          return base::Status::invalid_argument("gated grouped attention contract is incomplete");
        }
        const auto static_first_dimension = [&](std::size_t input_index) -> base::Result<std::uint64_t> {
          const auto& shape = semantic.tensors()[operation.inputs[input_index].value()].spec.shape;
          if (shape.size() != 2 || shape[0].is_symbolic || shape[1].is_symbolic ||
              shape[0].value == 0 || shape[1].value == 0) {
            return base::Status::invalid_argument("attention projection requires a static matrix shape");
          }
          return shape[0].value;
        };
        const auto q_elements = static_first_dimension(3);
        const auto k_elements = static_first_dimension(4);
        const auto v_elements = static_first_dimension(5);
        const auto o_elements = static_first_dimension(6);
        if (!q_elements.has_value() || !k_elements.has_value() || !v_elements.has_value() ||
            !o_elements.has_value() || q_elements.value() == 0 || q_elements.value() % 2 != 0) {
          return base::Status::invalid_argument("attention projection dimensions are invalid");
        }
        const auto q_projection = add_named_f32_scratch(
            operation.outputs.front(), {q_elements.value()}, operation.name + "$q_projection");
        const auto q_raw = add_named_f32_scratch(
            operation.outputs.front(), {q_elements.value() / 2U}, operation.name + "$q");
        const auto q_gate = add_named_f32_scratch(
            operation.outputs.front(), {q_elements.value() / 2U}, operation.name + "$gate");
        const auto k_projection = add_named_f32_scratch(
            operation.outputs.front(), {k_elements.value()}, operation.name + "$k_projection");
        const auto v_projection = add_named_f32_scratch(
            operation.outputs.front(), {v_elements.value()}, operation.name + "$v_projection");
        const auto q_norm = add_named_f32_scratch(
            operation.outputs.front(), {q_elements.value() / 2U}, operation.name + "$q_norm");
        const auto k_norm = add_named_f32_scratch(
            operation.outputs.front(), {k_elements.value()}, operation.name + "$k_norm");
        const auto q_rope = add_named_f32_scratch(
            operation.outputs.front(), {q_elements.value() / 2U}, operation.name + "$q_rope");
        const auto k_rope = add_named_f32_scratch(
            operation.outputs.front(), {k_elements.value()}, operation.name + "$k_rope");
        const auto attended = add_named_f32_scratch(
            operation.outputs.front(), {q_elements.value() / 2U}, operation.name + "$attended");
        const auto gated = add_named_f32_scratch(
            operation.outputs.front(), {q_elements.value() / 2U}, operation.name + "$gated");
        if (!q_projection.has_value() || !q_raw.has_value() || !q_gate.has_value() ||
            !k_projection.has_value() || !v_projection.has_value() || !q_norm.has_value() ||
            !k_norm.has_value() || !q_rope.has_value() || !k_rope.has_value() ||
            !attended.has_value() || !gated.has_value()) {
          return base::Status::resource_exhausted("gated grouped attention scratch lowering failed");
        }
        const auto q_sidecars = add_nvfp4_sidecars(operation.inputs[3]);
        const auto k_sidecars = add_nvfp4_sidecars(operation.inputs[4]);
        const auto v_sidecars = add_nvfp4_sidecars(operation.inputs[5]);
        const auto o_sidecars = add_nvfp4_sidecars(operation.inputs[6]);
        if (!q_sidecars.has_value() || !k_sidecars.has_value() || !v_sidecars.has_value() ||
            !o_sidecars.has_value()) {
          return base::Status::invalid_argument("gated grouped attention sidecar lowering failed");
        }
        const auto emit_projection = [&](std::size_t weight_index,
                                         const std::pair<ir::lowered::LoweredTensorId,
                                                         ir::lowered::LoweredTensorId>& sidecars,
                                         ir::lowered::LoweredTensorId destination) -> base::Status {
          return builder.add_kernel_requirement(
              "nvfp4_linear", options.target_capability,
              {inputs[0], lowered_tensors[operation.inputs[weight_index].value()], sidecars.first,
               sidecars.second, destination});
        };
        base::Status status = emit_projection(3, q_sidecars.value(), q_projection.value());
        if (!status.ok()) return status.with_context("attention Q projection lowering");
        status = builder.add_kernel_requirement(
            "split", options.target_capability,
            {q_projection.value(), q_raw.value(), q_gate.value()});
        if (!status.ok()) return status.with_context("attention Q/gate split lowering");
        status = emit_projection(4, k_sidecars.value(), k_projection.value());
        if (!status.ok()) return status.with_context("attention K projection lowering");
        status = emit_projection(5, v_sidecars.value(), v_projection.value());
        if (!status.ok()) return status.with_context("attention V projection lowering");

        ir::semantic::OperationAttributes norm_attributes;
        norm_attributes.epsilon = 1.0e-6F;
        const auto q_norm_weight = lowered_tensors[operation.inputs[7].value()];
        const auto k_norm_weight = lowered_tensors[operation.inputs[8].value()];
        status = builder.add_kernel_requirement(
            "rms_norm", options.target_capability,
            {q_raw.value(), q_norm_weight, q_norm.value()}, norm_attributes);
        if (!status.ok()) return status.with_context("attention Q norm lowering");
        status = builder.add_kernel_requirement(
            "rms_norm", options.target_capability,
            {k_projection.value(), k_norm_weight, k_norm.value()}, norm_attributes);
        if (!status.ok()) return status.with_context("attention K norm lowering");

        auto q_rope_attributes = operation.attributes;
        status = builder.add_kernel_requirement(
            "rope", options.target_capability, {q_norm.value(), q_rope.value()}, q_rope_attributes);
        if (!status.ok()) return status.with_context("attention Q RoPE lowering");
        auto k_rope_attributes = operation.attributes;
        k_rope_attributes.num_heads = operation.attributes.num_kv_heads;
        status = builder.add_kernel_requirement(
            "rope", options.target_capability, {k_norm.value(), k_rope.value()}, k_rope_attributes);
        if (!status.ok()) return status.with_context("attention K RoPE lowering");
        status = builder.add_kernel_requirement(
            "cache_append", options.target_capability,
            {k_rope.value(), v_projection.value(), inputs[1], inputs[2]}, operation.attributes);
        if (!status.ok()) return status.with_context("attention KV cache lowering");
        status = builder.add_kernel_requirement(
            "attention_bf16_cache", options.target_capability,
            {q_rope.value(), inputs[1], inputs[2], attended.value()}, operation.attributes);
        if (!status.ok()) return status.with_context("attention cached GQA lowering");
        status = builder.add_kernel_requirement(
            "sigmoid_mul", options.target_capability,
            {q_gate.value(), attended.value(), gated.value()});
        if (!status.ok()) return status.with_context("attention output gate lowering");
        status = emit_projection(6, o_sidecars.value(), outputs.front());
        if (!status.ok()) return status.with_context("attention O projection lowering");
        if (cast_output) {
          base::Status cast = emit_cast(ir::semantic::DType::f32, ir::semantic::DType::bf16,
                                        output_target, lowered_tensors[operation.outputs.front().value()]);
          if (!cast.ok()) return cast.with_context("activation output lowering");
        }
        continue;
      }
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
