#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <superinfer/compiler/model_frontend.hpp>
#include <superinfer/ir/semantic/builder.hpp>

namespace superinfer::frontends::qwen38 {

inline constexpr std::string_view kSourceIdentity =
    "Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    "+gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090@0cc27958cefbbe231782ec8511de8c4eb5233348"
    "+LMHead4@62abd1d060bd801005f47754f01619054cc248d3417699ecea414c7ede1b3a4a";
inline constexpr std::uint64_t kTensorCount = 2402;
inline constexpr std::string_view kTensorInventorySha256 =
    "7342659a53eecbb04c47b5de89d957ca47cb021970cb252575b8b9161d0a84fc";

/**
 * Emits canonical Semantic IR for the pinned Qwen3.8 language path.
 *
 * The frontend owns model topology and semantic attributes only. It does not select layouts,
 * storage encodings, CUDA kernels, or target capabilities. This V0 slice emits one-token graph
 * topology for all 64 language layers; tensor payload mapping is validated by the Python source
 * inventory before this frontend is invoked.
 */
class Frontend final : public compiler::ModelFrontend {
 public:
  base::Status validate(const compiler::SourceInventory& source) const override {
    if (source.identity != kSourceIdentity) {
      return base::Status::failed_precondition("Qwen3.8 source identity is not pinned revision");
    }
    if (source.tensor_count != kTensorCount || source.tensor_inventory_sha256 != kTensorInventorySha256) {
      return base::Status::failed_precondition("Qwen3.8 tensor inventory is not authenticated");
    }
    base::Status inventory_status = source.validate();
    if (!inventory_status.ok()) return inventory_status.with_context("Qwen3.8 source tensor inventory");
    if (source.tensors.empty()) {
      return base::Status::failed_precondition("Qwen3.8 source tensor records are missing");
    }
    return {};
  }

  base::Result<ir::semantic::Module> emit(const compiler::SourceInventory& source) const override {
    const base::Status source_status = validate(source);
    if (!source_status.ok()) {
      base::Status error = source_status;
      return error.with_context("Qwen3.8 frontend source");
    }

    using namespace ir::semantic;
    Builder builder;
    const TensorSpec token_spec{{Dimension::static_value(1)}, DType::int32,
                                QuantizationIntent::none, TensorRole::activation};
    const TensorSpec hidden_spec{{Dimension::static_value(1), Dimension::static_value(5120)},
                                 DType::bf16, QuantizationIntent::none, TensorRole::activation};
    const TensorSpec logits_spec{{Dimension::static_value(1), Dimension::static_value(248320)},
                                 DType::bf16, QuantizationIntent::none, TensorRole::logits};

    const auto token_ids = builder.add_tensor("token_ids", token_spec);
    if (!token_ids.has_value()) {
      base::Status error = token_ids.error();
      return error.with_context("Qwen3.8 token input");
    }
    const auto embedding = builder.add_tensor("embedding", hidden_spec);
    if (!embedding.has_value()) {
      base::Status error = embedding.error();
      return error.with_context("Qwen3.8 embedding");
    }
    std::vector<std::pair<std::string, TensorId>> source_weights;
    source_weights.reserve(source.tensors.size());
    for (const compiler::SourceTensorRecord& record : source.tensors) {
      if (!is_semantic_parameter(record.role)) continue;
      Shape shape;
      shape.reserve(record.shape.size());
      for (const std::uint64_t dimension : record.shape) {
        shape.push_back(Dimension::static_value(dimension));
      }
      const auto source_spec = source_tensor_spec(record, std::move(shape));
      if (!source_spec.has_value()) {
        base::Status error = source_spec.error();
        return error.with_context("Qwen3.8 source tensor " + record.name);
      }
      const auto source_weight = builder.add_tensor("weight/" + record.name, source_spec.value());
      if (!source_weight.has_value()) {
        base::Status error = source_weight.error();
        return error.with_context("Qwen3.8 source tensor " + record.name);
      }
      source_weights.emplace_back(record.name, source_weight.value());
    }
    const auto embedding_weight = find_source_weight(source_weights,
                                                      "model.language_model.embed_tokens.weight");
    if (!embedding_weight.has_value()) {
      base::Status error = embedding_weight.error();
      return error.with_context("Qwen3.8 embedding weight binding");
    }
    const auto append_required_weight = [&](std::vector<TensorId>& operands,
                                            std::string name) -> base::Status {
      const auto weight = find_source_weight(source_weights, name);
      if (!weight.has_value()) return weight.error();
      operands.push_back(weight.value());
      return {};
    };
    if (!builder.add_operation("embedding", OperationKind::embedding,
                               {token_ids.value(), embedding_weight.value()},
                               {embedding.value()})
             .has_value()) {
      return base::Status::invalid_argument("Qwen3.8 embedding operation could not be emitted");
    }

    TensorId current = embedding.value();
    std::uint32_t gated_delta_layers = 0;
    std::uint32_t full_attention_layers = 0;
    for (std::uint32_t layer = 0; layer < 64; ++layer) {
      const auto input_norm = builder.add_tensor("layer_" + index(layer) + "_input_norm", hidden_spec);
      const auto attention = builder.add_tensor("layer_" + index(layer) + "_attention", hidden_spec);
      const auto attention_residual = builder.add_tensor("layer_" + index(layer) + "_attention_residual", hidden_spec);
      const auto post_norm = builder.add_tensor("layer_" + index(layer) + "_post_norm", hidden_spec);
      const auto ffn = builder.add_tensor("layer_" + index(layer) + "_ffn", hidden_spec);
      const auto output = builder.add_tensor("layer_" + index(layer) + "_output", hidden_spec);
      const bool full_attention = (layer % 4U) == 3U;
      const TensorSpec linear_state_spec{{Dimension::static_value(48), Dimension::static_value(128),
                                          Dimension::static_value(128)},
                                         DType::f32, QuantizationIntent::none,
                                         TensorRole::kv_cache};
      const TensorSpec convolution_state_spec{{Dimension::static_value(4), Dimension::static_value(10240)},
                                              DType::bf16, QuantizationIntent::none,
                                              TensorRole::decode_state};
      const TensorSpec full_key_state_spec{{Dimension::static_value(4), Dimension::static_value(256)},
                                           DType::bf16, QuantizationIntent::none,
                                           TensorRole::kv_cache};
      const auto state_a_in = builder.add_tensor(
          "layer_" + index(layer) + (full_attention ? "_key_state_in" : "_delta_state_in"),
          full_attention ? full_key_state_spec : linear_state_spec);
      const auto state_a_out = builder.add_tensor(
          "layer_" + index(layer) + (full_attention ? "_key_state_out" : "_delta_state_out"),
          full_attention ? full_key_state_spec : linear_state_spec);
      const auto state_b_in = builder.add_tensor(
          "layer_" + index(layer) + (full_attention ? "_value_state_in" : "_convolution_state_in"),
          full_attention ? full_key_state_spec : convolution_state_spec);
      const auto state_b_out = builder.add_tensor(
          "layer_" + index(layer) + (full_attention ? "_value_state_out" : "_convolution_state_out"),
          full_attention ? full_key_state_spec : convolution_state_spec);
      if (!input_norm.has_value() || !attention.has_value() || !attention_residual.has_value() ||
          !post_norm.has_value() || !ffn.has_value() || !output.has_value() || !state_a_in.has_value() ||
          !state_a_out.has_value() || !state_b_in.has_value() || !state_b_out.has_value()) {
        return base::Status::resource_exhausted("Qwen3.8 frontend tensor emission failed");
      }
      std::vector<TensorId> input_norm_inputs{current};
      const auto input_norm_weight = find_source_weight(
          source_weights,
          "model.language_model.layers." + std::to_string(layer) + ".input_layernorm.weight");
      if (input_norm_weight.has_value()) input_norm_inputs.push_back(input_norm_weight.value());
      OperationAttributes norm_attributes;
      norm_attributes.epsilon = 1.0e-6F;
      norm_attributes.norm_scale_convention = NormScaleConvention::one_plus_weight;
      if (!builder.add_operation("layer_" + index(layer) + "_input_norm", OperationKind::rms_norm,
                                 std::move(input_norm_inputs), {input_norm.value()}, norm_attributes)
                   .has_value()) {
        return base::Status::invalid_argument("Qwen3.8 input norm operation could not be emitted");
      }
      OperationAttributes attention_attributes;
      attention_attributes.num_heads = full_attention ? 24 : 16;
      attention_attributes.num_kv_heads = full_attention ? 4 : 16;
      attention_attributes.head_dimension = full_attention ? 256 : 128;
      attention_attributes.rope_dimension = full_attention ? 64 : 0;
      if (full_attention) {
        attention_attributes.attention_output_gate = AttentionOutputGate::sigmoid;
      }
      if (!full_attention) {
        attention_attributes.key_head_dimension = 128;
        attention_attributes.value_head_dimension = 128;
        attention_attributes.convolution_kernel_dimension = 4;
        attention_attributes.value_head_count = 48;
      }
      const OperationKind attention_kind = full_attention
                                               ? OperationKind::grouped_query_attention
                                               : OperationKind::gated_delta_attention;
      std::vector<TensorId> attention_inputs{input_norm.value(), state_a_in.value(), state_b_in.value()};
      const std::string weight_prefix = "model.language_model.layers." + std::to_string(layer) + ".";
      const std::vector<std::string> attention_weight_names = full_attention
          ? std::vector<std::string>{weight_prefix + "self_attn.q_proj.weight",
                                     weight_prefix + "self_attn.k_proj.weight",
                                     weight_prefix + "self_attn.v_proj.weight",
                                     weight_prefix + "self_attn.o_proj.weight",
                                     weight_prefix + "self_attn.q_norm.weight",
                                     weight_prefix + "self_attn.k_norm.weight"}
          : std::vector<std::string>{weight_prefix + "linear_attn.in_proj_qkv.weight",
                                     weight_prefix + "linear_attn.in_proj_z.weight",
                                     weight_prefix + "linear_attn.in_proj_a.weight",
                                     weight_prefix + "linear_attn.in_proj_b.weight",
                                     weight_prefix + "linear_attn.out_proj.weight",
                                     weight_prefix + "linear_attn.A_log",
                                     weight_prefix + "linear_attn.dt_bias",
                                     weight_prefix + "linear_attn.norm.weight",
                                     weight_prefix + "linear_attn.conv1d.weight"};
      for (const std::string& weight_name : attention_weight_names) {
        base::Status binding = append_required_weight(attention_inputs, weight_name);
        if (!binding.ok()) return binding.with_context("Qwen3.8 attention weight binding");
      }
      if (!builder.add_operation("layer_" + index(layer) + "_attention", attention_kind,
                                 std::move(attention_inputs),
                                 {attention.value(), state_a_out.value(), state_b_out.value()},
                                 attention_attributes)
                   .has_value()) {
        return base::Status::invalid_argument("Qwen3.8 attention operation could not be emitted");
      }
      if (!builder.add_state_edge("layer_" + index(layer) + "_state_a", state_a_in.value(),
                                 state_a_out.value())
                   .ok() ||
          !builder.add_state_edge("layer_" + index(layer) + "_state_b", state_b_in.value(),
                                 state_b_out.value())
                   .ok()) {
        return base::Status::invalid_argument("Qwen3.8 attention state could not be emitted");
      }
      if (full_attention) {
        ++full_attention_layers;
      } else {
        ++gated_delta_layers;
      }
      std::vector<TensorId> post_norm_inputs{attention_residual.value()};
      const auto post_norm_weight = find_source_weight(
          source_weights,
          "model.language_model.layers." + std::to_string(layer) + ".post_attention_layernorm.weight");
      if (post_norm_weight.has_value()) post_norm_inputs.push_back(post_norm_weight.value());
      std::vector<TensorId> ffn_inputs{post_norm.value()};
      for (const std::string& suffix : {std::string{"mlp.gate_proj.weight"},
                                        std::string{"mlp.up_proj.weight"},
                                        std::string{"mlp.down_proj.weight"}}) {
        base::Status binding = append_required_weight(ffn_inputs, weight_prefix + suffix);
        if (!binding.ok()) return binding.with_context("Qwen3.8 FFN weight binding");
      }
      if (!builder.add_operation("layer_" + index(layer) + "_attention_residual",
                                 OperationKind::residual,
                                 {current, attention.value()}, {attention_residual.value()})
                   .has_value() ||
          !builder.add_operation("layer_" + index(layer) + "_post_norm", OperationKind::rms_norm,
                                 std::move(post_norm_inputs), {post_norm.value()}, norm_attributes)
                   .has_value() ||
          !builder.add_operation("layer_" + index(layer) + "_ffn", OperationKind::gated_dense_ffn,
                                 std::move(ffn_inputs), {ffn.value()})
                   .has_value() ||
          !builder.add_operation("layer_" + index(layer) + "_output", OperationKind::residual,
                                 {attention_residual.value(), ffn.value()}, {output.value()})
                   .has_value()) {
        return base::Status::invalid_argument("Qwen3.8 feed-forward operation could not be emitted");
      }
      current = output.value();
    }

    const auto logits = builder.add_tensor("logits", logits_spec);
    if (!logits.has_value()) {
      base::Status error = logits.error();
      return error.with_context("Qwen3.8 logits");
    }
    const auto lm_head_weight = find_source_weight(source_weights, "lm_head.weight");
    if (!lm_head_weight.has_value()) {
      base::Status error = lm_head_weight.error();
      return error.with_context("Qwen3.8 LM head weight binding");
    }
    if (!builder.add_operation("lm_head", OperationKind::lm_head,
                               {current, lm_head_weight.value()}, {logits.value()})
             .has_value() ||
        !builder.add_entry_point("decode", {token_ids.value()}, {logits.value()}).ok()) {
      return base::Status::invalid_argument("Qwen3.8 output graph could not be emitted");
    }
    if (gated_delta_layers != 48 || full_attention_layers != 16) {
      return base::Status::internal("Qwen3.8 layer topology invariant was not emitted");
    }
    return std::move(builder).build();
  }

 private:
  static bool is_semantic_parameter(std::string_view role) noexcept {
    return role == "embedding" || role == "lm_head" || role == "normalization" ||
           role == "attention" || role == "feed_forward" || role == "weight" || role == "bias";
  }

  static base::Result<ir::semantic::TensorSpec> source_tensor_spec(
      const compiler::SourceTensorRecord& record, ir::semantic::Shape shape) {
    ir::semantic::DType dtype;
    ir::semantic::QuantizationIntent quantization = ir::semantic::QuantizationIntent::none;
    if (record.dtype == "F32") {
      dtype = ir::semantic::DType::f32;
    } else if (record.dtype == "F16") {
      dtype = ir::semantic::DType::f16;
    } else if (record.dtype == "BF16") {
      dtype = ir::semantic::DType::bf16;
    } else if (record.dtype == "I8") {
      dtype = ir::semantic::DType::int8;
    } else if (record.dtype == "I32") {
      dtype = ir::semantic::DType::int32;
    } else if (record.dtype == "U8" || record.dtype == "F8_E4M3") {
      // These records are packed/scaled storage for a semantic low-precision weight. The
      // storage policy remains outside Semantic IR; the intent is the only semantic fact here.
      dtype = ir::semantic::DType::int4;
      quantization = ir::semantic::QuantizationIntent::symmetric;
    } else {
      return base::Status::unsupported("source tensor dtype is not supported by semantic IR: " +
                                      record.dtype);
    }
    return ir::semantic::TensorSpec{std::move(shape), dtype, quantization,
                                    ir::semantic::TensorRole::weight};
  }

  static base::Result<ir::semantic::TensorId> find_source_weight(
      const std::vector<std::pair<std::string, ir::semantic::TensorId>>& source_weights,
      std::string_view name) {
    for (const auto& source_weight : source_weights) {
      if (source_weight.first == name) return source_weight.second;
    }
    return base::Status::failed_precondition("required source tensor is absent: " + std::string{name});
  }

  static std::string index(std::uint32_t value) {
    if (value < 10) return "0" + std::to_string(value);
    return std::to_string(value);
  }
};

}  // namespace superinfer::frontends::qwen38
