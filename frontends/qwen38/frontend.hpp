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
    "cab1e4afdb94d48c0a1cfe6ee3833b22d9ec856c077fb1eeef92f674032aa3ab";

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
    if (!builder.add_operation("embedding", OperationKind::embedding, {token_ids.value()},
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
      if (!input_norm.has_value() || !attention.has_value() || !attention_residual.has_value() ||
          !post_norm.has_value() || !ffn.has_value() || !output.has_value()) {
        return base::Status::resource_exhausted("Qwen3.8 frontend tensor emission failed");
      }
      if (!builder.add_operation("layer_" + index(layer) + "_input_norm", OperationKind::rms_norm,
                                 {current}, {input_norm.value()})
                   .has_value()) {
        return base::Status::invalid_argument("Qwen3.8 input norm operation could not be emitted");
      }
      const bool full_attention = (layer % 4U) == 3U;
      OperationAttributes attention_attributes;
      attention_attributes.num_heads = full_attention ? 24 : 48;
      attention_attributes.num_kv_heads = full_attention ? 4 : 16;
      attention_attributes.head_dimension = full_attention ? 256 : 128;
      attention_attributes.rope_dimension = full_attention ? 64 : 0;
      if (!full_attention) {
        attention_attributes.key_head_dimension = 128;
        attention_attributes.value_head_dimension = 128;
        attention_attributes.convolution_kernel_dimension = 4;
      }
      const OperationKind attention_kind = full_attention
                                               ? OperationKind::grouped_query_attention
                                               : OperationKind::gated_delta_attention;
      if (!builder.add_operation("layer_" + index(layer) + "_attention", attention_kind,
                                 {input_norm.value()}, {attention.value()}, attention_attributes)
                   .has_value()) {
        return base::Status::invalid_argument("Qwen3.8 attention operation could not be emitted");
      }
      if (full_attention) {
        ++full_attention_layers;
      } else {
        ++gated_delta_layers;
      }
      if (!builder.add_operation("layer_" + index(layer) + "_attention_residual",
                                 OperationKind::residual,
                                 {current, attention.value()}, {attention_residual.value()})
                   .has_value() ||
          !builder.add_operation("layer_" + index(layer) + "_post_norm", OperationKind::rms_norm,
                                 {attention_residual.value()}, {post_norm.value()})
                   .has_value() ||
          !builder.add_operation("layer_" + index(layer) + "_ffn", OperationKind::gated_dense_ffn,
                                 {post_norm.value()}, {ffn.value()})
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
    if (!builder.add_operation("lm_head", OperationKind::lm_head, {current}, {logits.value()})
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
  static std::string index(std::uint32_t value) {
    if (value < 10) return "0" + std::to_string(value);
    return std::to_string(value);
  }
};

}  // namespace superinfer::frontends::qwen38
