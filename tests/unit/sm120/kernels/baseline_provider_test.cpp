#include <sm120/kernels/baseline/provider.h>

#include <array>
#include <cassert>
#include <string_view>
#include <vector>

int main() {
  superinfer::sm120::BaselineProvider provider;
  const auto norm = provider.enumerate({"rms_norm", 120});
  assert(norm.has_value());
  assert(norm.value().size() == 1);
  assert(norm.value().front().id.value() == 5);
  assert(norm.value().front().deterministic);

  const auto unsupported_target = provider.enumerate({"rms_norm", 89});
  assert(!unsupported_target.has_value());
  assert(unsupported_target.error().code() == superinfer::base::StatusCode::unsupported);

  const auto unsupported_operation = provider.enumerate({"unknown", 120});
  assert(!unsupported_operation.has_value());
  assert(unsupported_operation.error().code() == superinfer::base::StatusCode::unsupported);

  constexpr std::array<std::pair<std::string_view, std::uint64_t>, 13> operations{{
      {"copy", 1}, {"residual", 4}, {"rms_norm", 5}, {"layer_norm", 6}, {"embedding", 7},
      {"nvfp4_dequantize", 9}, {"lm_head", 10}, {"gated_dense_ffn", 11},
      {"nvfp4_linear", 13}, {"attention", 14}, {"gated_delta_attention", 15},
      {"silu_mul", 18}, {"sigmoid_mul", 19}}};
  for (const auto& operation : operations) {
    const auto candidates = provider.enumerate({operation.first, 120});
    assert(candidates.has_value());
    assert(candidates.value().size() == 1);
    assert(candidates.value().front().id.value() == operation.second);
    assert(candidates.value().front().deterministic);
    assert(candidates.value().front().workspace_bytes == 0);
  }
  const auto bf16_embedding = provider.enumerate({"embedding", 120, "bf16"});
  assert(bf16_embedding.has_value());
  assert(bf16_embedding.value().size() == 1);
  assert(bf16_embedding.value().front().id.value() == 8);
  assert(bf16_embedding.value().front().deterministic);
  assert(bf16_embedding.value().front().workspace_bytes == 0);
  const auto unsupported_embedding_dtype = provider.enumerate({"embedding", 120, "int4"});
  assert(!unsupported_embedding_dtype.has_value());
  assert(unsupported_embedding_dtype.error().code() == superinfer::base::StatusCode::unsupported);
  const auto unsupported_lm_dtype = provider.enumerate({"lm_head", 120, "int4"});
  assert(!unsupported_lm_dtype.has_value());
  assert(unsupported_lm_dtype.error().code() == superinfer::base::StatusCode::unsupported);
  const auto bf16_norm = provider.enumerate({"rms_norm", 120, "bf16"});
  assert(bf16_norm.has_value());
  assert(bf16_norm.value().size() == 1);
  assert(bf16_norm.value().front().id.value() == 12);
  const auto wrong_attention_arity = provider.enumerate({"attention", 120, "f32", 5});
  assert(!wrong_attention_arity.has_value());
  assert(wrong_attention_arity.error().code() == superinfer::base::StatusCode::unsupported);
  const auto wrong_delta_arity = provider.enumerate({"gated_delta_attention", 120, "f32", 6});
  assert(!wrong_delta_arity.has_value());
  assert(wrong_delta_arity.error().code() == superinfer::base::StatusCode::unsupported);
  const std::vector<std::string_view> bf16_residual_operands{"bf16", "bf16", "bf16"};
  const auto wrong_residual_dtype = provider.enumerate(
      {"residual", 120, "f32", bf16_residual_operands.size(), bf16_residual_operands});
  assert(!wrong_residual_dtype.has_value());
  assert(wrong_residual_dtype.error().code() == superinfer::base::StatusCode::unsupported);
  const auto unsupported_attention_gate = provider.enumerate(
      {"attention", 120, "f32", 4, {}, true});
  assert(!unsupported_attention_gate.has_value());
  assert(unsupported_attention_gate.error().code() == superinfer::base::StatusCode::unsupported);
  const std::vector<std::string_view> bf16_to_f32{"bf16", "f32"};
  const auto cast_bf16_to_f32 = provider.enumerate(
      {"cast", 120, "f32", bf16_to_f32.size(), bf16_to_f32});
  assert(cast_bf16_to_f32.has_value());
  assert(cast_bf16_to_f32.value().front().id.value() == 16);
  const std::vector<std::string_view> f32_to_bf16{"f32", "bf16"};
  const auto cast_f32_to_bf16 = provider.enumerate(
      {"cast", 120, "f32", f32_to_bf16.size(), f32_to_bf16});
  assert(cast_f32_to_bf16.has_value());
  assert(cast_f32_to_bf16.value().front().id.value() == 17);
  const std::vector<std::string_view> invalid_cast{"bf16", "bf16"};
  assert(!provider.enumerate({"cast", 120, "f32", invalid_cast.size(), invalid_cast}).has_value());
  for (const std::string_view operation : {"elementwise", "rope", "matmul", "moe_route",
                                            "activation", "sampling"}) {
    const auto candidates = provider.enumerate({operation, 120});
    assert(!candidates.has_value());
    assert(candidates.error().code() == superinfer::base::StatusCode::unsupported);
  }
  return 0;
}
