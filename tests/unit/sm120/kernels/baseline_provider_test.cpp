#include <sm120/kernels/baseline/provider.h>

#include <array>
#include <cassert>
#include <string_view>

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

  constexpr std::array<std::pair<std::string_view, std::uint64_t>, 5> operations{{
      {"copy", 1}, {"residual", 4}, {"rms_norm", 5}, {"layer_norm", 6}, {"embedding", 7}}};
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
  for (const std::string_view operation : {"cast", "elementwise", "rope", "matmul", "attention",
                                            "moe_route", "activation", "sampling"}) {
    const auto candidates = provider.enumerate({operation, 120});
    assert(!candidates.has_value());
    assert(candidates.error().code() == superinfer::base::StatusCode::unsupported);
  }
  return 0;
}
