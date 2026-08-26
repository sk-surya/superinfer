#include <sm120/kernels/baseline/provider.h>

#include <cassert>
#include <array>
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

  constexpr std::array<std::string_view, 13> operations{
      "copy", "cast", "elementwise", "residual", "rms_norm", "layer_norm", "rope",
      "matmul", "embedding", "attention", "moe_route", "activation", "sampling"};
  for (std::size_t index = 0; index < operations.size(); ++index) {
    const auto candidates = provider.enumerate({operations[index], 120});
    assert(candidates.has_value());
    assert(candidates.value().size() == 1);
    assert(candidates.value().front().id.value() == index + 1);
    assert(candidates.value().front().deterministic);
    assert(candidates.value().front().workspace_bytes == 0);
  }
  return 0;
}
