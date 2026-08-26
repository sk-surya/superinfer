#include <sm120/kernels/baseline/provider.h>

#include <cassert>

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
  return 0;
}
