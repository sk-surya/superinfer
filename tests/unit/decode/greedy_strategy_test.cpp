#include <superinfer/decode/greedy_strategy.hpp>

#include <cassert>
#include <cstddef>
#include <array>
#include <limits>
#include <vector>

int main() {
  using superinfer::decode::DecodeStateView;
  using superinfer::decode::GreedyDecodeStrategy;
  using superinfer::decode::GreedyDecodeConfig;

  const GreedyDecodeStrategy strategy{GreedyDecodeConfig{7, 3, 10}};
  assert(strategy.requirements().workspace_bytes == 0);
  assert(!strategy.requirements().needs_verification_graph);
  std::vector<std::byte> state(GreedyDecodeStrategy::state_bytes());
  DecodeStateView view{state.data(), state.size()};
  assert(strategy.initialize(view).ok());
  const std::array<float, 10> logits{1.0F, 4.0F, 4.0F, -1.0F, 0.0F,
                                     0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  assert(strategy.select_token(logits).value() == 1);
  auto non_finite = logits;
  non_finite[3] = std::numeric_limits<float>::quiet_NaN();
  assert(!strategy.select_token(non_finite).error().ok());

  assert(strategy.accept(view, 2).ok());
  assert(!strategy.finished(view));
  assert(strategy.accept(view, 7).ok());
  assert(strategy.finished(view));
  assert(!strategy.accept(view, 2).ok());

  assert(strategy.initialize(view).ok());
  assert(strategy.accept(view, 2).ok());
  assert(strategy.accept(view, 3).ok());
  assert(strategy.accept(view, 4).ok());
  assert(strategy.finished(view));
  assert(strategy.initialize({state.data(), state.size() - 1}).code() ==
         superinfer::base::StatusCode::resource_exhausted);
  return 0;
}
