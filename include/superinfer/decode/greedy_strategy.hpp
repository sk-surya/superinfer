#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>

#include <superinfer/decode/decode_strategy.hpp>

namespace superinfer::decode {

struct GreedyDecodeConfig final {
  std::uint32_t eos_token{0};
  std::uint32_t max_tokens{0};
  std::uint32_t vocabulary_size{0};
};

struct GreedyState final {
  std::uint32_t generated_tokens{0};
  std::uint32_t last_token{0};
  std::uint8_t finished{0};
  std::uint8_t reserved[3]{0, 0, 0};
};

/** Deterministic argmax decode policy with caller-owned fixed-size state. */
class GreedyDecodeStrategy final : public DecodeStrategy {
 public:
  explicit GreedyDecodeStrategy(GreedyDecodeConfig config) : config_(config) {}

  static constexpr std::uint64_t state_bytes() noexcept { return sizeof(GreedyState); }

  DecodeRequirements requirements() const noexcept override { return {0, false}; }

  base::Status initialize(DecodeStateView view) const override {
    if (config_.max_tokens == 0 || config_.vocabulary_size == 0 || config_.eos_token >= config_.vocabulary_size) {
      return base::Status::invalid_argument("greedy decode configuration is incomplete");
    }
    if (view.data == nullptr || view.size < state_bytes()) {
      return base::Status::resource_exhausted("greedy decode state buffer is too small");
    }
    const GreedyState state{};
    std::memcpy(view.data, &state, sizeof(state));
    return {};
  }

  base::Result<std::uint32_t> select_token(std::span<const float> logits) const override {
    if (logits.size() != config_.vocabulary_size || logits.empty()) {
      return base::Status::invalid_argument("greedy logits do not match vocabulary size");
    }
    std::uint32_t best = 0;
    float best_value = logits[0];
    if (!std::isfinite(best_value)) return base::Status::invalid_argument("greedy logits contain non-finite values");
    for (std::uint32_t index = 1; index < logits.size(); ++index) {
      const float value = logits[index];
      if (!std::isfinite(value)) return base::Status::invalid_argument("greedy logits contain non-finite values");
      if (value > best_value) {
        best = index;
        best_value = value;
      }
    }
    return best;
  }

  base::Status accept(DecodeStateView view, std::uint32_t token) const noexcept {
    GreedyState state{};
    const base::Status status = load(view, state);
    if (!status.ok()) return status;
    if (state.finished != 0) return base::Status::failed_precondition("greedy decode is already finished");
    if (token >= config_.vocabulary_size) return base::Status::out_of_range("selected token exceeds vocabulary");
    state.last_token = token;
    ++state.generated_tokens;
    state.finished = static_cast<std::uint8_t>(token == config_.eos_token ||
                                                state.generated_tokens >= config_.max_tokens);
    std::memcpy(view.data, &state, sizeof(state));
    return {};
  }

  [[nodiscard]] bool finished(DecodeStateView view) const noexcept {
    GreedyState state{};
    if (!load(view, state).ok()) return false;
    return state.finished != 0;
  }

 private:
  static base::Status load(DecodeStateView view, GreedyState& state) noexcept {
    if (view.data == nullptr || view.size < state_bytes()) {
      return base::Status::resource_exhausted("greedy decode state buffer is too small");
    }
    std::memcpy(&state, view.data, sizeof(state));
    return {};
  }

  GreedyDecodeConfig config_;
};

}  // namespace superinfer::decode
