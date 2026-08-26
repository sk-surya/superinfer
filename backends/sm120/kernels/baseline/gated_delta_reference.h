#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include <superinfer/base/result.hpp>

namespace superinfer::sm120 {

/** Host-only input contract for the recurrent gated-delta rule oracle. */
struct GatedDeltaProblem final {
  std::uint32_t sequence_length{0};
  std::uint32_t key_head_count{0};
  std::uint32_t value_head_count{0};
  std::uint32_t key_dimension{0};
  std::uint32_t value_dimension{0};
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> log_decay;
  std::vector<float> beta;
  std::vector<float> initial_state;
  bool normalize_query_key{true};
};

struct GatedDeltaResult final {
  std::vector<float> output;
  std::vector<float> final_state;
};

/**
 * Independent CPU oracle for recurrent Gated DeltaNet state updates.
 *
 * The state layout is [value_head][key_dimension][value_dimension]. Query/key heads are repeated
 * across value-head groups. The oracle owns result vectors and is never used as a hot-path
 * candidate; its purpose is split-step and provider differential testing.
 */
class GatedDeltaReference final {
 public:
  static base::Result<GatedDeltaResult> run(const GatedDeltaProblem& problem) {
    const base::Status shape_status = validate(problem);
    if (!shape_status.ok()) return shape_status;
    const std::size_t state_size = static_cast<std::size_t>(problem.value_head_count) *
                                   problem.key_dimension * problem.value_dimension;
    const std::size_t output_size = static_cast<std::size_t>(problem.sequence_length) *
                                    problem.value_head_count * problem.value_dimension;
    std::vector<float> state = problem.initial_state;
    if (state.empty()) state.assign(state_size, 0.0F);
    std::vector<float> output(output_size, 0.0F);
    const std::uint32_t heads_per_value = problem.value_head_count / problem.key_head_count;
    const float scale = 1.0F / std::sqrt(static_cast<float>(problem.key_dimension));

    for (std::uint32_t time = 0; time < problem.sequence_length; ++time) {
      for (std::uint32_t value_head = 0; value_head < problem.value_head_count; ++value_head) {
        const std::uint32_t key_head = value_head / heads_per_value;
        const std::size_t q_base =
            (static_cast<std::size_t>(time) * problem.key_head_count + key_head) * problem.key_dimension;
        const std::size_t k_base = q_base;
        const std::size_t v_base =
            (static_cast<std::size_t>(time) * problem.value_head_count + value_head) * problem.value_dimension;
        const std::size_t state_base = static_cast<std::size_t>(value_head) * problem.key_dimension *
                                       problem.value_dimension;
        const float decay = std::exp(problem.log_decay[static_cast<std::size_t>(time) *
                                                        problem.value_head_count + value_head]);
        for (std::uint32_t key_index = 0; key_index < problem.key_dimension; ++key_index) {
          for (std::uint32_t value_index = 0; value_index < problem.value_dimension; ++value_index) {
            state[state_base + static_cast<std::size_t>(key_index) * problem.value_dimension + value_index] *= decay;
          }
        }

        std::vector<float> normalized_query(problem.key_dimension);
        std::vector<float> normalized_key(problem.key_dimension);
        float query_norm = 0.0F;
        float key_norm = 0.0F;
        for (std::uint32_t key_index = 0; key_index < problem.key_dimension; ++key_index) {
          const float q = problem.query[q_base + key_index];
          const float k = problem.key[k_base + key_index];
          query_norm += q * q;
          key_norm += k * k;
          normalized_query[key_index] = q;
          normalized_key[key_index] = k;
        }
        if (problem.normalize_query_key) {
          const float q_scale = 1.0F / std::sqrt(query_norm + 1.0e-6F);
          const float k_scale = 1.0F / std::sqrt(key_norm + 1.0e-6F);
          for (float& q : normalized_query) q *= q_scale;
          for (float& k : normalized_key) k *= k_scale;
        }

        const float beta = problem.beta[static_cast<std::size_t>(time) * problem.value_head_count + value_head];
        for (std::uint32_t value_index = 0; value_index < problem.value_dimension; ++value_index) {
          float key_value = 0.0F;
          for (std::uint32_t key_index = 0; key_index < problem.key_dimension; ++key_index) {
            key_value += state[state_base + static_cast<std::size_t>(key_index) * problem.value_dimension + value_index] *
                         normalized_key[key_index];
          }
          const float delta = (problem.value[v_base + value_index] - key_value) * beta;
          for (std::uint32_t key_index = 0; key_index < problem.key_dimension; ++key_index) {
            state[state_base + static_cast<std::size_t>(key_index) * problem.value_dimension + value_index] +=
                normalized_key[key_index] * delta;
          }
        }
        const std::size_t output_base =
            (static_cast<std::size_t>(time) * problem.value_head_count + value_head) * problem.value_dimension;
        for (std::uint32_t value_index = 0; value_index < problem.value_dimension; ++value_index) {
          float result = 0.0F;
          for (std::uint32_t key_index = 0; key_index < problem.key_dimension; ++key_index) {
            result += state[state_base + static_cast<std::size_t>(key_index) * problem.value_dimension + value_index] *
                      normalized_query[key_index];
          }
          output[output_base + value_index] = result * scale;
        }
      }
    }
    return GatedDeltaResult{std::move(output), std::move(state)};
  }

 private:
  static base::Status validate(const GatedDeltaProblem& problem) {
    if (problem.sequence_length == 0 || problem.key_head_count == 0 || problem.value_head_count == 0 ||
        problem.key_dimension == 0 || problem.value_dimension == 0 ||
        problem.value_head_count % problem.key_head_count != 0) {
      return base::Status::invalid_argument("gated delta dimensions are incomplete");
    }
    const std::size_t qk_size = static_cast<std::size_t>(problem.sequence_length) *
                                problem.key_head_count * problem.key_dimension;
    const std::size_t value_size = static_cast<std::size_t>(problem.sequence_length) *
                                   problem.value_head_count * problem.value_dimension;
    const std::size_t gate_size = static_cast<std::size_t>(problem.sequence_length) * problem.value_head_count;
    const std::size_t state_size = static_cast<std::size_t>(problem.value_head_count) *
                                   problem.key_dimension * problem.value_dimension;
    if (problem.query.size() != qk_size || problem.key.size() != qk_size ||
        problem.value.size() != value_size || problem.log_decay.size() != gate_size ||
        problem.beta.size() != gate_size ||
        (!problem.initial_state.empty() && problem.initial_state.size() != state_size)) {
      return base::Status::invalid_argument("gated delta tensor shapes do not match dimensions");
    }
    for (const float value : problem.query) {
      if (!std::isfinite(value)) return base::Status::invalid_argument("gated delta query is non-finite");
    }
    for (const float value : problem.key) {
      if (!std::isfinite(value)) return base::Status::invalid_argument("gated delta key is non-finite");
    }
    for (const float value : problem.value) {
      if (!std::isfinite(value)) return base::Status::invalid_argument("gated delta value is non-finite");
    }
    for (const float value : problem.log_decay) {
      if (!std::isfinite(value)) return base::Status::invalid_argument("gated delta log decay is non-finite");
    }
    for (const float value : problem.beta) {
      if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
        return base::Status::invalid_argument("gated delta beta is outside [0,1]");
      }
    }
    return {};
  }
};

}  // namespace superinfer::sm120
