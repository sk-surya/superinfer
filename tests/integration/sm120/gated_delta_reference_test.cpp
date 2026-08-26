#include <sm120/kernels/baseline/gated_delta_reference.h>

#include <cassert>
#include <cmath>
#include <vector>

int main() {
  using superinfer::sm120::GatedDeltaProblem;
  using superinfer::sm120::GatedDeltaReference;

  GatedDeltaProblem problem;
  problem.sequence_length = 2;
  problem.key_head_count = 1;
  problem.value_head_count = 2;
  problem.key_dimension = 2;
  problem.value_dimension = 2;
  problem.query = {1.0F, 0.0F, 0.0F, 1.0F};
  problem.key = {1.0F, 0.0F, 0.0F, 1.0F};
  problem.value = {2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
  problem.log_decay = {0.0F, 0.0F, 0.0F, 0.0F};
  problem.beta = {1.0F, 1.0F, 1.0F, 1.0F};

  const auto full = GatedDeltaReference::run(problem);
  assert(full.has_value());
  assert(full.value().output.size() == 8);
  assert(std::fabs(full.value().output[0] - 2.0F / std::sqrt(2.0F)) < 1.0e-5F);
  assert(std::fabs(full.value().output[1] - 3.0F / std::sqrt(2.0F)) < 1.0e-5F);

  GatedDeltaProblem first = problem;
  first.sequence_length = 1;
  first.query.resize(2);
  first.key.resize(2);
  first.value.resize(4);
  first.log_decay.resize(2);
  first.beta.resize(2);
  const auto first_step = GatedDeltaReference::run(first);
  assert(first_step.has_value());
  GatedDeltaProblem second = problem;
  second.sequence_length = 1;
  second.query.erase(second.query.begin(), second.query.begin() + 2);
  second.key.erase(second.key.begin(), second.key.begin() + 2);
  second.value.erase(second.value.begin(), second.value.begin() + 4);
  second.log_decay.erase(second.log_decay.begin(), second.log_decay.begin() + 2);
  second.beta.erase(second.beta.begin(), second.beta.begin() + 2);
  second.initial_state = first_step.value().final_state;
  const auto second_step = GatedDeltaReference::run(second);
  assert(second_step.has_value());
  assert(std::fabs(second_step.value().output[0] - full.value().output[4]) < 1.0e-5F);
  assert(std::fabs(second_step.value().output[1] - full.value().output[5]) < 1.0e-5F);

  second.query.pop_back();
  assert(!GatedDeltaReference::run(second).has_value());
  return 0;
}
