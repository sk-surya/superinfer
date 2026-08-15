#include <superinfer/ir/physical_plan.hpp>
#include <superinfer/runtime/executor_contract.hpp>

int main() {
  const auto plan = superinfer::ir::physical::Plan::empty();
  const superinfer::runtime::ExecutorContract executor{plan};
  return executor.plan_version() == plan.version() ? 0 : 1;
}

