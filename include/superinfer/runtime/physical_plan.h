#pragma once

#include <superinfer/ir/physical_plan.hpp>

namespace superinfer::runtime {

/** Runtime spelling for the immutable physical representation; no fourth IR is introduced. */
using PhysicalPlan = ir::physical::Plan;
using PhysicalPlanBuilder = ir::physical::PlanBuilder;

}  // namespace superinfer::runtime

