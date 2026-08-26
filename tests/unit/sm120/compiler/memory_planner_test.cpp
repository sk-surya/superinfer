#include <superinfer/compiler/memory_planner.h>

#include <cassert>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

int main() {
  using namespace superinfer::compiler;

  const std::vector<AllocationRequest> requests{
      {0, "activation_a", ArenaKind::device, 32, 16, {0, 2}},
      {1, "activation_b", ArenaKind::device, 32, 16, {2, 4}},
      {2, "activation_c", ArenaKind::device, 16, 16, {1, 3}},
      {3, "workspace", ArenaKind::workspace, 24, 16, {0, 4}},
  };
  MemoryPlanner planner{128, 64};
  const auto planned = planner.plan(requests);
  assert(planned.has_value());
  assert(planned.value().device_arena_bytes == 48);
  assert(planned.value().workspace_bytes == 24);
  assert(planned.value().allocations.size() == requests.size());
  assert(planned.value().allocations[0].offset == planned.value().allocations[1].offset);
  assert(planned.value().allocations[2].offset != planned.value().allocations[0].offset);
  assert(planned.value().verify().ok());
  assert(planned.value().dump().find("peak device=48 workspace=24") != std::string::npos);

  const auto reordered = planner.plan({requests[2], requests[0], requests[3], requests[1]});
  assert(reordered.has_value());
  assert(reordered.value().dump() == planned.value().dump());

  MemoryPlanner tiny_device{32, 64};
  assert(tiny_device.plan(requests).error().code() ==
         superinfer::base::StatusCode::resource_exhausted);

  const auto bad_alignment = planner.plan({{0, "bad", ArenaKind::device, 8, 0, {0, 1}}});
  assert(!bad_alignment.has_value());
  assert(bad_alignment.error().code() == superinfer::base::StatusCode::invalid_argument);

  const auto bad_lifetime = planner.plan({{0, "bad", ArenaKind::device, 8, 8, {2, 1}}});
  assert(!bad_lifetime.has_value());
  assert(bad_lifetime.error().code() == superinfer::base::StatusCode::invalid_argument);

  MemoryPlanner unlimited{std::numeric_limits<std::uint64_t>::max(), 64};
  const auto overflow = unlimited.plan({
      {0, "near_limit", ArenaKind::device, std::numeric_limits<std::uint64_t>::max() - 8, 8, {0, 2}},
      {1, "overflow", ArenaKind::device, 16, 8, {0, 1}},
  });
  assert(!overflow.has_value());
  assert(overflow.error().code() == superinfer::base::StatusCode::overflow);
  return 0;
}
