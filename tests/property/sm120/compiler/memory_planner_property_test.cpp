#include <superinfer/compiler/memory_planner.h>

#include <cassert>
#include <cstdint>
#include <vector>

int main() {
  using namespace superinfer::compiler;
  std::uint32_t seed = 0x51A7U;
  for (std::uint32_t round = 0; round < 128; ++round) {
    std::vector<AllocationRequest> requests;
    for (std::uint64_t id = 0; id < 12; ++id) {
      seed = seed * 1664525U + 1013904223U;
      const std::uint64_t first = (seed >> 4U) % 8U;
      seed = seed * 1664525U + 1013904223U;
      const std::uint64_t duration = 1U + ((seed >> 8U) % 5U);
      seed = seed * 1664525U + 1013904223U;
      const std::uint64_t bytes = 8U + ((seed >> 12U) % 8U) * 8U;
      requests.push_back({id, "r" + std::to_string(id), ArenaKind::device, bytes, 16,
                          {first, first + duration}});
    }
    const auto result = MemoryPlanner{1U << 20U, 1U << 20U}.plan(requests);
    assert(result.has_value());
    assert(result.value().verify().ok());
  }
  return 0;
}
