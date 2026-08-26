#include <superinfer/compiler/memory_planner.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iterator>
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
  const auto result = MemoryPlanner{128, 64}.plan(requests);
  assert(result.has_value());
  std::ifstream input(std::filesystem::path{SUPERINFER_SOURCE_DIR} /
                      "tests/golden/sm120/compiler/memory-basic.txt");
  assert(input.good());
  const std::string expected{std::istreambuf_iterator<char>{input}, {}};
  assert(result.value().dump() == expected);
  return 0;
}
