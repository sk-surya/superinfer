#pragma once

#include <string_view>
#include <cstdint>
#include <type_traits>

#include <superinfer/base/status.hpp>

namespace superinfer::compiler {

enum class Representation { semantic_ir, lowered_ir };

/** Deterministic pass contract and the representation it reads. */
struct PassDescriptor final {
  Representation input;
  std::string_view name;
  std::string_view effects;
  std::string_view invalidated_analyses;
  std::uint32_t version{1};
  std::string_view configuration{};
  std::string_view preconditions{};
  std::string_view postconditions{};
  bool deterministic{true};
};

/** Applies one verified, deterministic graph transformation during compilation. */
class GraphPass {
 public:
  virtual ~GraphPass() = default;
  virtual PassDescriptor descriptor() const noexcept = 0;
  virtual base::Status apply() const = 0;
};

template <typename T>
inline constexpr bool is_graph_pass = std::is_base_of_v<GraphPass, T>;

}  // namespace superinfer::compiler
