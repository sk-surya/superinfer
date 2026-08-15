#pragma once

#include <cstdint>
#include <string>

#include <superinfer/base/status.hpp>

namespace superinfer::ir::physical {

/**
 * Immutable execution contract. Construction is compiler-owned; runtime consumers only retain
 * a const reference and may inspect its version after validation.
 */
class Plan final {
 public:
  static Plan empty() noexcept { return {}; }
  [[nodiscard]] std::uint32_t version() const noexcept { return version_; }
  [[nodiscard]] base::Status verify() const { return {}; }
  [[nodiscard]] std::string dump() const { return "physical-plan:v" + std::to_string(version_); }

 private:
  explicit Plan(std::uint32_t version) noexcept : version_(version) {}
  Plan() = default;
  std::uint32_t version_{1};
};

}  // namespace superinfer::ir::physical

