#pragma once

#include <cstdint>
#include <string>

#include <superinfer/base/status.hpp>

namespace superinfer::ir::semantic {

/** Meaning-level graph shell; it has no CUDA, offsets, addresses, or launch choices. */
class Module final {
 public:
  static Module empty() noexcept { return {}; }
  [[nodiscard]] std::uint32_t version() const noexcept { return version_; }
  [[nodiscard]] base::Status verify() const { return {}; }
  [[nodiscard]] std::string dump() const { return "semantic-ir:v" + std::to_string(version_); }

 private:
  std::uint32_t version_{1};
};

}  // namespace superinfer::ir::semantic

