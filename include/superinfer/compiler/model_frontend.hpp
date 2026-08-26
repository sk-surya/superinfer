#pragma once

#include <cstdint>
#include <string>
#include <type_traits>

#include <superinfer/base/result.hpp>
#include <superinfer/ir/semantic_ir.hpp>

namespace superinfer::compiler {

/** Validated source identity and tensor inventory supplied by a converter. */
struct SourceInventory final {
  std::string identity;
  std::uint64_t tensor_count{0};
  std::string tensor_inventory_sha256;
};

/** Converts a source inventory into meaning-level IR without selecting physical kernels. */
class ModelFrontend {
 public:
  virtual ~ModelFrontend() = default;
  virtual base::Status validate(const SourceInventory&) const = 0;
  virtual base::Result<ir::semantic::Module> emit(const SourceInventory&) const = 0;
};

template <typename T>
inline constexpr bool is_model_frontend = std::is_base_of_v<ModelFrontend, T>;

}  // namespace superinfer::compiler
