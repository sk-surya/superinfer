#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <superinfer/base/result.hpp>
#include <superinfer/ir/semantic_ir.hpp>

namespace superinfer::compiler {

/**
 * One converter-authenticated source tensor.
 *
 * The payload range is relative to the artifact payload section. It is metadata only: frontends
 * may use the record to bind semantic weights, but they do not choose a storage layout or device
 * allocation from it.
 */
struct SourceTensorRecord final {
  std::string name;
  std::string role;
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::uint64_t payload_offset{0};
  std::uint64_t payload_bytes{0};
};

/** Validated source identity and tensor inventory supplied by a converter. */
struct SourceInventory final {
  std::string identity;
  std::uint64_t tensor_count{0};
  std::string tensor_inventory_sha256;
  std::vector<SourceTensorRecord> tensors;

  /** Verifies the bounded, duplicate-free metadata contract before frontend emission. */
  [[nodiscard]] base::Status validate() const {
    constexpr std::size_t kMaximumTensors = 1U << 20U;
    if (identity.empty()) return base::Status::invalid_argument("source identity is empty");
    if (tensor_count != tensors.size()) {
      return base::Status::invalid_argument("source tensor count does not match records");
    }
    if (tensors.size() > kMaximumTensors) {
      return base::Status::resource_exhausted("source tensor count exceeds limit");
    }
    for (std::size_t index = 0; index < tensors.size(); ++index) {
      const SourceTensorRecord& tensor = tensors[index];
      if (tensor.name.empty() || tensor.role.empty() || tensor.dtype.empty() || tensor.shape.empty()) {
        return base::Status::invalid_argument("source tensor record is incomplete: " + tensor.name);
      }
      for (const std::uint64_t dimension : tensor.shape) {
        if (dimension == 0) {
          return base::Status::invalid_argument("source tensor has a zero dimension: " + tensor.name);
        }
      }
      if (tensor.payload_bytes == 0) {
        return base::Status::invalid_argument("source tensor has an empty payload: " + tensor.name);
      }
      if (tensor.payload_offset > std::numeric_limits<std::uint64_t>::max() - tensor.payload_bytes) {
        return base::Status::out_of_range("source tensor payload range overflows: " + tensor.name);
      }
      for (std::size_t prior = 0; prior < index; ++prior) {
        if (tensors[prior].name == tensor.name) {
          return base::Status::invalid_argument("duplicate source tensor: " + tensor.name);
        }
      }
    }
    return {};
  }

  /** Returns a borrowed record; the inventory owns it and must outlive the returned pointer. */
  [[nodiscard]] const SourceTensorRecord* find_tensor(std::string_view name) const noexcept {
    for (const SourceTensorRecord& tensor : tensors) {
      if (tensor.name == name) return &tensor;
    }
    return nullptr;
  }
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
