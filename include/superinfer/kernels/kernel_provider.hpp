#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <superinfer/base/ids.hpp>
#include <superinfer/base/result.hpp>

namespace superinfer::kernels {

/** Capability query expressed in operation/layout/storage facts, never model-family identity. */
struct KernelQuery final {
  KernelQuery(std::string_view operation_value, std::uint32_t target_capability_value,
              std::string_view storage_dtype_value = "f32", std::size_t operand_count_value = 0,
              std::vector<std::string_view> operand_dtypes_value = {})
      : operation(operation_value),
        target_capability(target_capability_value),
        storage_dtype(storage_dtype_value),
        operand_count(operand_count_value),
        operand_dtypes(std::move(operand_dtypes_value)) {}

  std::string_view operation;
  std::uint32_t target_capability;
  /** Storage dtype of the operation's weight operand when a provider needs it. */
  std::string_view storage_dtype{"f32"};
  /** Number of lowered operands, when the compiler has an explicit operation contract. */
  std::size_t operand_count{0};
  /** Storage dtypes in semantic operand order; an empty vector preserves legacy query callers. */
  std::vector<std::string_view> operand_dtypes;
};

/** Describes a candidate's correctness and resource envelope before selection. */
struct KernelCandidate final {
  base::KernelId id;
  std::string_view name;
  bool deterministic;
  std::uint64_t workspace_bytes;
};

/** Enumerates correct implementation candidates for a compiler-owned physical query. */
class KernelProvider {
 public:
  virtual ~KernelProvider() = default;
  virtual base::Result<std::vector<KernelCandidate>> enumerate(const KernelQuery&) const = 0;
};

template <typename T>
inline constexpr bool is_kernel_provider = std::is_base_of_v<KernelProvider, T>;

}  // namespace superinfer::kernels
