#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include <superinfer/base/result.hpp>

namespace superinfer::compiler {

/**
 * Versioned target facts used by offline compilation and runtime compatibility checks.
 *
 * The profile owns copied metadata and is safe to pass by value. It performs no device queries;
 * live probes populate an equivalent value outside the execution hot path. Compatibility checks
 * are deterministic, do not allocate device memory, and fail closed on unsupported targets.
 */
struct TargetProfile final {
  std::string target_name;
  std::uint32_t compute_capability{0};
  std::string kernel_catalog;
  std::uint64_t device_memory_bytes{0};
  std::uint64_t required_alignment{16};
  std::uint32_t max_threads_per_block{1024};
  std::uint32_t registers_per_block{65536};
  std::uint64_t shared_memory_per_block{0};

  /** Creates a declared RTX 5090 profile for CPU-only/offline compilation. */
  static TargetProfile offline_sm120a(std::uint64_t device_memory_bytes,
                                      std::string kernel_catalog) {
    TargetProfile profile;
    profile.target_name = "sm_120a";
    profile.compute_capability = 120;
    profile.kernel_catalog = std::move(kernel_catalog);
    profile.device_memory_bytes = device_memory_bytes;
    profile.required_alignment = 16;
    profile.max_threads_per_block = 1024;
    profile.registers_per_block = 65536;
    profile.shared_memory_per_block = 228U << 10U;
    return profile;
  }

  /** Rejects incomplete or non-sm120a profiles before they can affect a plan. */
  [[nodiscard]] base::Status validate() const {
    if (target_name.empty() || kernel_catalog.empty() || device_memory_bytes == 0) {
      return base::Status::invalid_argument("target profile is incomplete");
    }
    if (target_name != "sm_120a" || compute_capability != 120) {
      return base::Status::unsupported("target profile is not NVIDIA sm_120a");
    }
    if (required_alignment == 0 || max_threads_per_block == 0 || registers_per_block == 0 ||
        shared_memory_per_block == 0) {
      return base::Status::invalid_argument("target profile limits are incomplete");
    }
    return {};
  }

  /** Compares a compile-time requirement with an observed runtime profile. */
  [[nodiscard]] base::Status compatible_with(const TargetProfile& observed) const {
    base::Status required_status = validate();
    if (!required_status.ok()) return required_status.with_context("required target");
    base::Status observed_status = observed.validate();
    if (!observed_status.ok()) return observed_status.with_context("observed target");
    if (target_name != observed.target_name || compute_capability != observed.compute_capability) {
      return base::Status::unsupported("target capability does not match physical plan");
    }
    if (kernel_catalog != observed.kernel_catalog) {
      return base::Status::failed_precondition("kernel catalog ABI does not match physical plan");
    }
    if (observed.device_memory_bytes < device_memory_bytes) {
      return base::Status::resource_exhausted("observed device memory is below plan requirement");
    }
    return {};
  }

  /** Returns the stable identity serialized into a Physical Plan fingerprint. */
  [[nodiscard]] std::string fingerprint() const {
    return target_name + ":" + std::to_string(compute_capability) + ":" + kernel_catalog;
  }
};

}  // namespace superinfer::compiler
