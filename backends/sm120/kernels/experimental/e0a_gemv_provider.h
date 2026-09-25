#pragma once

#include <cstdint>
#include <vector>

#include <sm120/kernels/baseline/provider.h>

namespace superinfer::sm120 {

/**
 * E0a donor-scheduled NVFP4 projection provider (S04 performance reset, D-022).
 *
 * Advertises kernel 29 (`nvfp4_gemv_rows_f32`) for `nvfp4_linear` when the target is sm_120a and the
 * geometry satisfies the donor schedule contract: `K % 512 == 0` (512-wide phase) and
 * `rows % 32 == 0` (8 warps x 4 rows per CTA). Every other operation and every non-conforming
 * projection is delegated unchanged to the retained provider (P7 baseline), so unsupported shapes
 * fall back safely instead of failing.
 *
 * It consumes the existing row-major `.sinf` packed weights, the existing row-major E4M3 block-scale
 * plane, and the existing FP32 activation buffer. No offline repack and no per-token conversion
 * kernel is required: only the execution schedule changed.
 *
 * Selection happens once at specialization time from compile-time geometry. No model-name branch, no
 * per-token environment branch.
 *
 * Thread safety: stateless and const; `enumerate` may be called concurrently.
 * Lifetime: the referenced fallback provider must outlive this provider.
 */
class E0aGemmProvider final : public kernels::KernelProvider {
 public:
  explicit E0aGemmProvider(const kernels::KernelProvider& fallback) noexcept : fallback_(fallback) {}

  base::Result<std::vector<kernels::KernelCandidate>> enumerate(
      const kernels::KernelQuery& query) const override {
    if (query.operation == "nvfp4_linear" && query.target_capability == 120 &&
        query.activation_elements != 0 && query.output_elements != 0 &&
        query.activation_elements % 512U == 0 && query.output_elements % 32U == 0) {
      return std::vector<kernels::KernelCandidate>{
          {base::KernelId{29}, "sm120.nvfp4-gemv-rows", true, 0}};
    }
    if (query.operation == "linear" && query.target_capability == 120 &&
        query.activation_elements != 0 && query.activation_elements % 4U == 0) {
      // E0a: specialized small-output FP32 control projection (GDN a/b). Same operands, same
      // recipe; only the schedule changed.
      return std::vector<kernels::KernelCandidate>{
          {base::KernelId{30}, "sm120.linear-f32-rows", true, 0}};
    }
    return fallback_.enumerate(query);
  }

 private:
  const kernels::KernelProvider& fallback_;
};

}  // namespace superinfer::sm120
