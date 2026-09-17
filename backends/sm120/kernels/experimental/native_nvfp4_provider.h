#pragma once

#include <cstdint>
#include <vector>

#include <sm120/kernels/baseline/provider.h>

namespace superinfer::sm120 {

/**
 * Experimental SM120 native block-scaled NVFP4 MMA provider (S04-P8RQ).
 *
 * Advertises the native `nvfp4_linear` kernel (id 27) only when the target capability is 120 and the
 * projection geometry satisfies the native contract (`K % 64 == 0`, `rows % 16 == 0`; the M16N8K64
 * tile and the four 16-wide scale blocks). Every other operation, and every projection that does not
 * satisfy the contract, is delegated unchanged to the retained P7 baseline provider, so unsupported
 * shapes fall back safely instead of failing.
 *
 * The decision is made once at specialization time from compile-time geometry. There is no model-name
 * branch and no per-token environment or policy branch; an experiment-level selector may choose this
 * provider instead of the baseline in order to produce A/B evidence.
 *
 * The provider only uses the existing row-major `.sinf` weight layout. It intentionally does not
 * request an MMA-native repacked weight layout: changing StoragePolicy before D-021 passes would mix
 * a numerical experiment with a storage-architecture change.
 *
 * Thread safety: stateless and const; `enumerate` may be called concurrently.
 * Lifetime: the referenced fallback provider must outlive this provider.
 */
class NativeNvfp4Provider final : public kernels::KernelProvider {
 public:
  explicit NativeNvfp4Provider(const kernels::KernelProvider& fallback,
                               bool canonical_two_level = false) noexcept
      : fallback_(fallback), canonical_two_level_(canonical_two_level) {}

  base::Result<std::vector<kernels::KernelCandidate>> enumerate(
      const kernels::KernelQuery& query) const override {
    if (query.operation == "nvfp4_linear" && query.target_capability == 120 &&
        query.activation_elements != 0 && query.output_elements != 0 &&
        query.activation_elements % 64U == 0 && query.output_elements % 16U == 0) {
      // Workspace holds the packed activation (K/2, 16-byte aligned), the UE4M3 block scales
      // (K/16), and for the canonical two-level form an FP32 activation global scale.
      const std::uint64_t scratch = (query.activation_elements / 2U + 15U) / 16U * 16U +
                                    (query.activation_elements / 16U + 15U) / 16U * 16U + 16U;
      return std::vector<kernels::KernelCandidate>{{base::KernelId{canonical_two_level_ ? 28U
                                                                                       : 27U},
                                                    canonical_two_level_
                                                        ? "sm120.native-mma-nvfp4-two-level"
                                                        : "sm120.native-mma-nvfp4",
                                                    true, scratch}};
    }
    return fallback_.enumerate(query);
  }

 private:
  const kernels::KernelProvider& fallback_;
  bool canonical_two_level_{false};
};

}  // namespace superinfer::sm120
