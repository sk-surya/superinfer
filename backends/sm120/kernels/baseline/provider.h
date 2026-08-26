#pragma once

#include <algorithm>
#include <cstdint>
#include <string_view>
#include <vector>

#include <superinfer/kernels/kernel_provider.hpp>

namespace superinfer::sm120 {

/**
 * Correctness-first capability registry for the currently implemented baseline operation set.
 *
 * The provider owns no device state and only returns immutable candidate metadata. Queries are
 * operation/target based; model identity is neither accepted nor inspected.
 */
class BaselineProvider final : public kernels::KernelProvider {
 public:
  base::Result<std::vector<kernels::KernelCandidate>> enumerate(
      const kernels::KernelQuery& query) const override {
    if (query.target_capability != 120) {
      return base::Status::unsupported("baseline provider requires sm_120a");
    }
    // Keep this registry synchronized with the CUDA resolver. An advertised candidate must be
    // executable; operations without a physical baseline remain explicit selector misses until
    // their command contract and differential fixture exist.
    if (query.operation == "copy") {
      if (!query.operand_dtypes.empty() && query.operand_dtypes.size() != 2) {
        return base::Status::unsupported("baseline copy requires two typed operands");
      }
      return std::vector<kernels::KernelCandidate>{{base::KernelId{1}, "sm120.baseline", true, 0}};
    }
    if (query.operation == "residual") {
      if (!query.operand_dtypes.empty() &&
          std::any_of(query.operand_dtypes.begin(), query.operand_dtypes.end(),
                      [](std::string_view dtype) { return dtype != "f32"; })) {
        return base::Status::unsupported("baseline residual requires f32 physical operands");
      }
      return std::vector<kernels::KernelCandidate>{{base::KernelId{4}, "sm120.baseline", true, 0}};
    }
    if (query.operation == "rms_norm") {
      if (query.storage_dtype == "bf16") {
        if (!query.operand_dtypes.empty() &&
            (query.operand_dtypes.size() != 3 || query.operand_dtypes[0] != "f32" ||
             query.operand_dtypes[1] != "bf16" || query.operand_dtypes[2] != "f32")) {
          return base::Status::unsupported("baseline BF16 RMSNorm requires f32 input/output");
        }
        return std::vector<kernels::KernelCandidate>{{base::KernelId{12}, "sm120.baseline", true, 0}};
      }
      if (query.storage_dtype != "f32") {
        return base::Status::unsupported("baseline RMSNorm does not support the requested scale dtype");
      }
      if (!query.operand_dtypes.empty() &&
          (query.operand_dtypes.size() != 3 ||
           std::any_of(query.operand_dtypes.begin(), query.operand_dtypes.end(),
                       [](std::string_view dtype) { return dtype != "f32"; }))) {
        return base::Status::unsupported("baseline RMSNorm requires f32 physical operands");
      }
      return std::vector<kernels::KernelCandidate>{{base::KernelId{5}, "sm120.baseline", true, 0}};
    }
    if (query.operation == "layer_norm") {
      if (!query.operand_dtypes.empty() &&
          (query.operand_dtypes.size() != 4 ||
           std::any_of(query.operand_dtypes.begin(), query.operand_dtypes.end(),
                       [](std::string_view dtype) { return dtype != "f32"; }))) {
        return base::Status::unsupported("baseline LayerNorm requires f32 physical operands");
      }
      return std::vector<kernels::KernelCandidate>{{base::KernelId{6}, "sm120.baseline", true, 0}};
    }
    if (query.operation == "embedding") {
      if (query.storage_dtype == "bf16") {
        if (!query.operand_dtypes.empty() &&
            (query.operand_dtypes.size() != 3 || query.operand_dtypes[0] != "int32" ||
             query.operand_dtypes[1] != "bf16" || query.operand_dtypes[2] != "f32")) {
          return base::Status::unsupported("baseline BF16 embedding requires int32 input and f32 output");
        }
        return std::vector<kernels::KernelCandidate>{{base::KernelId{8}, "sm120.baseline", true, 0}};
      }
      if (query.storage_dtype != "f32") {
        return base::Status::unsupported("baseline embedding does not support the requested storage dtype");
      }
      if (!query.operand_dtypes.empty() &&
          (query.operand_dtypes.size() != 3 || query.operand_dtypes[0] != "int32" ||
           query.operand_dtypes[1] != "f32" || query.operand_dtypes[2] != "f32")) {
        return base::Status::unsupported("baseline embedding requires int32 input and f32 table/output");
      }
      return std::vector<kernels::KernelCandidate>{{base::KernelId{7}, "sm120.baseline", true, 0}};
    }
    if (query.operation == "nvfp4_dequantize") {
      return std::vector<kernels::KernelCandidate>{{base::KernelId{9}, "sm120.baseline", true, 0}};
    }
    if (query.operation == "nvfp4_linear") {
      return std::vector<kernels::KernelCandidate>{{base::KernelId{13}, "sm120.baseline", true, 0}};
    }
    if (query.operation == "attention") {
      if (query.attention_output_gate) {
        return base::Status::unsupported("baseline attention does not implement an output gate");
      }
      if (query.operand_count != 0 && query.operand_count != 4) {
        return base::Status::unsupported("baseline attention requires four physical operands");
      }
      if (!query.operand_dtypes.empty() &&
          std::any_of(query.operand_dtypes.begin(), query.operand_dtypes.end(),
                      [](std::string_view dtype) { return dtype != "f32"; })) {
        return base::Status::unsupported("baseline attention requires f32 physical operands");
      }
      return std::vector<kernels::KernelCandidate>{{base::KernelId{14}, "sm120.baseline", true, 0}};
    }
    if (query.operation == "gated_delta_attention") {
      if (query.operand_count != 0 && query.operand_count != 7) {
        return base::Status::unsupported(
            "baseline gated delta attention requires seven physical operands");
      }
      if (!query.operand_dtypes.empty() &&
          std::any_of(query.operand_dtypes.begin(), query.operand_dtypes.end(),
                      [](std::string_view dtype) { return dtype != "f32"; })) {
        return base::Status::unsupported("baseline gated delta attention requires f32 physical operands");
      }
      return std::vector<kernels::KernelCandidate>{{base::KernelId{15}, "sm120.baseline", true, 0}};
    }
    if (query.operation == "lm_head") {
      if (query.storage_dtype != "f32") {
        return base::Status::unsupported("baseline LM head does not support the requested storage dtype");
      }
      if (!query.operand_dtypes.empty() &&
          (query.operand_dtypes.size() != 3 ||
           std::any_of(query.operand_dtypes.begin(), query.operand_dtypes.end(),
                       [](std::string_view dtype) { return dtype != "f32"; }))) {
        return base::Status::unsupported("baseline LM head requires f32 physical operands");
      }
      return std::vector<kernels::KernelCandidate>{{base::KernelId{10}, "sm120.baseline", true, 0}};
    }
    if (query.operation == "gated_dense_ffn") {
      if (query.storage_dtype != "f32") {
        return base::Status::unsupported("baseline gated FFN does not support the requested storage dtype");
      }
      if (!query.operand_dtypes.empty() &&
          (query.operand_dtypes.size() != 5 ||
           std::any_of(query.operand_dtypes.begin(), query.operand_dtypes.end(),
                       [](std::string_view dtype) { return dtype != "f32"; }))) {
        return base::Status::unsupported("baseline gated FFN requires f32 physical operands");
      }
      return std::vector<kernels::KernelCandidate>{{base::KernelId{11}, "sm120.baseline", true, 0}};
    }
    return base::Status::unsupported("no executable baseline candidate for operation");
  }
};

}  // namespace superinfer::sm120
