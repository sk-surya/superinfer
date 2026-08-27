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
    if (query.operation == "cast") {
      if (query.operand_dtypes.size() != 2) {
        return base::Status::unsupported("baseline cast requires source and destination dtypes");
      }
      if (query.operand_dtypes[0] == "bf16" && query.operand_dtypes[1] == "f32") {
        return std::vector<kernels::KernelCandidate>{{base::KernelId{16}, "sm120.baseline", true, 0}};
      }
      if (query.operand_dtypes[0] == "f32" && query.operand_dtypes[1] == "bf16") {
        return std::vector<kernels::KernelCandidate>{{base::KernelId{17}, "sm120.baseline", true, 0}};
      }
      return base::Status::unsupported("baseline cast only supports BF16/F32 conversion");
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
      if (query.operand_count != 0 && query.operand_count != 5) {
        return base::Status::unsupported(
            "baseline NVFP4 linear requires five typed operands");
      }
      if (!query.operand_dtypes.empty() &&
          (query.operand_dtypes.size() != 5 || query.operand_dtypes[0] != "f32" ||
           query.operand_dtypes[1] != "int4" || query.operand_dtypes[2] != "int8" ||
           query.operand_dtypes[3] != "f32" || query.operand_dtypes[4] != "f32")) {
        return base::Status::unsupported(
            "baseline NVFP4 linear requires f32, packed int4, int8 scale, f32 tensor scale, and f32 output operands");
      }
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
    if (query.operation == "silu_mul") {
      if (query.operand_count != 0 && query.operand_count != 3) {
        return base::Status::unsupported("baseline SiLU multiply requires three typed operands");
      }
      if (!query.operand_dtypes.empty() &&
          (query.operand_dtypes.size() != 3 ||
           std::any_of(query.operand_dtypes.begin(), query.operand_dtypes.end(),
                       [](std::string_view dtype) { return dtype != "f32"; }))) {
        return base::Status::unsupported("baseline SiLU multiply requires f32 operands");
      }
      return std::vector<kernels::KernelCandidate>{{base::KernelId{18}, "sm120.baseline", true, 0}};
    }
    if (query.operation == "sigmoid_mul") {
      if (query.operand_count != 0 && query.operand_count != 3) {
        return base::Status::unsupported("baseline sigmoid multiply requires three typed operands");
      }
      if (!query.operand_dtypes.empty() &&
          (query.operand_dtypes.size() != 3 ||
           std::any_of(query.operand_dtypes.begin(), query.operand_dtypes.end(),
                       [](std::string_view dtype) { return dtype != "f32"; }))) {
        return base::Status::unsupported("baseline sigmoid multiply requires f32 operands");
      }
      return std::vector<kernels::KernelCandidate>{{base::KernelId{19}, "sm120.baseline", true, 0}};
    }
    if (query.operation == "rope") {
      if (query.operand_count != 0 && query.operand_count != 2) {
        return base::Status::unsupported("baseline RoPE requires input and output operands");
      }
      if (!query.operand_dtypes.empty() &&
          (query.operand_dtypes.size() != 2 || query.operand_dtypes[0] != "f32" ||
           query.operand_dtypes[1] != "f32")) {
        return base::Status::unsupported("baseline RoPE requires f32 operands");
      }
      return std::vector<kernels::KernelCandidate>{{base::KernelId{20}, "sm120.baseline", true, 0}};
    }
    if (query.operation == "split") {
      if (query.operand_count != 0 && query.operand_count != 3) {
        return base::Status::unsupported("baseline split requires input and two output operands");
      }
      if (!query.operand_dtypes.empty() &&
          (query.operand_dtypes.size() != 3 ||
           std::any_of(query.operand_dtypes.begin(), query.operand_dtypes.end(),
                       [](std::string_view dtype) { return dtype != "f32"; }))) {
        return base::Status::unsupported("baseline split requires f32 operands");
      }
      return std::vector<kernels::KernelCandidate>{{base::KernelId{21}, "sm120.baseline", true, 0}};
    }
    if (query.operation == "cache_append") {
      if (query.operand_count != 0 && query.operand_count != 4) {
        return base::Status::unsupported("baseline cache append requires four typed operands");
      }
      if (!query.operand_dtypes.empty() &&
          (query.operand_dtypes.size() != 4 || query.operand_dtypes[0] != "f32" ||
           query.operand_dtypes[1] != "f32" || query.operand_dtypes[2] != "bf16" ||
           query.operand_dtypes[3] != "bf16")) {
        return base::Status::unsupported("baseline cache append requires f32 inputs and BF16 states");
      }
      return std::vector<kernels::KernelCandidate>{{base::KernelId{22}, "sm120.baseline", true, 0}};
    }
    if (query.operation == "attention_bf16_cache") {
      if (query.operand_count != 0 && query.operand_count != 4) {
        return base::Status::unsupported("baseline BF16-cache attention requires four typed operands");
      }
      if (!query.operand_dtypes.empty() &&
          (query.operand_dtypes.size() != 4 || query.operand_dtypes[0] != "f32" ||
           query.operand_dtypes[1] != "bf16" || query.operand_dtypes[2] != "bf16" ||
           query.operand_dtypes[3] != "f32")) {
        return base::Status::unsupported("baseline BF16-cache attention requires f32 query/output and BF16 cache");
      }
      return std::vector<kernels::KernelCandidate>{{base::KernelId{23}, "sm120.baseline", true, 0}};
    }
    return base::Status::unsupported("no executable baseline candidate for operation");
  }
};

}  // namespace superinfer::sm120
