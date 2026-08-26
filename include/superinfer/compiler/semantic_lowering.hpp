#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include <superinfer/base/memory_space.hpp>
#include <superinfer/base/result.hpp>
#include <superinfer/ir/lowered/module.hpp>
#include <superinfer/ir/semantic/module.hpp>

namespace superinfer::compiler {

struct SemanticLoweringOptions final {
  std::uint32_t target_capability{0};
  std::uint64_t required_alignment{1};
};

/**
 * Generic shape/origin lowering for canonical semantic graphs.
 *
 * This pass does not inspect model identity or select a physical kernel. It carries each static
 * semantic tensor into Lowered IR and records operation capability requirements with explicit
 * lowered tensor operands. Target-specific layout and provider selection remain later boundaries.
 */
class SemanticLowering final {
 public:
  [[nodiscard]] base::Result<ir::lowered::Module> lower(
      const ir::semantic::Module& semantic, SemanticLoweringOptions options) const {
    base::Status semantic_status = semantic.verify();
    if (!semantic_status.ok()) return semantic_status.with_context("semantic lowering input");
    if (options.target_capability == 0 || options.required_alignment == 0) {
      return base::Status::invalid_argument("semantic lowering target/alignment is incomplete");
    }

    ir::lowered::ModuleBuilder builder;
    std::vector<ir::lowered::LoweredTensorId> lowered_tensors;
    lowered_tensors.reserve(semantic.tensors().size());
    for (const ir::semantic::Tensor& tensor : semantic.tensors()) {
      std::vector<std::uint64_t> shape;
      shape.reserve(tensor.spec.shape.size());
      for (const ir::semantic::Dimension& dimension : tensor.spec.shape) {
        if (dimension.is_symbolic) {
          return base::Status::unsupported("semantic lowering requires static tensor dimensions");
        }
        shape.push_back(dimension.value);
      }
      const auto lowered = builder.add_tensor(
          tensor.id, std::move(shape), ir::lowered::LayoutKind::row_major,
          base::MemorySpace::device, options.required_alignment, tensor.spec.dtype,
          ir::semantic::DType::f32);
      if (!lowered.has_value()) {
        base::Status error = lowered.error();
        return error.with_context("semantic tensor lowering");
      }
      lowered_tensors.push_back(lowered.value());
    }

    for (const ir::semantic::Operation& operation : semantic.operations()) {
      const std::string_view capability = capability_name(operation.kind);
      if (capability.empty()) {
        return base::Status::unsupported("semantic operation has no generic lowering capability");
      }
      std::vector<ir::lowered::LoweredTensorId> operands;
      operands.reserve(operation.inputs.size() + operation.outputs.size());
      for (const ir::semantic::TensorId input : operation.inputs) operands.push_back(lowered_tensors[input.value()]);
      for (const ir::semantic::TensorId output : operation.outputs) operands.push_back(lowered_tensors[output.value()]);
      base::Status requirement = builder.add_kernel_requirement(
          std::string{capability}, options.target_capability, std::move(operands));
      if (!requirement.ok()) return requirement.with_context("semantic operation lowering");
    }
    return std::move(builder).build();
  }

 private:
  static std::string_view capability_name(ir::semantic::OperationKind kind) noexcept {
    using ir::semantic::OperationKind;
    switch (kind) {
      case OperationKind::embedding: return "embedding";
      case OperationKind::rms_norm: return "rms_norm";
      case OperationKind::layer_norm: return "layer_norm";
      case OperationKind::rope: return "rope";
      case OperationKind::qkv_projection: return "qkv_projection";
      case OperationKind::gated_delta_attention: return "gated_delta_attention";
      case OperationKind::multi_head_attention: return "attention";
      case OperationKind::grouped_query_attention: return "attention";
      case OperationKind::local_attention: return "attention";
      case OperationKind::residual: return "residual";
      case OperationKind::gated_dense_ffn: return "gated_dense_ffn";
      case OperationKind::moe_route: return "moe_route";
      case OperationKind::moe_top_k: return "moe_top_k";
      case OperationKind::moe_expert: return "moe_expert";
      case OperationKind::moe_combine: return "moe_combine";
      case OperationKind::lm_head: return "lm_head";
      case OperationKind::decode_logits: return "decode_logits";
      case OperationKind::sampling_inputs: return "sampling_inputs";
    }
    return {};
  }
};

}  // namespace superinfer::compiler
