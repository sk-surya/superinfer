#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <superinfer/base/checked_math.hpp>
#include <superinfer/base/result.hpp>
#include <superinfer/ir/semantic/module.hpp>

namespace superinfer::sm120 {

struct ReferenceInput final {
  ir::semantic::TensorId id;
  std::vector<std::uint64_t> shape;
  std::vector<float> values;
};

struct ReferenceTensor final {
  std::string name;
  std::vector<std::uint64_t> shape;
  std::vector<float> values;
};

/** Independent CPU result table used as a differential oracle for device providers. */
struct ReferenceResult final {
  std::vector<ReferenceTensor> tensors;

  [[nodiscard]] const ReferenceTensor* find(std::string_view name) const noexcept {
    for (const ReferenceTensor& tensor : tensors) {
      if (tensor.name == name) return &tensor;
    }
    return nullptr;
  }
};

/**
 * Executes the small high-precision reference subset without calling provider or runtime code.
 *
 * Inputs and outputs are caller-owned value copies. The implementation is deterministic and
 * intentionally unspecialized; it is an oracle, never a candidate for the hot path.
 */
class ReferenceExecutor final {
 public:
  static base::Result<ReferenceResult> run(const ir::semantic::Module& module,
                                           const std::vector<ReferenceInput>& inputs) {
    base::Status module_status = module.verify();
    if (!module_status.ok()) return module_status.with_context("reference semantic module");
    ReferenceResult result;
    result.tensors.resize(module.tensors().size());
    std::vector<bool> initialized(module.tensors().size(), false);
    for (const ReferenceInput& input : inputs) {
      if (input.id.value() >= module.tensors().size()) {
        return base::Status::out_of_range("reference input tensor is undefined");
      }
      const auto expected = static_elements(module.tensors()[input.id.value()].spec.shape);
      if (!expected.has_value()) return expected.error();
      if (input.shape != expected_shape(module.tensors()[input.id.value()].spec.shape) ||
          input.values.size() != expected.value()) {
        return base::Status::invalid_argument("reference input shape or value count mismatches semantic tensor");
      }
      result.tensors[input.id.value()] = {module.tensors()[input.id.value()].name, input.shape,
                                          input.values};
      initialized[input.id.value()] = true;
    }

    for (const ir::semantic::Operation& operation : module.operations()) {
      if (operation.outputs.size() != 1 || operation.inputs.empty()) {
        return base::Status::invalid_argument("reference operation requires one output and inputs");
      }
      for (const ir::semantic::TensorId input : operation.inputs) {
        if (!initialized[input.value()]) return base::Status::failed_precondition("reference input is uninitialized");
      }
      const ir::semantic::TensorId output = operation.outputs.front();
      const auto shape = expected_shape(module.tensors()[output.value()].spec.shape);
      const auto elements = static_elements(module.tensors()[output.value()].spec.shape);
      if (!elements.has_value()) return elements.error();
      if (operation.kind == ir::semantic::OperationKind::residual) {
        if (operation.inputs.size() != 2 || result.tensors[operation.inputs[0].value()].shape !=
                                                result.tensors[operation.inputs[1].value()].shape) {
          return base::Status::invalid_argument("reference residual requires two equal-shaped inputs");
        }
        const auto& left = result.tensors[operation.inputs[0].value()].values;
        const auto& right = result.tensors[operation.inputs[1].value()].values;
        result.tensors[output.value()] = {module.tensors()[output.value()].name, shape,
                                          std::vector<float>(elements.value())};
        for (std::size_t index = 0; index < left.size(); ++index) {
          result.tensors[output.value()].values[index] = left[index] + right[index];
        }
      } else if (operation.kind == ir::semantic::OperationKind::rms_norm ||
                 operation.kind == ir::semantic::OperationKind::layer_norm) {
        if (operation.inputs.size() != 1) return base::Status::invalid_argument("reference norm requires one input");
        const auto& input = result.tensors[operation.inputs.front().value()].values;
        if (input.size() != elements.value()) return base::Status::invalid_argument("reference norm shape mismatch");
        float mean = 0.0F;
        for (const float value : input) mean += value;
        mean /= static_cast<float>(input.size());
        float variance = 0.0F;
        for (const float value : input) {
          const float centered = operation.kind == ir::semantic::OperationKind::layer_norm ? value - mean : value;
          variance += centered * centered;
        }
        variance /= static_cast<float>(input.size());
        const float denominator = std::sqrt(variance + 1.0e-5F);
        result.tensors[output.value()] = {module.tensors()[output.value()].name, shape,
                                          std::vector<float>(elements.value())};
        for (std::size_t index = 0; index < input.size(); ++index) {
          const float centered = operation.kind == ir::semantic::OperationKind::layer_norm ? input[index] - mean : input[index];
          result.tensors[output.value()].values[index] = centered / denominator;
        }
      } else {
        return base::Status::unsupported("reference operation is not implemented");
      }
      initialized[output.value()] = true;
    }
    return result;
  }

 private:
  static base::Result<std::uint64_t> static_elements(const ir::semantic::Shape& shape) {
    std::uint64_t elements = 1;
    for (const ir::semantic::Dimension& dimension : shape) {
      if (dimension.is_symbolic) return base::Status::unsupported("reference executor requires static shapes");
      const auto product = base::checked_mul(elements, dimension.value);
      if (!product.has_value()) return product.error();
      elements = product.value();
    }
    return elements;
  }

  static std::vector<std::uint64_t> expected_shape(const ir::semantic::Shape& shape) {
    std::vector<std::uint64_t> result;
    result.reserve(shape.size());
    for (const ir::semantic::Dimension& dimension : shape) result.push_back(dimension.value);
    return result;
  }
};

}  // namespace superinfer::sm120
