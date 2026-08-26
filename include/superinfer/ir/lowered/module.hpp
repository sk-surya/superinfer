#pragma once

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <superinfer/base/memory_space.hpp>
#include <superinfer/base/result.hpp>
#include <superinfer/ir/semantic/module.hpp>

namespace superinfer::ir::lowered {

struct LoweredTensorIdTag;
using LoweredTensorId = base::StrongId<LoweredTensorIdTag>;

enum class LayoutKind { row_major, column_major, blocked };

/** A target-aware tensor descriptor with no device address or executable handle. */
struct Tensor final {
  LoweredTensorId id;
  semantic::TensorId semantic_origin;
  semantic::TensorRole role;
  std::vector<std::uint64_t> physical_shape;
  LayoutKind layout;
  base::MemorySpace memory_space;
  std::uint64_t alignment;
  semantic::DType storage_dtype;
  semantic::DType accumulation_dtype;
};

struct FusedRegion final {
  std::string name;
  std::vector<LoweredTensorId> tensors;
};

struct KernelRequirement final {
  std::string operation;
  std::uint32_t target_capability;
  std::vector<LoweredTensorId> operands;
  semantic::OperationAttributes attributes;
};

/** Immutable target-aware module used as input to physical planning. */
class Module final {
 public:
  static Module empty() noexcept { return {}; }
  [[nodiscard]] std::uint32_t version() const noexcept { return 1; }
  [[nodiscard]] const std::vector<Tensor>& tensors() const noexcept { return tensors_; }
  [[nodiscard]] const std::vector<FusedRegion>& fused_regions() const noexcept { return fused_regions_; }
  [[nodiscard]] const std::vector<KernelRequirement>& kernel_requirements() const noexcept {
    return kernel_requirements_;
  }

  [[nodiscard]] base::Status verify() const {
    for (std::size_t index = 0; index < tensors_.size(); ++index) {
      const Tensor& tensor = tensors_[index];
      if (tensor.id.value() != index || tensor.semantic_origin.value() == UINT64_MAX ||
          tensor.physical_shape.empty() || tensor.alignment == 0) {
        return base::Status::invalid_argument("lowered tensor has invalid identity, shape, or alignment");
      }
      for (std::uint64_t dimension : tensor.physical_shape) {
        if (dimension == 0) return base::Status::invalid_argument("lowered tensor has zero dimension");
      }
    }
    for (const FusedRegion& region : fused_regions_) {
      if (region.name.empty() || region.tensors.empty()) {
        return base::Status::invalid_argument("fused region requires name and tensors");
      }
      for (LoweredTensorId id : region.tensors) {
        if (id.value() >= tensors_.size()) {
          return base::Status::out_of_range("fused region tensor is undefined");
        }
      }
    }
    for (const KernelRequirement& requirement : kernel_requirements_) {
      if (requirement.operation.empty() || requirement.target_capability == 0) {
        return base::Status::invalid_argument("kernel requirement is incomplete");
      }
      for (const LoweredTensorId operand : requirement.operands) {
        if (operand.value() >= tensors_.size()) {
          return base::Status::out_of_range("kernel requirement operand is undefined");
        }
      }
    }
    return {};
  }

  [[nodiscard]] std::string dump() const {
    std::ostringstream output;
    output << "lowered-ir:v1\n";
    std::vector<const Tensor*> tensors;
    for (const Tensor& tensor : tensors_) tensors.push_back(&tensor);
    std::sort(tensors.begin(), tensors.end(), [](const Tensor* left, const Tensor* right) {
      return left->id.value() < right->id.value();
    });
    for (const Tensor* tensor : tensors) {
      output << "tensor id=" << tensor->id.value() << " layout=" << layout_name(tensor->layout)
             << " memory=" << base::memory_space_name(tensor->memory_space)
             << " alignment=" << tensor->alignment << " shape=";
      output << "[";
      for (std::size_t index = 0; index < tensor->physical_shape.size(); ++index) {
        if (index != 0) output << ",";
        output << tensor->physical_shape[index];
      }
      output << "]\n";
    }
    return output.str();
  }

 private:
  friend class ModuleBuilder;
  Module(std::vector<Tensor> tensors, std::vector<FusedRegion> fused_regions,
         std::vector<KernelRequirement> requirements)
      : tensors_(std::move(tensors)),
        fused_regions_(std::move(fused_regions)),
        kernel_requirements_(std::move(requirements)) {}
  Module() = default;

  static std::string_view layout_name(LayoutKind layout) {
    switch (layout) {
      case LayoutKind::row_major: return "row_major";
      case LayoutKind::column_major: return "column_major";
      case LayoutKind::blocked: return "blocked";
    }
    return "unknown";
  }

  std::vector<Tensor> tensors_;
  std::vector<FusedRegion> fused_regions_;
  std::vector<KernelRequirement> kernel_requirements_;
};

/** Checked builder for target/layout/fusion descriptors. */
class ModuleBuilder final {
 public:
  base::Result<LoweredTensorId> add_tensor(semantic::TensorId origin,
                                           std::vector<std::uint64_t> shape, LayoutKind layout,
                                           base::MemorySpace memory_space, std::uint64_t alignment,
                                           semantic::DType storage_dtype,
                                           semantic::DType accumulation_dtype,
                                           semantic::TensorRole role = semantic::TensorRole::activation) {
    if (shape.empty() || alignment == 0 || origin.value() == UINT64_MAX) {
      return base::Status::invalid_argument("lowered tensor descriptor is incomplete");
    }
    tensors_.push_back({LoweredTensorId{tensors_.size()}, origin, role, std::move(shape), layout,
                        memory_space, alignment, storage_dtype, accumulation_dtype});
    return tensors_.back().id;
  }

  base::Status add_fusion(std::string name, std::vector<LoweredTensorId> tensors) {
    fused_regions_.push_back({std::move(name), std::move(tensors)});
    return {};
  }

  base::Status add_kernel_requirement(std::string operation, std::uint32_t target_capability,
                                      std::vector<LoweredTensorId> operands = {},
                                      semantic::OperationAttributes attributes = {}) {
    kernel_requirements_.push_back(
        {std::move(operation), target_capability, std::move(operands), attributes});
    return {};
  }

  [[nodiscard]] base::Result<Module> build() && {
    Module module{std::move(tensors_), std::move(fused_regions_), std::move(kernel_requirements_)};
    base::Status status = module.verify();
    if (!status.ok()) return status.with_context("lowered-ir builder");
    return base::Result<Module>(std::move(module));
  }

 private:
  std::vector<Tensor> tensors_;
  std::vector<FusedRegion> fused_regions_;
  std::vector<KernelRequirement> kernel_requirements_;
};

}  // namespace superinfer::ir::lowered
