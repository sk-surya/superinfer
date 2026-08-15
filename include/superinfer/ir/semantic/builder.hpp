#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <superinfer/ir/semantic/module.hpp>

namespace superinfer::ir::semantic {

/** Transactional Semantic IR builder; failed builds never expose a partial Module. */
class Builder final {
 public:
  base::Result<TensorId> add_tensor(std::string name, TensorSpec spec) {
    if (name.empty()) return base::Status::invalid_argument("semantic tensor name is empty");
    for (const Tensor& tensor : tensors_) {
      if (tensor.name == name) return base::Status::invalid_argument("duplicate semantic tensor name: " + name);
    }
    if (tensors_.size() >= (1U << 20U)) {
      return base::Status::resource_exhausted("semantic tensor count exceeds limit");
    }
    tensors_.push_back({TensorId{tensors_.size()}, std::move(name), std::move(spec)});
    return tensors_.back().id;
  }

  base::Result<OperationId> add_operation(std::string name, OperationKind kind,
                                          std::vector<TensorId> inputs,
                                          std::vector<TensorId> outputs,
                                          OperationAttributes attributes = {}) {
    if (name.empty()) return base::Status::invalid_argument("semantic operation name is empty");
    for (const Operation& operation : operations_) {
      if (operation.name == name) {
        return base::Status::invalid_argument("duplicate semantic operation name: " + name);
      }
    }
    if (operations_.size() >= (1U << 20U)) {
      return base::Status::resource_exhausted("semantic operation count exceeds limit");
    }
    operations_.push_back({OperationId{operations_.size()}, std::move(name), kind, std::move(inputs),
                            std::move(outputs), attributes});
    return operations_.back().id;
  }

  base::Status add_entry_point(std::string name, std::vector<TensorId> inputs,
                               std::vector<TensorId> outputs) {
    if (name.empty()) return base::Status::invalid_argument("entry point name is empty");
    for (const EntryPoint& entry : entry_points_) {
      if (entry.name == name) return base::Status::invalid_argument("duplicate entry point: " + name);
    }
    entry_points_.push_back({std::move(name), std::move(inputs), std::move(outputs)});
    return {};
  }

  base::Status add_state_edge(std::string name, TensorId source, TensorId destination) {
    if (name.empty()) return base::Status::invalid_argument("state edge name is empty");
    state_edges_.push_back({std::move(name), source, destination});
    return {};
  }

  [[nodiscard]] base::Result<Module> build() && {
    Module module{std::move(tensors_), std::move(operations_), std::move(entry_points_),
                  std::move(state_edges_)};
    base::Status status = module.verify();
    if (!status.ok()) return status.with_context("semantic-ir builder");
    return base::Result<Module>(std::move(module));
  }

 private:
  std::vector<Tensor> tensors_;
  std::vector<Operation> operations_;
  std::vector<EntryPoint> entry_points_;
  std::vector<StateEdge> state_edges_;
};

}  // namespace superinfer::ir::semantic
