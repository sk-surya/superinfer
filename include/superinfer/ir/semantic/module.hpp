#pragma once

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <superinfer/base/ids.hpp>
#include <superinfer/base/result.hpp>

namespace superinfer::ir::semantic {

struct SemanticTensorIdTag;
struct SemanticOperationIdTag;
using TensorId = base::StrongId<SemanticTensorIdTag>;
using OperationId = base::StrongId<SemanticOperationIdTag>;

/** A positive static dimension or a named symbolic dimension. */
struct Dimension final {
  static Dimension static_value(std::uint64_t value) noexcept { return {false, value, {}}; }
  static Dimension symbolic(std::string name) { return {true, 0, std::move(name)}; }

  bool is_symbolic;
  std::uint64_t value;
  std::string symbol;
};

using Shape = std::vector<Dimension>;

enum class DType { f32, f16, bf16, int8, int4 };
enum class QuantizationIntent { none, symmetric, asymmetric };
enum class TensorRole { activation, weight, kv_cache, decode_state, logits };

/** Meaning-level tensor type; storage encoding is deliberately absent. */
struct TensorSpec final {
  Shape shape;
  DType dtype;
  QuantizationIntent quantization;
  TensorRole role;
};

enum class OperationKind {
  embedding,
  rms_norm,
  layer_norm,
  rope,
  qkv_projection,
  multi_head_attention,
  grouped_query_attention,
  local_attention,
  residual,
  gated_dense_ffn,
  moe_route,
  moe_top_k,
  moe_expert,
  moe_combine,
  lm_head,
  decode_logits,
  sampling_inputs,
};

/** Explicit semantic variants used by attention, RoPE, and MoE verifiers. */
struct OperationAttributes final {
  std::uint32_t num_heads{0};
  std::uint32_t num_kv_heads{0};
  std::uint32_t head_dimension{0};
  std::uint32_t rope_dimension{0};
  std::uint32_t expert_count{0};
  std::uint32_t top_k{0};
  std::uint32_t local_window{0};
};

struct Tensor final {
  TensorId id;
  std::string name;
  TensorSpec spec;
};

struct Operation final {
  OperationId id;
  std::string name;
  OperationKind kind;
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;
  OperationAttributes attributes;
};

struct EntryPoint final {
  std::string name;
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;
};

/** A state transition such as KV-cache read/write, explicit at semantic level. */
struct StateEdge final {
  std::string name;
  TensorId source;
  TensorId destination;
};

class Builder;

/**
 * Immutable model-meaning graph. Physical target and storage details terminate outside this type.
 */
class Module final {
 public:
  static Module empty() noexcept { return {}; }

  [[nodiscard]] std::uint32_t version() const noexcept { return 1; }
  [[nodiscard]] const std::vector<Tensor>& tensors() const noexcept { return tensors_; }
  [[nodiscard]] const std::vector<Operation>& operations() const noexcept { return operations_; }
  [[nodiscard]] const std::vector<EntryPoint>& entry_points() const noexcept { return entry_points_; }
  [[nodiscard]] const std::vector<StateEdge>& state_edges() const noexcept { return state_edges_; }

  /** Verifies definitions, uses, topology, shape contracts, and state references. */
  [[nodiscard]] base::Status verify() const {
    constexpr std::size_t kMaximumObjects = 1U << 20U;
    if (tensors_.size() > kMaximumObjects || operations_.size() > kMaximumObjects) {
      return base::Status::resource_exhausted("semantic graph object count exceeds limit");
    }
    for (std::size_t index = 0; index < tensors_.size(); ++index) {
      if (tensors_[index].id.value() != index || tensors_[index].name.empty()) {
        return base::Status::invalid_argument("semantic tensor has invalid identity or name");
      }
      for (std::size_t prior = 0; prior < index; ++prior) {
        if (tensors_[prior].name == tensors_[index].name) {
          return base::Status::invalid_argument("duplicate semantic tensor name: " + tensors_[index].name);
        }
      }
      for (const Dimension& dimension : tensors_[index].spec.shape) {
        if ((!dimension.is_symbolic && dimension.value == 0) ||
            (dimension.is_symbolic && dimension.symbol.empty())) {
          return base::Status::invalid_argument("semantic tensor has invalid dimension: " +
                                                tensors_[index].name);
        }
      }
    }

    std::vector<std::int64_t> producers(tensors_.size(), -1);
    for (std::size_t index = 0; index < operations_.size(); ++index) {
      const Operation& operation = operations_[index];
      if (operation.id.value() != index || operation.name.empty()) {
        return base::Status::invalid_argument("semantic operation has invalid identity or name");
      }
      for (std::size_t prior = 0; prior < index; ++prior) {
        if (operations_[prior].name == operation.name) {
          return base::Status::invalid_argument("duplicate semantic operation name: " + operation.name);
        }
      }
      for (TensorId input : operation.inputs) {
        if (input.value() >= tensors_.size()) {
          return base::Status::out_of_range("semantic operation input tensor is undefined");
        }
        if (producers[input.value()] >= 0 && producers[input.value()] >= static_cast<std::int64_t>(index)) {
          return base::Status::failed_precondition("semantic operation uses a later-produced tensor");
        }
      }
      for (TensorId output : operation.outputs) {
        if (output.value() >= tensors_.size()) {
          return base::Status::out_of_range("semantic operation output tensor is undefined");
        }
        if (producers[output.value()] >= 0) {
          return base::Status::failed_precondition("semantic tensor has multiple producers");
        }
        producers[output.value()] = static_cast<std::int64_t>(index);
      }
      const OperationAttributes& attributes = operation.attributes;
      const bool attention = operation.kind == OperationKind::multi_head_attention ||
                             operation.kind == OperationKind::grouped_query_attention ||
                             operation.kind == OperationKind::local_attention;
      if (attention && (attributes.num_heads == 0 || attributes.num_kv_heads == 0 ||
                        attributes.head_dimension == 0)) {
        return base::Status::invalid_argument("attention requires positive heads and head dimension");
      }
      if (attention && attributes.num_heads % attributes.num_kv_heads != 0) {
        return base::Status::invalid_argument("attention kv heads must divide query heads");
      }
      if (attention && attributes.rope_dimension > attributes.head_dimension) {
        return base::Status::invalid_argument("attention rope dimension exceeds head dimension");
      }
      if (attention && attributes.rope_dimension % 2 != 0) {
        return base::Status::invalid_argument("attention rope dimension must be even");
      }
      const bool moe = operation.kind == OperationKind::moe_route ||
                       operation.kind == OperationKind::moe_top_k ||
                       operation.kind == OperationKind::moe_expert ||
                       operation.kind == OperationKind::moe_combine;
      if (moe && (attributes.expert_count == 0 || attributes.top_k == 0 ||
                  attributes.top_k > attributes.expert_count)) {
        return base::Status::invalid_argument("MoE top-k must be within positive expert count");
      }
    }
    for (const EntryPoint& entry : entry_points_) {
      if (entry.name.empty() || entry.inputs.empty() || entry.outputs.empty()) {
        return base::Status::invalid_argument("entry point requires name, inputs, and outputs");
      }
      for (TensorId id : entry.inputs) {
        if (id.value() >= tensors_.size()) {
          return base::Status::out_of_range("entry point input tensor is undefined");
        }
      }
      for (TensorId id : entry.outputs) {
        if (id.value() >= tensors_.size()) {
          return base::Status::out_of_range("entry point output tensor is undefined");
        }
      }
    }
    for (const StateEdge& edge : state_edges_) {
      if (edge.name.empty() || edge.source.value() >= tensors_.size() ||
          edge.destination.value() >= tensors_.size()) {
        return base::Status::invalid_argument("state edge has invalid name or tensor reference");
      }
    }
    return {};
  }

  /** Returns canonical bytes independent of insertion order, addresses, and source paths. */
  [[nodiscard]] std::string dump() const {
    std::ostringstream output;
    output << "semantic-ir:v1\n";
    std::vector<const Tensor*> tensors;
    for (const Tensor& tensor : tensors_) tensors.push_back(&tensor);
    std::sort(tensors.begin(), tensors.end(), [](const Tensor* left, const Tensor* right) {
      return left->name < right->name;
    });
    for (const Tensor* tensor : tensors) {
      output << "tensor name=" << tensor->name << " role=" << role_name(tensor->spec.role)
             << " dtype=" << dtype_name(tensor->spec.dtype)
             << " quant=" << quantization_name(tensor->spec.quantization) << " shape="
             << shape_name(tensor->spec.shape) << "\n";
    }
    std::vector<const Operation*> operations;
    for (const Operation& operation : operations_) operations.push_back(&operation);
    std::sort(operations.begin(), operations.end(), [](const Operation* left, const Operation* right) {
      return left->name < right->name;
    });
    for (const Operation* operation : operations) {
      output << "operation name=" << operation->name << " kind=" << operation_kind_name(operation->kind)
             << " inputs=" << tensor_names(operation->inputs) << " outputs="
             << tensor_names(operation->outputs) << "\n";
    }
    std::vector<const EntryPoint*> entries;
    for (const EntryPoint& entry : entry_points_) entries.push_back(&entry);
    std::sort(entries.begin(), entries.end(), [](const EntryPoint* left, const EntryPoint* right) {
      return left->name < right->name;
    });
    for (const EntryPoint* entry : entries) {
      output << "entry name=" << entry->name << " inputs=" << tensor_names(entry->inputs)
             << " outputs=" << tensor_names(entry->outputs) << "\n";
    }
    return output.str();
  }

 private:
  friend class Builder;
  Module(std::vector<Tensor> tensors, std::vector<Operation> operations,
         std::vector<EntryPoint> entry_points, std::vector<StateEdge> state_edges)
      : tensors_(std::move(tensors)),
        operations_(std::move(operations)),
        entry_points_(std::move(entry_points)),
        state_edges_(std::move(state_edges)) {}
  Module() = default;

  [[nodiscard]] std::string tensor_name(TensorId id) const {
    if (id.value() >= tensors_.size()) return "<undefined>";
    return tensors_[id.value()].name;
  }
  [[nodiscard]] std::string tensor_names(const std::vector<TensorId>& ids) const {
    std::ostringstream output;
    output << "[";
    for (std::size_t index = 0; index < ids.size(); ++index) {
      if (index != 0) output << ",";
      output << tensor_name(ids[index]);
    }
    output << "]";
    return output.str();
  }
  static std::string shape_name(const Shape& shape) {
    std::ostringstream output;
    output << "[";
    for (std::size_t index = 0; index < shape.size(); ++index) {
      if (index != 0) output << ",";
      output << (shape[index].is_symbolic ? shape[index].symbol : std::to_string(shape[index].value));
    }
    output << "]";
    return output.str();
  }
  static std::string_view dtype_name(DType dtype) {
    switch (dtype) {
      case DType::f32: return "f32";
      case DType::f16: return "f16";
      case DType::bf16: return "bf16";
      case DType::int8: return "int8";
      case DType::int4: return "int4";
    }
    return "unknown";
  }
  static std::string_view quantization_name(QuantizationIntent quantization) {
    switch (quantization) {
      case QuantizationIntent::none: return "none";
      case QuantizationIntent::symmetric: return "symmetric";
      case QuantizationIntent::asymmetric: return "asymmetric";
    }
    return "unknown";
  }
  static std::string_view role_name(TensorRole role) {
    switch (role) {
      case TensorRole::activation: return "activation";
      case TensorRole::weight: return "weight";
      case TensorRole::kv_cache: return "kv_cache";
      case TensorRole::decode_state: return "decode_state";
      case TensorRole::logits: return "logits";
    }
    return "unknown";
  }
  static std::string_view operation_kind_name(OperationKind kind) {
    switch (kind) {
      case OperationKind::embedding: return "embedding";
      case OperationKind::rms_norm: return "rms_norm";
      case OperationKind::layer_norm: return "layer_norm";
      case OperationKind::rope: return "rope";
      case OperationKind::qkv_projection: return "qkv_projection";
      case OperationKind::multi_head_attention: return "multi_head_attention";
      case OperationKind::grouped_query_attention: return "grouped_query_attention";
      case OperationKind::local_attention: return "local_attention";
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
    return "unknown";
  }

  std::vector<Tensor> tensors_;
  std::vector<Operation> operations_;
  std::vector<EntryPoint> entry_points_;
  std::vector<StateEdge> state_edges_;
};

}  // namespace superinfer::ir::semantic
