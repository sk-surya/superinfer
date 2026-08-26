#include <superinfer/compiler/model_frontend.hpp>
#include <superinfer/compiler/semantic_lowering.hpp>
#include <frontends/qwen38/frontend.hpp>
#include <sm120/compiler/specializer.h>

#include <cassert>

int main() {
  using namespace superinfer;
  frontends::qwen38::Frontend frontend;
  compiler::SourceInventory source{std::string{frontends::qwen38::kSourceIdentity},
                                   frontends::qwen38::kTensorCount,
                                   std::string{frontends::qwen38::kTensorInventorySha256}};
  assert(frontend.validate(source).ok());
  const auto module = frontend.emit(source);
  assert(module.has_value());
  assert(module.value().verify().ok());
  assert(module.value().entry_points().size() == 1);
  assert(module.value().state_edges().size() == 128);
  assert(module.value().dump().find("cuda") == std::string::npos);
  assert(module.value().dump().find("Qwen") == std::string::npos);
  assert(module.value().dump().find("gated_delta_attention") != std::string::npos);
  assert(module.value().dump().find("tensor name=token_ids role=activation dtype=int32") !=
         std::string::npos);

  std::size_t gated_delta = 0;
  std::size_t full_attention = 0;
  for (const auto& operation : module.value().operations()) {
    gated_delta += operation.kind == ir::semantic::OperationKind::gated_delta_attention;
    full_attention += operation.kind == ir::semantic::OperationKind::grouped_query_attention;
  }
  assert(gated_delta == 48);
  assert(full_attention == 16);
  for (const auto& operation : module.value().operations()) {
    if (operation.kind == ir::semantic::OperationKind::gated_delta_attention ||
        operation.kind == ir::semantic::OperationKind::grouped_query_attention) {
      assert(operation.inputs.size() == 3);
      assert(operation.outputs.size() == 3);
    }
    if (operation.kind == ir::semantic::OperationKind::gated_delta_attention) {
      assert(operation.attributes.num_heads == 16);
      assert(operation.attributes.num_kv_heads == 16);
      assert(operation.attributes.value_head_count == 48);
    }
  }

  const auto lowered = compiler::SemanticLowering{}.lower(module.value(), {120, 16});
  assert(lowered.has_value());
  assert(lowered.value().verify().ok());
  assert(lowered.value().tensors().size() == module.value().tensors().size());
  assert(lowered.value().kernel_requirements().size() == module.value().operations().size());
  assert(lowered.value().kernel_requirements()[1].operation == "rms_norm");
  assert(lowered.value().kernel_requirements()[2].operation == "gated_delta_attention");
  std::size_t lowered_kv = 0;
  std::size_t lowered_decode_state = 0;
  for (const auto& tensor : lowered.value().tensors()) {
    lowered_kv += tensor.role == ir::semantic::TensorRole::kv_cache;
    lowered_decode_state += tensor.role == ir::semantic::TensorRole::decode_state;
  }
  assert(lowered_kv == 160);
  assert(lowered_decode_state == 96);
  bool saw_f32_delta_state = false;
  for (const auto& tensor : module.value().tensors()) {
    if (tensor.name == "layer_00_delta_state_in") {
      saw_f32_delta_state = tensor.spec.dtype == ir::semantic::DType::f32;
    }
  }
  assert(saw_f32_delta_state);
  const auto physical = sm120::Specializer{}.compile(
      lowered.value(),
      {compiler::TargetProfile::offline_sm120a(32ULL << 30U, "baseline-v1"), 0, 512});
  assert(!physical.has_value());
  assert(physical.error().code() == base::StatusCode::unsupported);
  assert(!physical.error().context().empty());
  assert(physical.error().context().back() == "embedding");
  assert(physical.error().message().find("Qwen") == std::string::npos);

  const auto rejected_inventory = frontend.validate({std::string{frontends::qwen38::kSourceIdentity}, 1,
                                                     std::string{frontends::qwen38::kTensorInventorySha256}});
  assert(!rejected_inventory.ok());
  const auto rejected = frontend.validate({"Qwen/Qwen3.8-27B@wrong", 0, {}});
  assert(!rejected.ok());
  assert(rejected.code() == base::StatusCode::failed_precondition);
  return 0;
}
