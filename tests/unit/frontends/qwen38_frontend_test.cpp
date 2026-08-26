#include <superinfer/compiler/model_frontend.hpp>
#include <superinfer/compiler/semantic_lowering.hpp>
#include <frontends/qwen38/frontend.hpp>

#include <cassert>

int main() {
  using namespace superinfer;
  frontends::qwen38::Frontend frontend;
  compiler::SourceInventory source{std::string{frontends::qwen38::kSourceIdentity}};
  assert(frontend.validate(source).ok());
  const auto module = frontend.emit(source);
  assert(module.has_value());
  assert(module.value().verify().ok());
  assert(module.value().entry_points().size() == 1);
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

  const auto lowered = compiler::SemanticLowering{}.lower(module.value(), {120, 16});
  assert(lowered.has_value());
  assert(lowered.value().verify().ok());
  assert(lowered.value().tensors().size() == module.value().tensors().size());
  assert(lowered.value().kernel_requirements().size() == module.value().operations().size());
  assert(lowered.value().kernel_requirements()[1].operation == "rms_norm");
  assert(lowered.value().kernel_requirements()[2].operation == "gated_delta_attention");

  const auto rejected = frontend.validate({"Qwen/Qwen3.8-27B@wrong"});
  assert(!rejected.ok());
  assert(rejected.code() == base::StatusCode::failed_precondition);
  return 0;
}
