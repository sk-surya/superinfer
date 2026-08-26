#include <superinfer/compiler/model_frontend.hpp>
#include <superinfer/compiler/semantic_lowering.hpp>
#include <frontends/qwen38/frontend.hpp>
#include <sm120/compiler/specializer.h>
#include <sm120/kernels/baseline/provider.h>

#include <cassert>
#include <vector>

namespace {

std::vector<superinfer::compiler::SourceTensorRecord> source_tensors() {
  using superinfer::compiler::SourceTensorRecord;
  std::vector<SourceTensorRecord> tensors;
  const auto add = [&tensors](std::string name, std::string role, std::string dtype = "BF16") {
    const std::uint64_t offset = tensors.size() * 2;
    tensors.push_back({std::move(name), std::move(role), std::move(dtype), {1}, offset, 2});
  };
  add("model.language_model.embed_tokens.weight", "embedding");
  add("lm_head.weight", "lm_head", "U8");
  add("lm_head.weight_scale", "scale", "F8_E4M3");
  for (std::uint32_t layer = 0; layer < 64; ++layer) {
    const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
    add(prefix + "input_layernorm.weight", "normalization");
    add(prefix + "post_attention_layernorm.weight", "normalization");
    add(prefix + "mlp.gate_proj.weight", "feed_forward", "U8");
    add(prefix + "mlp.up_proj.weight", "feed_forward", "U8");
    add(prefix + "mlp.down_proj.weight", "feed_forward", "U8");
    if (layer % 4U == 3U) {
      add(prefix + "self_attn.q_proj.weight", "attention", "U8");
      add(prefix + "self_attn.k_proj.weight", "attention", "U8");
      add(prefix + "self_attn.v_proj.weight", "attention", "U8");
      add(prefix + "self_attn.o_proj.weight", "attention", "U8");
      add(prefix + "self_attn.q_norm.weight", "attention");
      add(prefix + "self_attn.k_norm.weight", "attention");
    } else {
      add(prefix + "linear_attn.in_proj_qkv.weight", "attention", "U8");
      add(prefix + "linear_attn.in_proj_z.weight", "attention", "U8");
      add(prefix + "linear_attn.in_proj_a.weight", "attention", "BF16");
      add(prefix + "linear_attn.in_proj_b.weight", "attention", "BF16");
      add(prefix + "linear_attn.out_proj.weight", "attention", "U8");
      add(prefix + "linear_attn.A_log", "attention", "F32");
      add(prefix + "linear_attn.dt_bias", "bias", "F32");
      add(prefix + "linear_attn.norm.weight", "attention");
      add(prefix + "linear_attn.conv1d.weight", "attention");
    }
  }
  while (tensors.size() < superinfer::frontends::qwen38::kTensorCount) {
    const std::uint64_t offset = tensors.size() * 2;
    tensors.push_back({"filler_" + std::to_string(tensors.size()), "metadata", "BF16", {1}, offset, 2});
  }
  return tensors;
}

}  // namespace

int main() {
  using namespace superinfer;
  frontends::qwen38::Frontend frontend;
  compiler::SourceInventory source{std::string{frontends::qwen38::kSourceIdentity},
                                   frontends::qwen38::kTensorCount,
                                   std::string{frontends::qwen38::kTensorInventorySha256}, source_tensors()};
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
  assert(module.value().dump().find(
             "tensor name=weight/model.language_model.embed_tokens.weight role=weight") !=
         std::string::npos);
  assert(module.value().dump().find("weight/lm_head.weight_scale") == std::string::npos);

  for (const auto& operation : module.value().operations()) {
    if (operation.name == "embedding") {
      assert(operation.inputs.size() == 2);
      assert(module.value().tensors()[operation.inputs[1].value()].name ==
             "weight/model.language_model.embed_tokens.weight");
    }
    if (operation.name == "lm_head") {
      assert(operation.inputs.size() == 2);
      assert(module.value().tensors()[operation.inputs[1].value()].name == "weight/lm_head.weight");
    }
  }

  std::size_t gated_delta = 0;
  std::size_t full_attention = 0;
  std::size_t qwen_rms_norms = 0;
  for (const auto& operation : module.value().operations()) {
    gated_delta += operation.kind == ir::semantic::OperationKind::gated_delta_attention;
    full_attention += operation.kind == ir::semantic::OperationKind::grouped_query_attention;
    if (operation.kind == ir::semantic::OperationKind::rms_norm) {
      ++qwen_rms_norms;
      assert(operation.attributes.epsilon == 1.0e-6F);
      assert(operation.attributes.norm_scale_convention ==
             ir::semantic::NormScaleConvention::one_plus_weight);
    }
  }
  assert(gated_delta == 48);
  assert(full_attention == 16);
  assert(qwen_rms_norms == 128);
  for (const auto& operation : module.value().operations()) {
    if (operation.kind == ir::semantic::OperationKind::gated_delta_attention ||
        operation.kind == ir::semantic::OperationKind::grouped_query_attention) {
      assert(operation.inputs.size() >= 9);
      assert(operation.outputs.size() == 3);
    }
    if (operation.kind == ir::semantic::OperationKind::gated_delta_attention) {
      assert(operation.attributes.num_heads == 16);
      assert(operation.attributes.num_kv_heads == 16);
      assert(operation.attributes.value_head_count == 48);
    }
    if (operation.kind == ir::semantic::OperationKind::gated_dense_ffn) {
      assert(operation.inputs.size() == 4);
    }
  }

  const auto lowered = compiler::SemanticLowering{}.lower(module.value(), {120, 16});
  assert(lowered.has_value());
  assert(lowered.value().verify().ok());
  assert(lowered.value().tensors().size() == module.value().tensors().size());
  assert(lowered.value().kernel_requirements().size() == module.value().operations().size());
  assert(lowered.value().state_slots().size() == 128);
  assert(lowered.value().state_transitions().size() == 384);
  assert(lowered.value().entry_points().size() == 1);
  assert(lowered.value().entry_points().front().inputs.size() == 1);
  assert(lowered.value().entry_points().front().outputs.size() == 1);
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
  sm120::BaselineProvider provider;
  const auto physical = sm120::Specializer{}.compile(
      lowered.value(),
      {compiler::TargetProfile::offline_sm120a(32ULL << 30U, "baseline-v1"), 0, 512}, provider);
  assert(!physical.has_value());
  assert(physical.error().code() == base::StatusCode::unsupported);
  assert(!physical.error().context().empty());
  assert(physical.error().context().back() == "embedding");
  assert(physical.error().message().find("Qwen") == std::string::npos);

  const auto rejected_inventory = frontend.validate({std::string{frontends::qwen38::kSourceIdentity}, 1,
                                                     std::string{frontends::qwen38::kTensorInventorySha256}, {}});
  assert(!rejected_inventory.ok());
  const auto rejected = frontend.validate({"Qwen/Qwen3.8-27B@wrong", 0, {}, {}});
  assert(!rejected.ok());
  assert(rejected.code() == base::StatusCode::failed_precondition);
  return 0;
}
