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
  const auto add = [&tensors](std::string name, std::string role, std::string dtype = "BF16",
                              std::vector<std::uint64_t> shape = {1}) {
    const std::uint64_t offset = tensors.size() * 2;
    tensors.push_back({std::move(name), std::move(role), std::move(dtype), std::move(shape), offset, 2});
  };
  add("model.language_model.embed_tokens.weight", "embedding");
  add("model.language_model.norm.weight", "normalization");
  add("lm_head.weight", "lm_head", "U8", {8, 8});
  add("lm_head.weight_scale", "scale", "F8_E4M3");
  for (std::uint32_t layer = 0; layer < 64; ++layer) {
    const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
    add(prefix + "input_layernorm.weight", "normalization");
    add(prefix + "post_attention_layernorm.weight", "normalization");
    add(prefix + "mlp.gate_proj.weight", "feed_forward", "U8", {16, 2560});
    add(prefix + "mlp.up_proj.weight", "feed_forward", "U8", {16, 2560});
    add(prefix + "mlp.down_proj.weight", "feed_forward", "U8", {2560, 8});
    if (layer % 4U == 3U) {
      add(prefix + "self_attn.q_proj.weight", "attention", "U8", {8, 8});
      add(prefix + "self_attn.k_proj.weight", "attention", "U8", {8, 8});
      add(prefix + "self_attn.v_proj.weight", "attention", "U8", {8, 8});
      add(prefix + "self_attn.o_proj.weight", "attention", "U8", {8, 8});
      add(prefix + "self_attn.q_norm.weight", "attention");
      add(prefix + "self_attn.k_norm.weight", "attention");
    } else {
      add(prefix + "linear_attn.in_proj_qkv.weight", "attention", "U8", {8, 8});
      add(prefix + "linear_attn.in_proj_z.weight", "attention", "U8", {8, 8});
      add(prefix + "linear_attn.in_proj_a.weight", "attention", "BF16");
      add(prefix + "linear_attn.in_proj_b.weight", "attention", "BF16");
      add(prefix + "linear_attn.out_proj.weight", "attention", "U8", {8, 8});
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
      assert(module.value().tensors()[operation.inputs[0].value()].name == "final_norm");
    }
    if (operation.name == "final_norm") {
      assert(operation.kind == ir::semantic::OperationKind::rms_norm);
      assert(operation.inputs.size() == 2);
      assert(module.value().tensors()[operation.inputs[1].value()].name ==
             "weight/model.language_model.norm.weight");
      assert(operation.attributes.epsilon == 1.0e-6F);
      assert(operation.attributes.norm_scale_convention ==
             ir::semantic::NormScaleConvention::one_plus_weight);
    }
  }
  for (const auto& tensor : module.value().tensors()) {
    if (tensor.name == "weight/lm_head.weight") {
      assert(tensor.spec.shape.size() == 2);
      assert(!tensor.spec.shape[1].is_symbolic && tensor.spec.shape[1].value == 16);
    }
  }
  auto malformed_source = source_tensors();
  for (auto& tensor : malformed_source) {
    if (tensor.name == "lm_head.weight") tensor.shape = {8};
  }
  const auto malformed = frontend.emit(
      {std::string{frontends::qwen38::kSourceIdentity}, frontends::qwen38::kTensorCount,
       std::string{frontends::qwen38::kTensorInventorySha256}, std::move(malformed_source)});
  assert(!malformed.has_value());
  assert(malformed.error().message().find("packed NVFP4") != std::string::npos);

  std::size_t gated_delta = 0;
  std::size_t full_attention = 0;
  std::size_t qwen_rms_norms = 0;
  for (const auto& operation : module.value().operations()) {
    gated_delta += operation.kind == ir::semantic::OperationKind::gated_delta_attention;
    full_attention += operation.kind == ir::semantic::OperationKind::gated_grouped_query_attention;
    if (operation.kind == ir::semantic::OperationKind::rms_norm) {
      ++qwen_rms_norms;
      assert(operation.attributes.epsilon == 1.0e-6F);
      assert(operation.attributes.norm_scale_convention ==
             ir::semantic::NormScaleConvention::one_plus_weight);
    }
  }
  assert(gated_delta == 48);
  assert(full_attention == 16);
  assert(qwen_rms_norms == 129);
  for (const auto& operation : module.value().operations()) {
    if (operation.kind == ir::semantic::OperationKind::gated_delta_attention ||
        operation.kind == ir::semantic::OperationKind::gated_grouped_query_attention) {
      assert(operation.inputs.size() >= 9);
      assert(operation.outputs.size() == 3);
    }
    if (operation.kind == ir::semantic::OperationKind::gated_delta_attention) {
      assert(operation.attributes.num_heads == 16);
      assert(operation.attributes.num_kv_heads == 16);
      assert(operation.attributes.value_head_count == 48);
    }
    if (operation.kind == ir::semantic::OperationKind::gated_grouped_query_attention) {
      assert(operation.attributes.attention_output_gate ==
             ir::semantic::AttentionOutputGate::sigmoid);
      assert(operation.attributes.rope_theta == 10000000.0F);
      assert(module.value().tensors()[operation.inputs[1].value()].spec.shape.size() == 3);
      assert(module.value().tensors()[operation.inputs[1].value()].spec.shape[0].value ==
             frontends::qwen38::kMaxContext);
      assert(module.value().tensors()[operation.inputs[1].value()].spec.shape[1].value == 4);
      assert(module.value().tensors()[operation.inputs[1].value()].spec.shape[2].value == 256);
      assert(module.value().tensors()[operation.inputs[2].value()].spec.shape.size() == 3);
      assert(module.value().tensors()[operation.inputs[2].value()].spec.shape[0].value ==
             frontends::qwen38::kMaxContext);
      assert(module.value().tensors()[operation.inputs[2].value()].spec.shape[1].value == 4);
      assert(module.value().tensors()[operation.inputs[2].value()].spec.shape[2].value == 256);
    }
    if (operation.kind == ir::semantic::OperationKind::gated_dense_ffn) {
      assert(operation.inputs.size() == 4);
    }
  }

  const auto lowered = compiler::SemanticLowering{}.lower(module.value(), {120, 16});
  assert(lowered.has_value());
  assert(lowered.value().verify().ok());
  assert(lowered.value().tensors().size() > module.value().tensors().size());
  assert(lowered.value().kernel_requirements().size() > module.value().operations().size());
  std::size_t lowered_casts = 0;
  std::size_t lowered_nvfp4_projections = 0;
  std::size_t lowered_silu_mul = 0;
  std::size_t lowered_sigmoid_mul = 0;
  std::size_t lowered_splits = 0;
  std::size_t lowered_last_splits = 0;
  std::size_t lowered_rope = 0;
  std::size_t lowered_cache_appends = 0;
  std::size_t lowered_cached_attention = 0;
  std::size_t lowered_gated_delta = 0;
  std::size_t lowered_delta_parameters = 0;
  std::size_t lowered_convolution = 0;
  std::size_t lowered_linear = 0;
  std::size_t lowered_one_plus_norms = 0;
  for (const auto& requirement : lowered.value().kernel_requirements()) {
    lowered_casts += requirement.operation == "cast";
    if (requirement.operation == "nvfp4_linear") {
      ++lowered_nvfp4_projections;
      assert(requirement.operands.size() == 5);
    }
    lowered_silu_mul += requirement.operation == "silu_mul";
    lowered_sigmoid_mul += requirement.operation == "sigmoid_mul";
    lowered_splits += requirement.operation == "split";
    lowered_last_splits += requirement.operation == "split_last";
    lowered_rope += requirement.operation == "rope";
    lowered_cache_appends += requirement.operation == "cache_append";
    lowered_cached_attention += requirement.operation == "attention_bf16_cache";
    lowered_gated_delta += requirement.operation == "gated_delta_attention";
    lowered_delta_parameters += requirement.operation == "gated_delta_parameters";
    lowered_convolution += requirement.operation == "causal_conv_silu";
    lowered_linear += requirement.operation == "linear";
    if (requirement.operation == "rms_norm" &&
        requirement.attributes.norm_scale_convention ==
            ir::semantic::NormScaleConvention::one_plus_weight) {
      ++lowered_one_plus_norms;
    }
  }
  assert(lowered_casts > 0);
  assert(lowered_nvfp4_projections == 401);
  assert(lowered_silu_mul == 112);
  assert(lowered_sigmoid_mul == 16);
  assert(lowered_splits == 96);
  assert(lowered_last_splits == 16);
  assert(lowered_rope == 32);
  assert(lowered_cache_appends == 16);
  assert(lowered_cached_attention == 16);
  assert(lowered_gated_delta == 48);
  assert(lowered_delta_parameters == 48);
  assert(lowered_convolution == 48);
  assert(lowered_linear == 96);
  assert(lowered_one_plus_norms == 161);
  bool saw_lm_head_block_scale = false;
  bool saw_lm_head_tensor_scale = false;
  for (const auto& tensor : lowered.value().tensors()) {
    saw_lm_head_block_scale = saw_lm_head_block_scale ||
                              tensor.name == "weight/lm_head.weight_scale";
    saw_lm_head_tensor_scale = saw_lm_head_tensor_scale ||
                               tensor.name == "weight/lm_head.weight_scale_2";
  }
  assert(saw_lm_head_block_scale && saw_lm_head_tensor_scale);
  assert(lowered.value().state_slots().size() == 128);
  assert(lowered.value().state_transitions().size() == 384);
  for (const auto& tensor : lowered.value().tensors()) {
    if (tensor.name.ends_with("$fp32")) {
      assert(tensor.role == ir::semantic::TensorRole::activation);
    }
  }
  assert(lowered.value().entry_points().size() == 1);
  assert(lowered.value().entry_points().front().inputs.size() == 1);
  assert(lowered.value().entry_points().front().outputs.size() == 1);
  std::size_t lowered_kv = 0;
  std::size_t lowered_decode_state = 0;
  for (const auto& tensor : lowered.value().tensors()) {
    lowered_kv += tensor.role == ir::semantic::TensorRole::kv_cache;
    lowered_decode_state += tensor.role == ir::semantic::TensorRole::decode_state;
  }
  assert(lowered_kv == 160);
  assert(lowered_decode_state == 96);
  const auto capped_lowered = compiler::SemanticLowering{}.lower(module.value(), {120, 16, 4096});
  assert(capped_lowered.has_value());
  for (const auto& tensor : capped_lowered.value().tensors()) {
    if (tensor.role == ir::semantic::TensorRole::kv_cache && tensor.physical_shape.size() == 3) {
      assert(tensor.physical_shape[0] <= 4096);
    }
  }
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
      {compiler::TargetProfile::offline_sm120a(32ULL << 30U, "baseline-v1"), 0, 4096}, provider);
  assert(physical.has_value());
  assert(physical.value().plan.verify().ok());
  assert(physical.value().plan.commands().size() == lowered.value().kernel_requirements().size());
  assert(physical.value().plan.commands().size() > 2000);
  std::size_t cached_attention_commands = 0;
  std::size_t recurrent_commands = 0;
  for (const auto& command : physical.value().plan.commands()) {
    cached_attention_commands += command.kernel.value() == 23;
    if (command.kernel.value() == 26) assert(command.split.outer == 1);
    if (command.kernel.value() == 15) {
      ++recurrent_commands;
      assert(command.attention.query_heads == 16);
      assert(command.attention.key_value_heads == 16);
      assert(command.attention.value_heads == 48);
      assert(command.attention.head_dimension == 128);
      assert(command.attention.value_dimension == 128);
      assert(command.attention.positions == 1);
    }
    if (command.kernel.value() == 23) {
      assert(command.attention.query_heads == 24);
      assert(command.attention.key_value_heads == 4);
      assert(command.attention.head_dimension == 256);
      assert(command.attention.positions == 1);
    }
    if (command.kernel.value() == 22) {
      assert(command.cache_append.heads == 4);
      assert(command.cache_append.head_dimension == 256);
      assert(command.cache_append.position == 0);
      assert(command.cache_append.capacity == 262144);
    }
  }
  assert(cached_attention_commands == 16);
  assert(recurrent_commands == 48);

  const auto rejected_inventory = frontend.validate({std::string{frontends::qwen38::kSourceIdentity}, 1,
                                                     std::string{frontends::qwen38::kTensorInventorySha256}, {}});
  assert(!rejected_inventory.ok());
  const auto rejected = frontend.validate({"Qwen/Qwen3.8-27B@wrong", 0, {}, {}});
  assert(!rejected.ok());
  assert(rejected.code() == base::StatusCode::failed_precondition);
  return 0;
}
