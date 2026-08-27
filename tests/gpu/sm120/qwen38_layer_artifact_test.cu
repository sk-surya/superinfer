#include <sm120/runtime/cuda_plan_executor.cuh>
#include <superinfer/artifact/sinf.hpp>
#include <superinfer/artifact/tensor_table.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

class MappedFile final {
 public:
  explicit MappedFile(const char* path) {
    if (path == nullptr) return;
    fd_ = ::open(path, O_RDONLY);
    if (fd_ < 0) return;
    struct stat info {};
    if (::fstat(fd_, &info) != 0 || info.st_size <= 0) return;
    size_ = static_cast<std::size_t>(info.st_size);
    data_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) data_ = nullptr;
  }

  ~MappedFile() {
    if (data_ != nullptr) ::munmap(data_, size_);
    if (fd_ >= 0) ::close(fd_);
  }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  [[nodiscard]] bool valid() const noexcept { return data_ != nullptr; }
  [[nodiscard]] superinfer::base::ConstByteView bytes() const noexcept {
    return {static_cast<const std::byte*>(data_), size_};
  }

 private:
  int fd_{-1};
  void* data_{nullptr};
  std::size_t size_{0};
};

struct BufferBinding final {
  std::string name;
  superinfer::ir::physical::BufferId id;
  const superinfer::artifact::TensorTableRecord* record{nullptr};
};

superinfer::ir::physical::PhysicalTensorDescriptor descriptor_for(
    const superinfer::artifact::TensorTableRecord& record) {
  using namespace superinfer::ir::physical;
  PhysicalDType dtype = PhysicalDType::unknown;
  if (record.physical_dtype == "f32") dtype = PhysicalDType::f32;
  if (record.physical_dtype == "bf16") dtype = PhysicalDType::bf16;
  if (record.physical_dtype == "u8") dtype = PhysicalDType::u8;
  StorageEncoding encoding = StorageEncoding::none;
  if (record.storage_encoding == "nvfp4_packed") encoding = StorageEncoding::nvfp4_packed;
  if (record.storage_encoding == "fp8_e4m3_group_scale") {
    encoding = StorageEncoding::fp8_e4m3_group_scale;
  }
  return {dtype, record.logical_shape, PhysicalLayout::row_major, 256, encoding, record.name};
}

superinfer::ir::physical::PhysicalTensorDescriptor f32_descriptor(
    std::vector<std::uint64_t> shape) {
  return {superinfer::ir::physical::PhysicalDType::f32, std::move(shape),
          superinfer::ir::physical::PhysicalLayout::row_major, 256,
          superinfer::ir::physical::StorageEncoding::none, {}};
}

superinfer::ir::physical::Plan make_layer_plan(
    const std::vector<superinfer::artifact::TensorTableRecord>& records,
    std::vector<BufferBinding>& bindings) {
  using namespace superinfer;
  using namespace ir::physical;
  std::unordered_map<std::string, const artifact::TensorTableRecord*> by_name;
  for (const auto& record : records) by_name.emplace(record.name, &record);

  PlanBuilder builder;
  std::uint64_t offset = 0;
  auto add_buffer = [&](std::string name, std::uint64_t bytes,
                        PhysicalTensorDescriptor descriptor,
                        const artifact::TensorTableRecord* record = nullptr) {
    offset = (offset + 255U) & ~std::uint64_t{255U};
    const auto id = builder.add_buffer(offset, bytes, 256, std::move(descriptor));
    assert(id.has_value());
    offset += bytes;
    bindings.push_back({std::move(name), id.value(), record});
    return id.value();
  };
  auto add_artifact = [&](const std::string& name) {
    const auto found = by_name.find(name);
    if (found == by_name.end()) {
      std::cerr << "artifact tensor is missing: " << name << "\n";
      std::abort();
    }
    const auto& record = *found->second;
    if (record.payload_end <= record.payload_offset) {
      std::cerr << "artifact tensor has invalid payload range: " << name << "\n";
      std::abort();
    }
    return add_buffer(name, record.payload_end - record.payload_offset,
                      descriptor_for(record), &record);
  };
  auto add_f32 = [&](const std::string& name, std::uint64_t elements,
                     std::vector<std::uint64_t> shape = {}) {
    if (shape.empty()) shape = {elements};
    return add_buffer(name, elements * sizeof(float), f32_descriptor(std::move(shape)));
  };
  auto weight = [&](const std::string& suffix) {
    return add_artifact("model.language_model.layers.3." + suffix);
  };
  auto scale = [&](const std::string& suffix) { return weight(suffix + "_scale"); };
  auto tensor_scale = [&](const std::string& suffix) { return weight(suffix + "_scale_2"); };

  const auto hidden = add_f32("input", 5120);
  const auto input_norm = weight("input_layernorm.weight");
  const auto normalized = add_f32("normalized", 5120);
  const auto q_weight = weight("self_attn.q_proj.weight");
  const auto q_scale = scale("self_attn.q_proj.weight");
  const auto q_tensor_scale = tensor_scale("self_attn.q_proj.weight");
  const auto q_projection = add_f32("q_projection", 12288);
  const auto q = add_f32("q", 6144);
  const auto gate = add_f32("gate", 6144);
  const auto k_weight = weight("self_attn.k_proj.weight");
  const auto k_scale = scale("self_attn.k_proj.weight");
  const auto k_tensor_scale = tensor_scale("self_attn.k_proj.weight");
  const auto k_projection = add_f32("k_projection", 1024);
  const auto v_weight = weight("self_attn.v_proj.weight");
  const auto v_scale = scale("self_attn.v_proj.weight");
  const auto v_tensor_scale = tensor_scale("self_attn.v_proj.weight");
  const auto v_projection = add_f32("v_projection", 1024);
  const auto q_norm_weight = weight("self_attn.q_norm.weight");
  const auto q_norm = add_f32("q_norm", 6144);
  const auto k_norm_weight = weight("self_attn.k_norm.weight");
  const auto k_norm = add_f32("k_norm", 1024);
  const auto q_rope = add_f32("q_rope", 6144);
  const auto k_rope = add_f32("k_rope", 1024);
  const auto key_cache = add_buffer("key_cache", 4U * 256U * sizeof(std::uint16_t),
                                    {PhysicalDType::bf16, {1, 4, 256}, PhysicalLayout::row_major,
                                     256, StorageEncoding::none, {}});
  const auto value_cache = add_buffer("value_cache", 4U * 256U * sizeof(std::uint16_t),
                                      {PhysicalDType::bf16, {1, 4, 256}, PhysicalLayout::row_major,
                                       256, StorageEncoding::none, {}});
  const auto attended = add_f32("attended", 6144);
  const auto gated = add_f32("gated", 6144);
  const auto o_weight = weight("self_attn.o_proj.weight");
  const auto o_scale = scale("self_attn.o_proj.weight");
  const auto o_tensor_scale = tensor_scale("self_attn.o_proj.weight");
  const auto attention_output = add_f32("attention_output", 5120);
  const auto residual = add_f32("residual", 5120);
  const auto post_norm_weight = weight("post_attention_layernorm.weight");
  const auto post_norm = add_f32("post_norm", 5120);
  const auto gate_weight = weight("mlp.gate_proj.weight");
  const auto gate_scale = scale("mlp.gate_proj.weight");
  const auto gate_tensor_scale = tensor_scale("mlp.gate_proj.weight");
  const auto gate_projection = add_f32("gate_projection", 17408);
  const auto up_weight = weight("mlp.up_proj.weight");
  const auto up_scale = scale("mlp.up_proj.weight");
  const auto up_tensor_scale = tensor_scale("mlp.up_proj.weight");
  const auto up_projection = add_f32("up_projection", 17408);
  const auto gated_mlp = add_f32("gated_mlp", 17408);
  const auto down_weight = weight("mlp.down_proj.weight");
  const auto down_scale = scale("mlp.down_proj.weight");
  const auto down_tensor_scale = tensor_scale("mlp.down_proj.weight");
  const auto mlp_output = add_f32("mlp_output", 5120);
  const auto output = add_f32("output", 5120);

  std::vector<CommandId> dependency;
  auto command = [&](std::uint64_t kernel, std::vector<BufferId> operands,
                     float epsilon = 1.0e-5F, float scalar = 1.0F,
                     AttentionDimensions attention = {}, bool add_one = false,
                     RopeDimensions rope = {}, CacheAppendDimensions cache = {},
                     SplitDimensions split = {}) {
    const auto result = builder.add_command(base::KernelId{kernel}, std::move(operands), dependency,
                                            0, 0, 0, epsilon, scalar, attention, add_one, {}, rope,
                                            cache, {}, split);
    assert(result.has_value());
    dependency = {result.value()};
  };
  command(12, {hidden, normalized, input_norm}, 1.0e-6F, 1.0F, {}, true);
  command(13, {normalized, q_weight, q_scale, q_tensor_scale, q_projection});
  command(26, {q_projection, q, gate}, 1.0e-5F, 1.0F, {}, false, {}, {}, {24, 256, 256});
  command(13, {normalized, k_weight, k_scale, k_tensor_scale, k_projection});
  command(13, {normalized, v_weight, v_scale, v_tensor_scale, v_projection});
  command(12, {q, q_norm, q_norm_weight}, 1.0e-6F, 1.0F, {}, true);
  command(12, {k_projection, k_norm, k_norm_weight}, 1.0e-6F, 1.0F, {}, true);
  command(20, {q_norm, q_rope}, 1.0e-5F, 10000000.0F, {}, false, {24, 256, 64, 0});
  command(20, {k_norm, k_rope}, 1.0e-5F, 10000000.0F, {}, false, {4, 256, 64, 0});
  command(22, {k_rope, v_projection, key_cache, value_cache}, 1.0e-5F, 1.0F, {}, false, {},
          {4, 256, 0, 1});
  command(23, {q_rope, key_cache, value_cache, attended}, 1.0e-5F, 1.0F, {24, 4, 256, 1});
  command(19, {gate, attended, gated});
  command(13, {gated, o_weight, o_scale, o_tensor_scale, attention_output});
  command(4, {hidden, attention_output, residual});
  command(12, {residual, post_norm, post_norm_weight}, 1.0e-6F, 1.0F, {}, true);
  command(13, {post_norm, gate_weight, gate_scale, gate_tensor_scale, gate_projection});
  command(13, {post_norm, up_weight, up_scale, up_tensor_scale, up_projection});
  command(18, {gate_projection, up_projection, gated_mlp});
  command(13, {gated_mlp, down_weight, down_scale, down_tensor_scale, mlp_output});
  command(4, {residual, mlp_output, output});
  assert(builder.add_entry_point("layer3", {hidden}, {output}).ok());
  builder.set_resource_bounds({offset + 256U, 0, 32});
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

const BufferBinding& binding(const std::vector<BufferBinding>& bindings, const std::string& name) {
  for (const auto& item : bindings) {
    if (item.name == name) return item;
  }
  std::cerr << "binding is missing: " << name << " available=";
  for (const auto& item : bindings) std::cerr << "[" << item.name << "]";
  std::cerr << "\n";
  std::abort();
}

int skip_if_unconfigured() {
  std::cerr << "SKIP: set SUPERINFER_QWEN38_ARTIFACT and SUPERINFER_QWEN38_REFERENCE_F32\n";
  return 77;
}

}  // namespace

int main() {
  const char* artifact_path = std::getenv("SUPERINFER_QWEN38_ARTIFACT");
  const char* expected_path = std::getenv("SUPERINFER_QWEN38_REFERENCE_F32");
  if (artifact_path == nullptr || expected_path == nullptr) return skip_if_unconfigured();
  MappedFile artifact_file{artifact_path};
  if (!artifact_file.valid()) {
    std::cerr << "artifact mapping failed\n";
    return 1;
  }
  const auto artifact = superinfer::artifact::ArtifactReader::read(artifact_file.bytes());
  if (!artifact.has_value()) {
    std::cerr << "artifact validation failed: " << artifact.error().message() << '\n';
    return 1;
  }
  const auto table_section = artifact.value().section(superinfer::artifact::SectionKind::tensor_table);
  if (!table_section.has_value()) return 1;
  const auto records = superinfer::artifact::parse_tensor_table(table_section.value());
  if (!records.has_value()) {
    std::cerr << "tensor table parse failed: " << records.error().message() << '\n';
    return 1;
  }
  std::vector<BufferBinding> bindings;
  const auto plan = make_layer_plan(records.value(), bindings);
  auto session_result = superinfer::sm120::cuda_runtime::CudaPlanSession::create(plan, 120, "baseline-v1");
  if (!session_result.has_value()) {
    std::cerr << "session creation failed: " << session_result.error().message() << '\n';
    return 1;
  }
  auto session = std::move(session_result).value();
  for (const auto& item : bindings) {
    if (item.record == nullptr) continue;
    const auto payload = artifact.value().payload_range(
        item.record->payload_offset, item.record->payload_end - item.record->payload_offset);
    if (!payload.has_value() || !session.copy_to_device(item.id, payload.value()).ok()) {
      std::cerr << "payload binding failed: " << item.name << '\n';
      return 1;
    }
  }
  std::vector<float> hidden(5120);
  for (std::size_t index = 0; index < hidden.size(); ++index) {
    hidden[index] = -0.25F + 0.5F * static_cast<float>(index) / 5119.0F;
  }
  if (!session.copy_to_device(binding(bindings, "input").id,
                              {reinterpret_cast<const std::byte*>(hidden.data()),
                               hidden.size() * sizeof(float)}).ok()) {
    return 1;
  }
  if (!session.execute().ok() || !session.synchronize_for_test().ok()) {
    std::cerr << "layer execution failed\n";
    return 1;
  }
  std::ifstream expected_stream{expected_path, std::ios::binary};
  std::vector<float> expected(5120);
  expected_stream.read(reinterpret_cast<char*>(expected.data()),
                       static_cast<std::streamsize>(expected.size() * sizeof(float)));
  if (!expected_stream || expected_stream.peek() != std::ifstream::traits_type::eof()) {
    std::cerr << "reference output must contain exactly 5120 FP32 values\n";
    return 1;
  }
  std::vector<float> actual(expected.size());
  if (!session.copy_from_device(binding(bindings, "output").id,
                                {reinterpret_cast<std::byte*>(actual.data()),
                                 actual.size() * sizeof(float)}).ok()) {
    return 1;
  }
  float maximum = 0.0F;
  float mean = 0.0F;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float error = std::fabs(actual[index] - expected[index]);
    maximum = std::max(maximum, error);
    mean += error;
  }
  mean /= static_cast<float>(actual.size());
  std::cout << "qwen38 layer3 artifact differential max_abs=" << maximum
            << " mean_abs=" << mean << " commands=" << session.trace().commands_executed << '\n';
  return maximum <= 2.0e-2F && mean <= 2.0e-4F ? 0 : 1;
}
