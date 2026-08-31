#include <sm120/runtime/cuda_plan_executor.cuh>
#include <superinfer/artifact/sinf.hpp>
#include <superinfer/artifact/tensor_table.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
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

using Record = superinfer::artifact::TensorTableRecord;
using BufferId = superinfer::ir::physical::BufferId;

struct Binding final {
  std::string name;
  BufferId id;
  const Record* record{nullptr};
  std::vector<std::byte> host_bytes;
};

superinfer::ir::physical::PhysicalTensorDescriptor descriptor_for(const Record& record) {
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

float bf16_to_float(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::uint16_t float_to_bf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>(bits >> 16U);
}

template <typename T>
bool read_exact_values(const char* path, std::vector<T>& values) {
  std::ifstream input{path, std::ios::binary};
  if (!input.good()) return false;
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(T)));
  return input.good() && input.peek() == std::ifstream::traits_type::eof();
}

std::vector<std::byte> convert_bf16_to_f32(superinfer::base::ConstByteView bytes) {
  assert(bytes.size() % sizeof(std::uint16_t) == 0);
  std::vector<std::byte> result(bytes.size() / sizeof(std::uint16_t) * sizeof(float));
  for (std::size_t index = 0; index < bytes.size() / sizeof(std::uint16_t); ++index) {
    std::uint16_t encoded = 0;
    std::memcpy(&encoded, bytes.data() + index * sizeof(encoded), sizeof(encoded));
    const float value = bf16_to_float(encoded);
    std::memcpy(result.data() + index * sizeof(value), &value, sizeof(value));
  }
  return result;
}

std::uint32_t diagnostic_segments() {
  const char* encoded = std::getenv("SUPERINFER_QWEN38_GDN_SEGMENTS");
  if (encoded == nullptr) return 2;
  char* end = nullptr;
  errno = 0;
  const unsigned long parsed = std::strtoul(encoded, &end, 10);
  if (errno != 0 || end == encoded || *end != '\0' || parsed == 0 || parsed > 4096U) {
    std::cerr << "invalid SUPERINFER_QWEN38_GDN_SEGMENTS\n";
    std::abort();
  }
  return static_cast<std::uint32_t>(parsed);
}

superinfer::ir::physical::Plan make_plan(
    const superinfer::artifact::ArtifactView& artifact,
    const std::vector<Record>& records, std::vector<Binding>& bindings) {
  using namespace superinfer;
  using namespace ir::physical;
  std::unordered_map<std::string, const Record*> by_name;
  for (const auto& record : records) by_name.emplace(record.name, &record);

  PlanBuilder builder;
  std::uint64_t offset = 0;
  auto add_buffer = [&](std::string name, std::uint64_t bytes,
                        PhysicalTensorDescriptor descriptor,
                        const Record* record = nullptr,
                        std::vector<std::byte> host_bytes = {}) {
    offset = (offset + 255U) & ~std::uint64_t{255U};
    const auto id = builder.add_buffer(offset, bytes, 256, std::move(descriptor));
    assert(id.has_value());
    offset += bytes;
    bindings.push_back({std::move(name), id.value(), record, std::move(host_bytes)});
    return id.value();
  };
  auto add_artifact = [&](const std::string& name) {
    const auto found = by_name.find(name);
    if (found == by_name.end()) {
      std::cerr << "artifact tensor is missing: " << name << "\n";
      std::abort();
    }
    const auto& record = *found->second;
    const auto bytes = record.payload_end - record.payload_offset;
    return add_buffer(name, bytes, descriptor_for(record), &record);
  };
  auto add_f32_artifact = [&](const std::string& name) {
    const auto found = by_name.find(name);
    if (found == by_name.end()) {
      std::cerr << "artifact tensor is missing: " << name << "\n";
      std::abort();
    }
    const auto& record = *found->second;
    const auto payload = artifact.payload_range(record.payload_offset,
                                                record.payload_end - record.payload_offset);
    assert(payload.has_value());
    std::vector<std::byte> host = record.physical_dtype == "bf16"
                                      ? convert_bf16_to_f32(payload.value())
                                      : std::vector<std::byte>(payload.value().data(),
                                                               payload.value().data() + payload.value().size());
    const auto host_bytes = host.size();
    return add_buffer(name + "$f32", host_bytes,
                      f32_descriptor(record.logical_shape), &record, std::move(host));
  };
  auto add_f32 = [&](std::string name, std::uint64_t elements,
                     std::vector<std::uint64_t> shape = {}) {
    if (shape.empty()) shape = {elements};
    return add_buffer(std::move(name), elements * sizeof(float), f32_descriptor(std::move(shape)));
  };
  auto add_bf16_zero = [&](std::string name, std::uint64_t elements,
                           std::vector<std::uint64_t> shape) {
    std::vector<std::byte> zero(elements * sizeof(std::uint16_t));
    const auto bytes = zero.size();
    return add_buffer(std::move(name), bytes,
                      {PhysicalDType::bf16, std::move(shape), PhysicalLayout::row_major, 256,
                       StorageEncoding::none, {}}, nullptr, std::move(zero));
  };
  auto weight = [&](const std::string& suffix) {
    return add_artifact("model.language_model.layers.0." + suffix);
  };
  auto f32_weight = [&](const std::string& suffix) {
    return add_f32_artifact("model.language_model.layers.0." + suffix);
  };
  auto scale = [&](const std::string& suffix) { return weight(suffix + "_scale"); };
  auto tensor_scale = [&](const std::string& suffix) { return weight(suffix + "_scale_2"); };

  const auto hidden = add_f32("input", 5120);
  const auto input_norm_weight = weight("input_layernorm.weight");
  const auto normalized = add_f32("normalized", 5120);
  const auto qkv_weight = weight("linear_attn.in_proj_qkv.weight");
  const auto qkv_scale = scale("linear_attn.in_proj_qkv.weight");
  const auto qkv_tensor_scale = tensor_scale("linear_attn.in_proj_qkv.weight");
  const auto qkv_projection = add_f32("qkv_projection", 10240);
  const auto conv_weight = f32_weight("linear_attn.conv1d.weight");
  const auto conv_state = add_bf16_zero("conv_state", 4U * 10240U, {4, 10240});
  const auto convolved = add_f32("convolved", 10240);
  const auto query = add_f32("query", 2048);
  const auto qk_remainder = add_f32("qk_remainder", 8192);
  const auto key = add_f32("key", 2048);
  const auto value = add_f32("value", 6144);
  const auto z_weight = weight("linear_attn.in_proj_z.weight");
  const auto z_scale = scale("linear_attn.in_proj_z.weight");
  const auto z_tensor_scale = tensor_scale("linear_attn.in_proj_z.weight");
  const auto z_projection = add_f32("z_projection", 6144);
  const auto a_weight = f32_weight("linear_attn.in_proj_a.weight");
  const auto a = add_f32("a", 48);
  const auto b_weight = f32_weight("linear_attn.in_proj_b.weight");
  const auto b = add_f32("b", 48);
  const auto a_log = f32_weight("linear_attn.A_log");
  const auto dt_bias = f32_weight("linear_attn.dt_bias");
  const auto log_decay = add_f32("log_decay", 48);
  const auto beta = add_f32("beta", 48);
  const auto delta_state = add_f32("delta_state", 48U * 128U * 128U, {48, 128, 128});
  const auto core = add_f32("core", 6144);
  const auto output_norm_weight = weight("linear_attn.norm.weight");
  const auto normalized_core = add_f32("normalized_core", 6144);
  const auto gated = add_f32("gated", 6144);
  const auto out_weight = weight("linear_attn.out_proj.weight");
  const auto out_scale = scale("linear_attn.out_proj.weight");
  const auto out_tensor_scale = tensor_scale("linear_attn.out_proj.weight");
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
                     ConvolutionDimensions convolution = {}) {
    const auto result = builder.add_command(base::KernelId{kernel}, std::move(operands), dependency,
                                            0, 0, 0, epsilon, scalar, attention, add_one, {}, {},
                                            {}, convolution);
    assert(result.has_value());
    dependency = {result.value()};
  };
  command(12, {hidden, normalized, input_norm_weight}, 1.0e-6F, 1.0F, {}, true);
  command(13, {normalized, qkv_weight, qkv_scale, qkv_tensor_scale, qkv_projection});
  command(25, {qkv_projection, conv_weight, conv_state, convolved}, 1.0e-5F, 1.0F, {}, false,
          {10240, 4});
  command(21, {convolved, query, qk_remainder});
  command(21, {qk_remainder, key, value});
  command(13, {normalized, z_weight, z_scale, z_tensor_scale, z_projection});
  command(10, {normalized, a_weight, a});
  command(10, {normalized, b_weight, b});
  command(24, {a, b, a_log, dt_bias, log_decay, beta});
  command(15, {query, key, value, log_decay, beta, delta_state, core}, 1.0e-5F, 1.0F,
          {16, 16, 128, 1, 48, 128});
  command(12, {core, normalized_core, output_norm_weight}, 1.0e-6F);
  command(18, {z_projection, normalized_core, gated});
  command(13, {gated, out_weight, out_scale, out_tensor_scale, attention_output});
  command(4, {hidden, attention_output, residual});
  command(12, {residual, post_norm, post_norm_weight}, 1.0e-6F, 1.0F, {}, true);
  command(13, {post_norm, gate_weight, gate_scale, gate_tensor_scale, gate_projection});
  command(13, {post_norm, up_weight, up_scale, up_tensor_scale, up_projection});
  command(18, {gate_projection, up_projection, gated_mlp});
  command(13, {gated_mlp, down_weight, down_scale, down_tensor_scale, mlp_output});
  command(4, {residual, mlp_output, output});
  assert(builder.add_entry_point("gdn_layer0", {hidden}, {output}).ok());
  builder.set_resource_bounds({offset + 256U, 0, 32});
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

const Binding& binding(const std::vector<Binding>& bindings, const std::string& name) {
  for (const auto& item : bindings) {
    if (item.name == name) return item;
  }
  std::abort();
}

int skip_if_unconfigured() {
  std::cerr << "SKIP: set SUPERINFER_QWEN38_ARTIFACT and SUPERINFER_QWEN38_GDN_REFERENCE_F32\n";
  return 77;
}

}  // namespace

int main() {
  const char* artifact_path = std::getenv("SUPERINFER_QWEN38_ARTIFACT");
  const char* expected_path = std::getenv("SUPERINFER_QWEN38_GDN_REFERENCE_F32");
  const char* expected_qkv_path = std::getenv("SUPERINFER_QWEN38_GDN_QKV_F32");
  const char* expected_conv_path = std::getenv("SUPERINFER_QWEN38_GDN_CONV_F32");
  const char* expected_core_path = std::getenv("SUPERINFER_QWEN38_GDN_CORE_F32");
  const char* expected_gated_path = std::getenv("SUPERINFER_QWEN38_GDN_GATED_F32");
  const char* custom_input_path = std::getenv("SUPERINFER_QWEN38_GDN_INPUT_F32");
  const char* custom_state_path = std::getenv("SUPERINFER_QWEN38_GDN_STATE_F32");
  const char* conv_capture_path = std::getenv("SUPERINFER_QWEN38_GDN_CONV_OUTPUT_F32");
  if (artifact_path == nullptr || expected_path == nullptr) return skip_if_unconfigured();
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cerr << "no CUDA device\n";
    return 77;
  }
  if (cudaSetDevice(0) != cudaSuccess) {
    std::cerr << "cudaSetDevice(0) failed\n";
    return 77;
  }
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess || properties.major != 12 ||
      properties.minor != 0) {
    std::cerr << "GPU is not sm_120\n";
    return 77;
  }

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
  const auto table = artifact.value().section(superinfer::artifact::SectionKind::tensor_table);
  if (!table.has_value()) {
    std::cerr << "tensor table section missing\n";
    return 1;
  }
  const auto records = superinfer::artifact::parse_tensor_table(table.value());
  if (!records.has_value()) {
    std::cerr << "tensor table parse failed: " << records.error().message() << '\n';
    return 1;
  }
  std::vector<Binding> bindings;
  const auto plan = make_plan(artifact.value(), records.value(), bindings);
  auto session_result = superinfer::sm120::cuda_runtime::CudaPlanSession::create(plan, 120, "baseline-v1");
  if (!session_result.has_value()) {
    std::cerr << "session creation failed: " << session_result.error().message() << '\n';
    return 1;
  }
  auto session = std::move(session_result).value();
  const bool custom_step = custom_input_path != nullptr || custom_state_path != nullptr;
  const std::uint32_t segments = diagnostic_segments();
  if (custom_step && (custom_input_path == nullptr || segments != 1)) {
    std::cerr << "custom GDN execution requires input and segments=1\n";
    return 1;
  }
  for (const auto& item : bindings) {
    if (!item.host_bytes.empty()) {
      if (!session.copy_to_device(item.id, {item.host_bytes.data(), item.host_bytes.size()}).ok()) {
        std::cerr << "host upload failed: " << item.name << '\n';
        return 1;
      }
    } else if (item.record != nullptr) {
      const auto payload = artifact.value().payload_range(
          item.record->payload_offset, item.record->payload_end - item.record->payload_offset);
      if (!payload.has_value() || !session.copy_to_device(item.id, payload.value()).ok()) {
        std::cerr << "artifact upload failed: " << item.name << '\n';
        return 1;
      }
    }
  }
  if (custom_step && custom_state_path != nullptr) {
    std::vector<float> state(48U * 128U * 128U + 4U * 10240U);
    if (!read_exact_values(custom_state_path, state)) {
      std::cerr << "custom GDN state has an unexpected size\n";
      return 1;
    }
    std::vector<std::uint16_t> convolution(4U * 10240U);
    for (std::size_t index = 0; index < convolution.size(); ++index) {
      convolution[index] = float_to_bf16(state[48U * 128U * 128U + index]);
    }
    if (!session.copy_to_device(binding(bindings, "delta_state").id,
                                {reinterpret_cast<const std::byte*>(state.data()),
                                 48U * 128U * 128U * sizeof(float)}).ok() ||
        !session.copy_to_device(binding(bindings, "conv_state").id,
                                {reinterpret_cast<const std::byte*>(convolution.data()),
                                 convolution.size() * sizeof(std::uint16_t)}).ok()) {
      std::cerr << "custom GDN state upload failed\n";
      return 1;
    }
  }
  std::ifstream expected_stream{expected_path, std::ios::binary};
  std::vector<float> expected(static_cast<std::size_t>(segments) * 5120U);
  expected_stream.read(reinterpret_cast<char*>(expected.data()),
                       static_cast<std::streamsize>(expected.size() * sizeof(float)));
  if (!expected_stream || expected_stream.peek() != std::ifstream::traits_type::eof()) {
    std::cerr << "reference output must contain exactly 10240 FP32 values\n";
    return 1;
  }
  std::vector<float> expected_attention(static_cast<std::size_t>(segments) * 5120U);
  std::string attention_path{expected_path};
  const std::size_t extension = attention_path.rfind(".bin");
  if (extension != std::string::npos) attention_path.insert(extension, ".attn");
  std::ifstream attention_stream{attention_path, std::ios::binary};
  attention_stream.read(reinterpret_cast<char*>(expected_attention.data()),
                        static_cast<std::streamsize>(expected_attention.size() * sizeof(float)));
  if (!attention_stream || attention_stream.peek() != std::ifstream::traits_type::eof()) {
    std::cerr << "attention reference output must contain exactly 10240 FP32 values\n";
    return 1;
  }
  std::vector<float> expected_state(
      static_cast<std::size_t>(segments) * 48U * 128U * 128U);
  std::string state_path{expected_path};
  if (extension != std::string::npos) state_path.insert(extension, ".state");
  std::ifstream state_stream{state_path, std::ios::binary};
  state_stream.read(reinterpret_cast<char*>(expected_state.data()),
                    static_cast<std::streamsize>(expected_state.size() * sizeof(float)));
  if (!state_stream || state_stream.peek() != std::ifstream::traits_type::eof()) {
    std::cerr << "state reference output must contain exactly 1572864 FP32 values\n";
    return 1;
  }

  std::vector<float> expected_qkv;
  if (expected_qkv_path != nullptr) {
    expected_qkv.resize(static_cast<std::size_t>(segments) * 10240U);
    std::ifstream qkv_stream{expected_qkv_path, std::ios::binary};
    qkv_stream.read(reinterpret_cast<char*>(expected_qkv.data()),
                    static_cast<std::streamsize>(expected_qkv.size() * sizeof(float)));
    if (!qkv_stream || qkv_stream.peek() != std::ifstream::traits_type::eof()) {
      std::cerr << "qkv reference output must contain exactly 20480 FP32 values\n";
      return 1;
    }
  }
  std::vector<float> expected_conv;
  if (expected_conv_path != nullptr) {
    expected_conv.resize(static_cast<std::size_t>(segments) * 10240U);
    std::ifstream conv_stream{expected_conv_path, std::ios::binary};
    conv_stream.read(reinterpret_cast<char*>(expected_conv.data()),
                     static_cast<std::streamsize>(expected_conv.size() * sizeof(float)));
    if (!conv_stream || conv_stream.peek() != std::ifstream::traits_type::eof()) {
      std::cerr << "convolution reference output must contain exactly 20480 FP32 values\n";
      return 1;
    }
  }
  std::vector<float> expected_core;
  if (expected_core_path != nullptr) {
    expected_core.resize(static_cast<std::size_t>(segments) * 6144U);
    std::ifstream core_stream{expected_core_path, std::ios::binary};
    core_stream.read(reinterpret_cast<char*>(expected_core.data()),
                     static_cast<std::streamsize>(expected_core.size() * sizeof(float)));
    if (!core_stream || core_stream.peek() != std::ifstream::traits_type::eof()) {
      std::cerr << "core reference output must contain exactly 12288 FP32 values\n";
      return 1;
    }
  }
  std::vector<float> expected_gated;
  if (expected_gated_path != nullptr) {
    expected_gated.resize(static_cast<std::size_t>(segments) * 6144U);
    std::ifstream gated_stream{expected_gated_path, std::ios::binary};
    gated_stream.read(reinterpret_cast<char*>(expected_gated.data()),
                      static_cast<std::streamsize>(expected_gated.size() * sizeof(float)));
    if (!gated_stream || gated_stream.peek() != std::ifstream::traits_type::eof()) {
      std::cerr << "gated reference output must contain exactly 12288 FP32 values\n";
      return 1;
    }
  }

  std::vector<float> actual(expected.size());
  float attention_contract_maximum = 0.0F;
  float state_contract_maximum = 0.0F;
  float qkv_contract_maximum = 0.0F;
  float conv_contract_maximum = 0.0F;
  float core_contract_maximum = 0.0F;
  float gated_contract_maximum = 0.0F;
  for (std::size_t segment = 0; segment < segments; ++segment) {
    std::vector<float> hidden(5120);
    if (custom_step) {
      if (!read_exact_values(custom_input_path, hidden)) {
        std::cerr << "custom GDN input has an unexpected size\n";
        return 1;
      }
    } else {
      for (std::size_t index = 0; index < hidden.size(); ++index) {
        hidden[index] = -0.25F + 0.5F * static_cast<float>(index) / 5119.0F +
                        static_cast<float>(segment) * 0.03125F;
      }
    }
    if (!session.copy_to_device(binding(bindings, "input").id,
                                {reinterpret_cast<const std::byte*>(hidden.data()),
                                 hidden.size() * sizeof(float)}).ok() ||
        !session.execute().ok() || !session.synchronize_for_test().ok() ||
        !session.copy_from_device(binding(bindings, "output").id,
                                  {reinterpret_cast<std::byte*>(actual.data() + segment * 5120U),
                                   5120U * sizeof(float)}).ok()) {
      std::cerr << "GDN segment execution or copy failed: " << segment << '\n';
      return 1;
    }
    std::vector<float> attention_actual(5120);
    if (!session.copy_from_device(binding(bindings, "attention_output").id,
                                  {reinterpret_cast<std::byte*>(attention_actual.data()),
                                   attention_actual.size() * sizeof(float)}).ok()) {
      return 1;
    }
    float attention_maximum = 0.0F;
    for (std::size_t index = 0; index < attention_actual.size(); ++index) {
      attention_maximum = std::max(
          attention_maximum,
          std::fabs(attention_actual[index] - expected_attention[segment * 5120U + index]));
    }
    attention_contract_maximum = std::max(attention_contract_maximum, attention_maximum);
    if (!expected_qkv.empty()) {
      std::vector<float> qkv_actual(10240);
      if (!session.copy_from_device(binding(bindings, "qkv_projection").id,
                                    {reinterpret_cast<std::byte*>(qkv_actual.data()),
                                     qkv_actual.size() * sizeof(float)}).ok()) {
        return 1;
      }
      float qkv_maximum = 0.0F;
      for (std::size_t index = 0; index < qkv_actual.size(); ++index) {
        qkv_maximum = std::max(
            qkv_maximum,
            std::fabs(qkv_actual[index] - expected_qkv[segment * qkv_actual.size() + index]));
      }
      qkv_contract_maximum = std::max(qkv_contract_maximum, qkv_maximum);
    }
    if (!expected_conv.empty()) {
      std::vector<float> conv_actual(10240);
      if (!session.copy_from_device(binding(bindings, "convolved").id,
                                    {reinterpret_cast<std::byte*>(conv_actual.data()),
                                     conv_actual.size() * sizeof(float)}).ok()) {
        return 1;
      }
      if (conv_capture_path != nullptr) {
        std::ofstream capture{conv_capture_path, std::ios::binary | std::ios::trunc};
        if (!capture.good()) return 1;
        capture.write(reinterpret_cast<const char*>(conv_actual.data()),
                      static_cast<std::streamsize>(conv_actual.size() * sizeof(float)));
        if (!capture.good()) return 1;
      }
      float conv_maximum = 0.0F;
      for (std::size_t index = 0; index < conv_actual.size(); ++index) {
        conv_maximum = std::max(
            conv_maximum,
            std::fabs(conv_actual[index] - expected_conv[segment * conv_actual.size() + index]));
      }
      conv_contract_maximum = std::max(conv_contract_maximum, conv_maximum);
    }
    if (!expected_core.empty()) {
      std::vector<float> core_actual(6144);
      if (!session.copy_from_device(binding(bindings, "core").id,
                                    {reinterpret_cast<std::byte*>(core_actual.data()),
                                     core_actual.size() * sizeof(float)}).ok()) {
        return 1;
      }
      float core_maximum = 0.0F;
      for (std::size_t index = 0; index < core_actual.size(); ++index) {
        core_maximum = std::max(
            core_maximum,
            std::fabs(core_actual[index] - expected_core[segment * core_actual.size() + index]));
      }
      core_contract_maximum = std::max(core_contract_maximum, core_maximum);
    }
    if (!expected_gated.empty()) {
      std::vector<float> gated_actual(6144);
      if (!session.copy_from_device(binding(bindings, "gated").id,
                                    {reinterpret_cast<std::byte*>(gated_actual.data()),
                                     gated_actual.size() * sizeof(float)}).ok()) {
        return 1;
      }
      float gated_maximum = 0.0F;
      for (std::size_t index = 0; index < gated_actual.size(); ++index) {
        gated_maximum = std::max(
            gated_maximum,
            std::fabs(gated_actual[index] - expected_gated[segment * gated_actual.size() + index]));
      }
      gated_contract_maximum = std::max(gated_contract_maximum, gated_maximum);
    }
    std::vector<float> actual_state(48U * 128U * 128U);
    if (!session.copy_from_device(binding(bindings, "delta_state").id,
                                  {reinterpret_cast<std::byte*>(actual_state.data()),
                                   actual_state.size() * sizeof(float)}).ok()) {
      return 1;
    }
    float state_maximum = 0.0F;
    for (std::size_t index = 0; index < actual_state.size(); ++index) {
      state_maximum = std::max(
          state_maximum,
          std::fabs(actual_state[index] - expected_state[segment * actual_state.size() + index]));
    }
    state_contract_maximum = std::max(state_contract_maximum, state_maximum);
  }
  float maximum = 0.0F;
  float mean = 0.0F;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float error = std::fabs(actual[index] - expected[index]);
    maximum = std::max(maximum, error);
    mean += error;
  }
  mean /= static_cast<float>(actual.size());
  if (const char* capture_path = std::getenv("SUPERINFER_QWEN38_GDN_OUTPUT_F32");
      capture_path != nullptr) {
    std::ofstream capture{capture_path, std::ios::binary | std::ios::trunc};
    if (!capture.good()) return 1;
    capture.write(reinterpret_cast<const char*>(actual.data()),
                  static_cast<std::streamsize>(actual.size() * sizeof(float)));
    if (!capture.good()) return 1;
  }
  std::cout << "qwen38 GDN layer0 artifact differential max_abs=" << maximum
            << " mean_abs=" << mean << " segments=" << segments << " commands="
            << session.trace().commands_executed << " qkv_max_abs=" << qkv_contract_maximum
            << " conv_max_abs=" << conv_contract_maximum
            << " core_max_abs=" << core_contract_maximum
            << " gated_max_abs=" << gated_contract_maximum
            << '\n';
  // The contract covers NVFP4 dequantization plus FP32 accumulation-order differences against
  // Transformers. State and mixer boundaries have tighter diagnostics than the final MLP output.
  return maximum <= 5.0e-2F && mean <= 5.0e-4F && attention_contract_maximum <= 1.0e-2F &&
                 state_contract_maximum <= 1.0e-3F &&
                 (expected_qkv.empty() || qkv_contract_maximum <= 5.0e-2F) &&
                 (expected_conv.empty() || conv_contract_maximum <= 5.0e-2F) &&
                 (expected_core.empty() || core_contract_maximum <= 5.0e-2F) &&
                 (expected_gated.empty() || gated_contract_maximum <= 5.0e-2F)
             ? 0
             : 1;
}
