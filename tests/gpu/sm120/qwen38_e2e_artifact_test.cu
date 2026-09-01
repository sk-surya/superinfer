#include <frontends/qwen38/frontend.hpp>
#include <sm120/compiler/specializer.h>
#include <sm120/kernels/baseline/provider.h>
#include <sm120/runtime/cuda_plan_executor.cuh>
#include <superinfer/artifact/plan_binding.hpp>
#include <superinfer/artifact/sinf.hpp>
#include <superinfer/artifact/tensor_table.hpp>
#include <superinfer/compiler/model_frontend.hpp>
#include <superinfer/compiler/semantic_lowering.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
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

std::uint32_t bf16_to_f32_bits(std::uint16_t value) noexcept {
  return static_cast<std::uint32_t>(value) << 16U;
}

float bf16_to_float(std::uint16_t value) noexcept {
  const std::uint32_t bits = bf16_to_f32_bits(value);
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

bool is_state_shape(const superinfer::ir::physical::BufferDescriptor& buffer) {
  using superinfer::ir::physical::PhysicalDType;
  if (buffer.tensor.dtype != PhysicalDType::bf16 && buffer.tensor.dtype != PhysicalDType::f32) {
    return false;
  }
  return buffer.tensor.shape == std::vector<std::uint64_t>({4096, 4, 256}) ||
         buffer.tensor.shape == std::vector<std::uint64_t>({48, 128, 128}) ||
         buffer.tensor.shape == std::vector<std::uint64_t>({4, 10240});
}

std::vector<std::uint32_t> parse_token_list(const char* encoded) {
  std::vector<std::uint32_t> tokens;
  if (encoded == nullptr) return tokens;
  const char* cursor = encoded;
  while (*cursor != '\0') {
    while (*cursor == ',' || *cursor == ' ') ++cursor;
    if (*cursor == '\0') break;
    const char* end = cursor;
    while (*end != '\0' && *end != ',') ++end;
    std::uint32_t token = 0;
    const auto parsed = std::from_chars(cursor, end, token);
    if (parsed.ec != std::errc{} || parsed.ptr != end) return {};
    tokens.push_back(token);
    cursor = end;
  }
  return tokens;
}

std::vector<std::pair<superinfer::ir::physical::CommandId,
                      superinfer::ir::physical::BufferId>>
parse_trace_requests(const char* encoded) {
  using superinfer::ir::physical::BufferId;
  using superinfer::ir::physical::CommandId;
  std::vector<std::pair<CommandId, BufferId>> requests;
  if (encoded == nullptr) return requests;
  const char* cursor = encoded;
  while (*cursor != '\0') {
    while (*cursor == ',' || *cursor == ' ') ++cursor;
    if (*cursor == '\0') break;
    const char* command_end = cursor;
    while (*command_end != '\0' && *command_end != ':') ++command_end;
    if (*command_end != ':') return {};
    std::uint32_t command = 0;
    const auto command_result = std::from_chars(cursor, command_end, command);
    if (command_result.ec != std::errc{} || command_result.ptr != command_end) return {};
    const char* buffer_begin = command_end + 1;
    const char* buffer_end = buffer_begin;
    while (*buffer_end != '\0' && *buffer_end != ',') ++buffer_end;
    std::uint32_t buffer = 0;
    const auto buffer_result = std::from_chars(buffer_begin, buffer_end, buffer);
    if (buffer_result.ec != std::errc{} || buffer_result.ptr != buffer_end) return {};
    requests.emplace_back(CommandId{command}, BufferId{buffer});
    cursor = buffer_end;
  }
  return requests;
}

int skip(const char* message) {
  std::cerr << "SKIP: " << message << '\n';
  return 77;
}

std::uint64_t fnv1a64(superinfer::base::ConstByteView bytes) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    hash ^= std::to_integer<std::uint8_t>(bytes[index]);
    hash *= 1099511628211ULL;
  }
  return hash;
}

}  // namespace

int main() {
  const char* artifact_path = std::getenv("SUPERINFER_QWEN38_ARTIFACT");
  if (artifact_path == nullptr) return skip("set SUPERINFER_QWEN38_ARTIFACT");

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return skip("no CUDA device");
  }
  if (cudaSetDevice(0) != cudaSuccess) return skip("cudaSetDevice(0) failed");
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess || properties.major != 12 ||
      properties.minor != 0) {
    return skip("GPU 0 is not sm_120a");
  }

  MappedFile file{artifact_path};
  if (!file.valid()) {
    std::cerr << "artifact mapping failed\n";
    return 1;
  }
  const auto artifact = superinfer::artifact::ArtifactReader::read(file.bytes());
  if (!artifact.has_value()) {
    std::cerr << "artifact read failed: " << artifact.error().message() << '\n';
    return 1;
  }
  const auto table_section = artifact.value().section(
      superinfer::artifact::SectionKind::tensor_table);
  if (!table_section.has_value()) return 1;
  const auto records = superinfer::artifact::parse_tensor_table(table_section.value());
  if (!records.has_value()) return 1;

  std::vector<superinfer::compiler::SourceTensorRecord> source_records;
  source_records.reserve(records.value().size());
  for (const auto& record : records.value()) {
    std::vector<std::uint64_t> shape = record.shape.empty()
                                           ? std::vector<std::uint64_t>{1}
                                           : record.shape;
    source_records.push_back({record.name, record.role, record.dtype, std::move(shape),
                              record.payload_offset,
                              record.payload_end - record.payload_offset});
  }
  superinfer::compiler::SourceInventory source{
      std::string{superinfer::frontends::qwen38::kSourceIdentity}, source_records.size(),
      std::string{superinfer::frontends::qwen38::kTensorInventorySha256},
      std::move(source_records)};
  superinfer::frontends::qwen38::Frontend frontend;
  std::vector<std::uint32_t> prefill_tokens{0};
  if (const char* encoded = std::getenv("SUPERINFER_QWEN38_PREFILL_TOKENS");
      encoded != nullptr) {
    prefill_tokens = parse_token_list(encoded);
    if (prefill_tokens.empty()) {
      std::cerr << "invalid SUPERINFER_QWEN38_PREFILL_TOKENS\n";
      return 1;
    }
  } else if (const char* initial_token = std::getenv("SUPERINFER_QWEN38_INITIAL_TOKEN");
             initial_token != nullptr) {
    const auto parsed = std::from_chars(initial_token, initial_token + std::strlen(initial_token),
                                        prefill_tokens.front());
    if (parsed.ec != std::errc{} || parsed.ptr != initial_token + std::strlen(initial_token)) {
      std::cerr << "invalid SUPERINFER_QWEN38_INITIAL_TOKEN\n";
      return 1;
    }
  }
  const auto frontend_status = frontend.validate(source);
  if (!frontend_status.ok()) {
    std::cerr << "frontend validation failed: " << frontend_status.message() << '\n';
    return 1;
  }
  const auto semantic = frontend.emit(source, static_cast<std::uint32_t>(prefill_tokens.size()));
  if (!semantic.has_value()) {
    std::cerr << "frontend emission failed: " << semantic.error().message() << '\n';
    return 1;
  }
  const auto lowered = superinfer::compiler::SemanticLowering{}.lower(
      semantic.value(), {120, 256, 4096});
  if (!lowered.has_value()) {
    std::cerr << "lowering failed: " << lowered.error().message() << '\n';
    return 1;
  }
  superinfer::sm120::BaselineProvider provider;
  const bool disable_activation_reuse =
      std::getenv("SUPERINFER_QWEN38_DISABLE_ACTIVATION_REUSE") != nullptr;
  const auto specialized = superinfer::sm120::Specializer{}.compile(
      lowered.value(),
      {superinfer::compiler::TargetProfile::offline_sm120a(32ULL << 30U, "baseline-v1"),
       0, 10000, disable_activation_reuse},
      provider);
  if (!specialized.has_value()) {
    std::cerr << "specialization failed: " << specialized.error().message() << '\n';
    return 1;
  }
  if (std::getenv("SUPERINFER_QWEN38_DUMP_PLAN_COMMANDS") != nullptr) {
    for (const auto& command : specialized.value().plan.commands()) {
      std::cout << "id=" << command.id.value() << " kernel=" << command.kernel.value()
                << " buffers=";
      for (std::size_t index = 0; index < command.buffers.size(); ++index) {
        if (index != 0) std::cout << ',';
        const auto buffer_id = command.buffers[index];
        const auto& buffer = specialized.value().plan.buffers()[buffer_id.value()];
        std::cout << buffer_id.value() << ':' << buffer.size << ':'
                  << static_cast<int>(buffer.tensor.dtype);
      }
      std::cout << '\n';
    }
    return 0;
  }
  if (const char* encoded_dump_layer = std::getenv("SUPERINFER_QWEN38_DUMP_LAYER_COMMANDS");
      encoded_dump_layer != nullptr) {
    std::uint32_t dump_layer = 0;
    const auto parsed = std::from_chars(
        encoded_dump_layer, encoded_dump_layer + std::strlen(encoded_dump_layer), dump_layer);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != encoded_dump_layer + std::strlen(encoded_dump_layer)) {
      std::cerr << "invalid SUPERINFER_QWEN38_DUMP_LAYER_COMMANDS\n";
      return 1;
    }
    std::size_t model_layer_index = 0;
    bool saw_token_mixer = false;
    for (const auto& command : specialized.value().plan.commands()) {
      const auto kernel = command.kernel.value();
      if (saw_token_mixer && model_layer_index == dump_layer) {
        std::cout << "layer=" << model_layer_index << " id=" << command.id.value()
                  << " kernel=" << kernel << " buffers=";
        for (const auto buffer : command.buffers) std::cout << buffer.value() << ',';
        std::cout << " output_bytes="
                  << specialized.value().plan.buffers()[command.buffers.back().value()].size
                  << '\n';
      }
      if (kernel == 15 || kernel == 23) {
        saw_token_mixer = true;
        ++model_layer_index;
        if (model_layer_index > dump_layer) break;
      }
    }
    return 0;
  }
  const auto resolved = superinfer::artifact::ArtifactPlanBinding::resolve(
      specialized.value().plan, artifact.value(), records.value());
  if (!resolved.has_value()) {
    std::cerr << "artifact binding failed: " << resolved.error().message() << '\n';
    return 1;
  }
  auto session_result = superinfer::sm120::cuda_runtime::CudaPlanSession::create(
      specialized.value().plan, 120, "baseline-v1");
  if (!session_result.has_value()) {
    std::cerr << "session creation failed: " << session_result.error().message() << '\n';
    for (const auto& context : session_result.error().context()) {
      std::cerr << "  context: " << context << '\n';
    }
    return 1;
  }
  auto session = std::move(session_result).value();

  if (const char* encoded_poison = std::getenv("SUPERINFER_QWEN38_ARENA_POISON");
      encoded_poison != nullptr) {
    char* end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(encoded_poison, &end, 16);
    if (errno != 0 || end == encoded_poison || *end != '\0' || parsed > 0xFFU) {
      std::cerr << "invalid SUPERINFER_QWEN38_ARENA_POISON\n";
      return 1;
    }
    const auto poison_status = session.fill_device_for_test(static_cast<std::uint8_t>(parsed));
    if (!poison_status.ok()) {
      std::cerr << "arena poison failed: " << poison_status.message() << '\n';
      return 1;
    }
  }

  bool capacity_rejected = false;
  if (std::getenv("SUPERINFER_QWEN38_CAPACITY_REJECTION") != nullptr) {
    const auto rejection = session.execute_at_position_for_test(4096);
    if (rejection.ok()) {
      std::cerr << "over-capacity continuation was accepted\n";
      return 1;
    }
    capacity_rejected = true;
  }

  for (const auto& binding : resolved.value()) {
    const auto status = session.copy_to_device(binding.buffer, binding.payload);
    if (!status.ok()) {
      std::cerr << "artifact upload failed for " << binding.artifact_name << ": "
                << status.message() << '\n';
      return 1;
    }
  }
  std::vector<std::byte> zeros;
  std::size_t state_buffers = 0;
  for (const auto& buffer : specialized.value().plan.buffers()) {
    if (!buffer.tensor.artifact_name.empty() || !is_state_shape(buffer)) continue;
    zeros.assign(buffer.size, std::byte{0});
    const auto status = session.copy_to_device(
        buffer.id, {zeros.data(), zeros.size()});
    if (!status.ok()) {
      std::cerr << "state initialization failed: " << status.message() << '\n';
      return 1;
    }
    ++state_buffers;
  }
  if (specialized.value().plan.entry_points().size() != 1 ||
      specialized.value().plan.entry_points().front().inputs.size() != prefill_tokens.size() ||
      specialized.value().plan.entry_points().front().outputs.size() != prefill_tokens.size()) {
    std::cerr << "Qwen entry point does not match the requested token sequence contract\n";
    return 1;
  }
  const auto entry = specialized.value().plan.entry_points().front();
  for (std::size_t index = 0; index < prefill_tokens.size(); ++index) {
    if (!session.copy_to_device(
            entry.inputs[index],
            {reinterpret_cast<const std::byte*>(&prefill_tokens[index]), sizeof(std::uint32_t)}).ok()) {
      std::cerr << "token upload failed\n";
      return 1;
    }
  }

  const auto launch_status = session.execute();
  if (!launch_status.ok()) {
    std::cerr << "Qwen full graph launch failed: " << launch_status.message() << '\n';
    return 1;
  }
  const auto sync_status = session.synchronize_for_test();
  if (!sync_status.ok()) {
    std::cerr << "Qwen full graph synchronization failed: " << sync_status.message() << '\n';
    return 1;
  }

  const char* hidden_capture_path = std::getenv("SUPERINFER_QWEN38_HIDDEN_F32");
  std::optional<std::uint32_t> hidden_capture_step;
  if (const char* encoded_step = std::getenv("SUPERINFER_QWEN38_HIDDEN_STEP");
      encoded_step != nullptr) {
    std::uint32_t parsed_step = 0;
    const auto parsed = std::from_chars(encoded_step,
                                        encoded_step + std::strlen(encoded_step), parsed_step);
    if (parsed.ec != std::errc{} || parsed.ptr != encoded_step + std::strlen(encoded_step)) {
      std::cerr << "invalid SUPERINFER_QWEN38_HIDDEN_STEP\n";
      return 1;
    }
    hidden_capture_step = parsed_step;
    if (hidden_capture_path == nullptr) {
      std::cerr << "SUPERINFER_QWEN38_HIDDEN_STEP requires SUPERINFER_QWEN38_HIDDEN_F32\n";
      return 1;
    }
  }
  std::optional<superinfer::ir::physical::BufferId> hidden_buffer_id;
  std::size_t hidden_buffer_size = 0;
  if (hidden_capture_path != nullptr) {
    const auto& commands = specialized.value().plan.commands();
    const auto lm_head = std::find_if(
        commands.rbegin(), commands.rend(), [](const auto& command) {
          return command.kernel.value() == 13 && !command.buffers.empty();
        });
    if (lm_head == commands.rend()) {
      std::cerr << "Qwen LM-head command was not found for hidden capture\n";
      return 1;
    }
    const auto& hidden_buffer = specialized.value().plan.buffers()[lm_head->buffers.front().value()];
    if (hidden_buffer.tensor.dtype != superinfer::ir::physical::PhysicalDType::f32 ||
        hidden_buffer.size != 5120U * sizeof(float)) {
      std::cerr << "Qwen final hidden buffer is not F32[5120]\n";
      return 1;
    }
    hidden_buffer_id = hidden_buffer.id;
    hidden_buffer_size = hidden_buffer.size;
  }
  const auto capture_hidden = [&](std::uint32_t step) -> bool {
    if (hidden_capture_path == nullptr) return true;
    std::vector<float> hidden(hidden_buffer_size / sizeof(float));
    const auto hidden_status = session.copy_from_device(
        hidden_buffer_id.value(), {reinterpret_cast<std::byte*>(hidden.data()), hidden_buffer_size});
    if (!hidden_status.ok()) {
      std::cerr << "hidden capture download failed at step " << step << ": "
                << hidden_status.message() << '\n';
      return false;
    }
    std::ofstream capture{hidden_capture_path, std::ios::binary | std::ios::trunc};
    if (!capture.good()) return false;
    capture.write(reinterpret_cast<const char*>(hidden.data()),
                  static_cast<std::streamsize>(hidden_buffer_size));
    return capture.good();
  };
  if (hidden_capture_step.has_value() && hidden_capture_step.value() == 0 &&
      !capture_hidden(0)) return 1;

  const char* kv_capture_path = std::getenv("SUPERINFER_QWEN38_KV_F32");
  std::optional<std::uint32_t> kv_capture_step;
  std::optional<std::uint32_t> kv_capture_layer;
  std::optional<superinfer::ir::physical::BufferId> kv_key_buffer_id;
  std::optional<superinfer::ir::physical::BufferId> kv_value_buffer_id;
  if (kv_capture_path != nullptr) {
    const char* encoded_step = std::getenv("SUPERINFER_QWEN38_KV_STEP");
    const char* encoded_layer = std::getenv("SUPERINFER_QWEN38_KV_LAYER");
    if (encoded_step == nullptr || encoded_layer == nullptr) {
      std::cerr << "SUPERINFER_QWEN38_KV_F32 requires SUPERINFER_QWEN38_KV_STEP and "
                   "SUPERINFER_QWEN38_KV_LAYER\n";
      return 1;
    }
    std::uint32_t parsed_step = 0;
    const auto step_result = std::from_chars(
        encoded_step, encoded_step + std::strlen(encoded_step), parsed_step);
    std::uint32_t parsed_layer = 0;
    const auto layer_result = std::from_chars(
        encoded_layer, encoded_layer + std::strlen(encoded_layer), parsed_layer);
    if (step_result.ec != std::errc{} ||
        step_result.ptr != encoded_step + std::strlen(encoded_step) ||
        layer_result.ec != std::errc{} ||
        layer_result.ptr != encoded_layer + std::strlen(encoded_layer)) {
      std::cerr << "invalid Qwen KV capture step or layer\n";
      return 1;
    }
    if (parsed_layer >= 64U || parsed_layer < 3U || parsed_layer % 4U != 3U) {
      std::cerr << "Qwen KV capture layer must be one of 3,7,...,63\n";
      return 1;
    }
    const std::size_t requested_full_attention_index = (parsed_layer - 3U) / 4U;
    std::size_t full_attention_index = 0;
    for (const auto& command : specialized.value().plan.commands()) {
      if (command.kernel.value() != 22) continue;
      if (full_attention_index == requested_full_attention_index) {
        if (command.buffers.size() != 4) {
          std::cerr << "Qwen KV cache append command has an unexpected operand count\n";
          return 1;
        }
        kv_key_buffer_id = command.buffers[2];
        kv_value_buffer_id = command.buffers[3];
        break;
      }
      ++full_attention_index;
    }
    if (!kv_key_buffer_id.has_value() || !kv_value_buffer_id.has_value()) {
      std::cerr << "requested Qwen KV cache append command was not found\n";
      return 1;
    }
    kv_capture_step = parsed_step;
    kv_capture_layer = parsed_layer;
  }
  const auto capture_kv = [&](std::uint32_t step) -> bool {
    if (kv_capture_path == nullptr) return true;
    if (!kv_capture_step.has_value() || step != kv_capture_step.value()) return true;
    const auto& key_buffer = specialized.value().plan.buffers()[kv_key_buffer_id.value().value()];
    const auto& value_buffer = specialized.value().plan.buffers()[kv_value_buffer_id.value().value()];
    constexpr std::size_t kHeads = 4;
    constexpr std::size_t kHeadDimension = 256;
    const std::size_t elements = (static_cast<std::size_t>(step) + 1U) * kHeads * kHeadDimension;
    const std::size_t bytes = elements * sizeof(std::uint16_t);
    if (key_buffer.tensor.dtype != superinfer::ir::physical::PhysicalDType::bf16 ||
        value_buffer.tensor.dtype != superinfer::ir::physical::PhysicalDType::bf16 ||
        bytes > key_buffer.size || bytes > value_buffer.size) {
      std::cerr << "Qwen KV capture buffer contract mismatch\n";
      return false;
    }
    std::vector<std::uint16_t> key_storage(elements);
    std::vector<std::uint16_t> value_storage(elements);
    if (!session.copy_from_device(kv_key_buffer_id.value(),
                                  {reinterpret_cast<std::byte*>(key_storage.data()), bytes}).ok() ||
        !session.copy_from_device(kv_value_buffer_id.value(),
                                  {reinterpret_cast<std::byte*>(value_storage.data()), bytes}).ok()) {
      std::cerr << "Qwen KV capture download failed\n";
      return false;
    }
    std::vector<float> payload;
    payload.reserve(elements * 2U);
    for (const std::uint16_t value : key_storage) payload.push_back(bf16_to_float(value));
    for (const std::uint16_t value : value_storage) payload.push_back(bf16_to_float(value));
    std::ofstream capture{kv_capture_path, std::ios::binary | std::ios::trunc};
    if (!capture.good()) return false;
    capture.write(reinterpret_cast<const char*>(payload.data()),
                  static_cast<std::streamsize>(payload.size() * sizeof(float)));
    if (!capture.good()) return false;
    std::ofstream metadata{std::string{kv_capture_path} + ".json", std::ios::trunc};
    if (!metadata.good()) return false;
    metadata << "{\n"
             << "  \"case\": \"target-execution\",\n"
             << "  \"layer\": " << kv_capture_layer.value() << ",\n"
             << "  \"step\": " << step << ",\n"
             << "  \"positions\": " << (step + 1U) << ",\n"
             << "  \"heads\": 4,\n"
             << "  \"head_dimension\": 256,\n"
             << "  \"dtype\": \"float32\",\n"
             << "  \"layout\": \"[positions, heads, head_dimension] key then value\",\n"
             << "  \"source_dtype\": \"bf16\",\n"
             << "  \"bytes\": " << payload.size() * sizeof(float) << ",\n"
             << "  \"fnv1a64\": "
             << fnv1a64({reinterpret_cast<const std::byte*>(payload.data()),
                        payload.size() * sizeof(float)}) << "\n}\n";
    return metadata.good();
  };
  if (kv_capture_step.has_value() && kv_capture_step.value() == 0 && !capture_kv(0)) return 1;

  const char* gdn_state_capture_path = std::getenv("SUPERINFER_QWEN38_GDN_STATE_F32");
  std::uint32_t gdn_state_capture_layer = 0;
  std::optional<std::uint32_t> gdn_state_capture_step;
  std::optional<superinfer::ir::physical::BufferId> gdn_delta_state_id;
  std::optional<superinfer::ir::physical::BufferId> gdn_conv_state_id;
  if (gdn_state_capture_path != nullptr) {
    const char* encoded_step = std::getenv("SUPERINFER_QWEN38_GDN_STATE_STEP");
    if (encoded_step == nullptr) {
      std::cerr << "SUPERINFER_QWEN38_GDN_STATE_F32 requires SUPERINFER_QWEN38_GDN_STATE_STEP\n";
      return 1;
    }
    std::uint32_t parsed_step = 0;
    const auto step_result = std::from_chars(
        encoded_step, encoded_step + std::strlen(encoded_step), parsed_step);
    if (step_result.ec != std::errc{} ||
        step_result.ptr != encoded_step + std::strlen(encoded_step)) {
      std::cerr << "invalid SUPERINFER_QWEN38_GDN_STATE_STEP\n";
      return 1;
    }
    if (const char* encoded_layer = std::getenv("SUPERINFER_QWEN38_GDN_STATE_LAYER");
        encoded_layer != nullptr) {
      const auto layer_result = std::from_chars(
          encoded_layer, encoded_layer + std::strlen(encoded_layer), gdn_state_capture_layer);
      if (layer_result.ec != std::errc{} ||
          layer_result.ptr != encoded_layer + std::strlen(encoded_layer)) {
        std::cerr << "invalid SUPERINFER_QWEN38_GDN_STATE_LAYER\n";
        return 1;
      }
    }
    std::size_t model_layer_index = 0;
    for (const auto& command : specialized.value().plan.commands()) {
      const auto kernel = command.kernel.value();
      if (kernel != 15 && kernel != 23) continue;
      if (kernel == 15 && model_layer_index == gdn_state_capture_layer) {
        if (command.buffers.size() < 7) {
          std::cerr << "selected GDN state command lacks state operand\n";
          return 1;
        }
        gdn_delta_state_id = command.buffers[5];
        break;
      }
      ++model_layer_index;
    }
    if (!gdn_delta_state_id.has_value()) {
      std::cerr << "selected GDN state layer was not found\n";
      return 1;
    }
    model_layer_index = 0;
    for (const auto& command : specialized.value().plan.commands()) {
      const auto kernel = command.kernel.value();
      if (kernel != 15 && kernel != 23 && kernel != 25) continue;
      if (kernel == 25 && model_layer_index == gdn_state_capture_layer) {
        if (command.buffers.size() < 4) {
          std::cerr << "selected GDN convolution command lacks state operand\n";
          return 1;
        }
        gdn_conv_state_id = command.buffers[2];
        break;
      }
      if (kernel == 15 || kernel == 23) ++model_layer_index;
    }
    if (!gdn_conv_state_id.has_value()) {
      std::cerr << "selected GDN convolution layer was not found\n";
      return 1;
    }
    gdn_state_capture_step = parsed_step;
  }
  const auto capture_gdn_state = [&](std::uint32_t step) -> bool {
    if (gdn_state_capture_path == nullptr || !gdn_state_capture_step.has_value() ||
        step != gdn_state_capture_step.value()) return true;
    const auto& delta_state_buffer =
        specialized.value().plan.buffers()[gdn_delta_state_id.value().value()];
    const auto& conv_state_buffer =
        specialized.value().plan.buffers()[gdn_conv_state_id.value().value()];
    const std::size_t delta_elements = delta_state_buffer.size / sizeof(float);
    const std::size_t conv_elements = conv_state_buffer.size / sizeof(std::uint16_t);
    std::vector<float> delta(delta_elements);
    std::vector<std::uint16_t> conv(conv_elements);
    if (delta.empty()) {
      std::cerr << "selected GDN delta state is empty\n";
      return false;
    }
    if (!session.copy_from_device(
            gdn_delta_state_id.value(),
            {reinterpret_cast<std::byte*>(delta.data()), delta.size() * sizeof(float)}).ok() ||
        !session.copy_from_device(
            gdn_conv_state_id.value(),
            {reinterpret_cast<std::byte*>(conv.data()), conv.size() * sizeof(std::uint16_t)}).ok()) {
      std::cerr << "selected GDN state capture download failed\n";
      return false;
    }
    std::vector<float> payload = std::move(delta);
    payload.reserve(payload.size() + conv.size());
    for (const std::uint16_t value : conv) payload.push_back(bf16_to_float(value));
    std::ofstream capture{gdn_state_capture_path, std::ios::binary | std::ios::trunc};
    if (!capture.good()) return false;
    capture.write(reinterpret_cast<const char*>(payload.data()),
                  static_cast<std::streamsize>(payload.size() * sizeof(float)));
    if (!capture.good()) return false;
    std::ofstream metadata{std::string{gdn_state_capture_path} + ".json", std::ios::trunc};
    if (!metadata.good()) return false;
    metadata << "{\n"
             << "  \"layer\": " << gdn_state_capture_layer << ",\n"
             << "  \"step\": " << step << ",\n"
             << "  \"delta_buffer_id\": " << gdn_delta_state_id.value().value() << ",\n"
             << "  \"delta_buffer_size\": " << delta_state_buffer.size << ",\n"
             << "  \"conv_buffer_id\": " << gdn_conv_state_id.value().value() << ",\n"
             << "  \"conv_buffer_size\": " << conv_state_buffer.size << ",\n"
             << "  \"delta_elements\": " << payload.size() - conv.size() << ",\n"
             << "  \"convolution_elements\": " << conv.size() << ",\n"
             << "  \"delta_dtype\": \"float32\",\n"
             << "  \"conv_source_dtype\": \"bf16\",\n"
             << "  \"layout\": \"delta then convolution state\"\n}\n";
    return metadata.good();
  };
  if (gdn_state_capture_step.has_value() && gdn_state_capture_step.value() == 0 &&
      !capture_gdn_state(0)) return 1;

  const char* layer_trace_path = std::getenv("SUPERINFER_QWEN38_LAYER_TRACE_F32");
  const char* layer_trace_stage = std::getenv("SUPERINFER_QWEN38_LAYER_TRACE_STAGE");
  const char* command_trace_ids = std::getenv("SUPERINFER_QWEN38_COMMAND_TRACE_IDS");
  const char* command_trace_outputs = std::getenv("SUPERINFER_QWEN38_COMMAND_TRACE_OUTPUTS");
  std::optional<std::uint32_t> layer_trace_step;
  std::vector<superinfer::ir::physical::CommandId> layer_trace_commands;
  std::vector<std::pair<superinfer::ir::physical::CommandId,
                        superinfer::ir::physical::BufferId>> layer_trace_requests;
  std::vector<std::vector<std::byte>> layer_trace_captures;
  if (layer_trace_path != nullptr) {
    const char* encoded_step = std::getenv("SUPERINFER_QWEN38_LAYER_TRACE_STEP");
    if (encoded_step == nullptr) {
      std::cerr << "SUPERINFER_QWEN38_LAYER_TRACE_F32 requires SUPERINFER_QWEN38_LAYER_TRACE_STEP\n";
      return 1;
    }
    std::uint32_t parsed_step = 0;
    const auto parsed = std::from_chars(encoded_step,
                                        encoded_step + std::strlen(encoded_step), parsed_step);
    if (parsed.ec != std::errc{} || parsed.ptr != encoded_step + std::strlen(encoded_step)) {
      std::cerr << "invalid SUPERINFER_QWEN38_LAYER_TRACE_STEP\n";
      return 1;
    }
    layer_trace_step = parsed_step;
    if (command_trace_outputs != nullptr) {
      layer_trace_requests = parse_trace_requests(command_trace_outputs);
      if (layer_trace_requests.empty()) {
        std::cerr << "invalid SUPERINFER_QWEN38_COMMAND_TRACE_OUTPUTS\n";
        return 1;
      }
    } else if (command_trace_ids != nullptr) {
      const auto encoded_ids = parse_token_list(command_trace_ids);
      if (encoded_ids.empty()) {
        std::cerr << "invalid SUPERINFER_QWEN38_COMMAND_TRACE_IDS\n";
        return 1;
      }
      for (const auto id : encoded_ids) layer_trace_commands.emplace_back(id);
    } else {
      const bool post_attention = layer_trace_stage != nullptr &&
                                  std::string_view{layer_trace_stage} == "post_attention";
      if (layer_trace_stage != nullptr && !post_attention &&
          std::string_view{layer_trace_stage} != "post_mlp") {
        std::cerr << "SUPERINFER_QWEN38_LAYER_TRACE_STAGE must be post_attention or post_mlp\n";
        return 1;
      }
      std::size_t residual_index = 0;
      for (const auto& command : specialized.value().plan.commands()) {
        if (command.kernel.value() != 4) continue;
        // Kernel 4 is emitted once after the token mixer and once after the MLP.
        const bool select_post_attention = (residual_index++ % 2U) == 0U;
        if (select_post_attention == post_attention) layer_trace_commands.push_back(command.id);
      }
      if (layer_trace_commands.size() != 64) {
        std::cerr << "Qwen layer trace expected 64 post-MLP residual commands\n";
        return 1;
      }
    }
  }

  std::ofstream state_trace;
  std::vector<superinfer::ir::physical::BufferId> state_trace_ids;
  std::vector<std::vector<std::byte>> state_trace_storage;
  if (const char* state_trace_path = std::getenv("SUPERINFER_QWEN38_STATE_TRACE");
      state_trace_path != nullptr) {
    state_trace.open(state_trace_path, std::ios::trunc);
    if (!state_trace.good()) {
      std::cerr << "state trace open failed: " << state_trace_path << '\n';
      return 1;
    }
    for (const auto& buffer : specialized.value().plan.buffers()) {
      if (!buffer.tensor.artifact_name.empty() || !is_state_shape(buffer)) continue;
      state_trace_ids.push_back(buffer.id);
      state_trace_storage.emplace_back(buffer.size);
    }
    state_trace << "step buffer_id fnv1a64\n";
  }
  const auto capture_state_trace = [&](std::uint32_t step) -> bool {
    if (!state_trace.is_open()) return true;
    for (std::size_t index = 0; index < state_trace_ids.size(); ++index) {
      const auto id = state_trace_ids[index];
      auto& storage = state_trace_storage[index];
      const auto status = session.copy_from_device(id, {storage.data(), storage.size()});
      if (!status.ok()) {
        std::cerr << "state trace download failed: " << status.message() << '\n';
        return false;
      }
      state_trace << step << ' ' << id.value() << ' '
                  << fnv1a64({storage.data(), storage.size()}) << '\n';
    }
    return state_trace.good();
  };
  if (!capture_state_trace(0)) return 1;
  bool boundary_executed = false;
  if (std::getenv("SUPERINFER_QWEN38_BOUNDARY_EXECUTION") != nullptr) {
    const std::uint64_t allocations_before = session.lifecycle_trace().device_allocations;
    const auto boundary_status = session.execute_at_position_for_test(4095);
    if (!boundary_status.ok()) {
      std::cerr << "legal boundary continuation failed: " << boundary_status.message() << '\n';
      return 1;
    }
    const auto boundary_sync = session.synchronize_for_test();
    if (!boundary_sync.ok()) {
      std::cerr << "legal boundary synchronization failed: " << boundary_sync.message() << '\n';
      return 1;
    }
    if (session.lifecycle_trace().device_allocations != allocations_before) {
      std::cerr << "legal boundary continuation grew device allocations\n";
      return 1;
    }
    boundary_executed = true;
  }
  const std::uint32_t token = prefill_tokens.front();
  const auto& output = specialized.value().plan.buffers()[entry.outputs.front().value()];
  if (output.tensor.dtype != superinfer::ir::physical::PhysicalDType::bf16 ||
      output.size != 248320U * sizeof(std::uint16_t)) {
    std::cerr << "Qwen logits buffer is not BF16[vocab]\n";
    return 1;
  }
  std::vector<std::uint16_t> logits(output.size / sizeof(std::uint16_t));
  const auto copy_status = session.copy_from_device(
      entry.outputs.front(),
      {reinterpret_cast<std::byte*>(logits.data()), output.size});
  if (!copy_status.ok()) {
    std::cerr << "logits download failed: " << copy_status.message() << '\n';
    return 1;
  }
  std::size_t best = 0;
  float best_value = bf16_to_float(logits.front());
  float checksum = 0.0F;
  for (std::size_t index = 0; index < logits.size(); ++index) {
    const float value = bf16_to_float(logits[index]);
    if (!std::isfinite(value)) {
      std::cerr << "Qwen logits contain non-finite value at " << index << '\n';
      return 1;
    }
    checksum += value;
    if (value > best_value) {
      best = index;
      best_value = value;
    }
  }
  const char* capture_path = std::getenv("SUPERINFER_QWEN38_LOGITS_F32");
  if (capture_path != nullptr) {
    std::ofstream capture{capture_path, std::ios::binary | std::ios::trunc};
    if (!capture.good()) return 1;
    for (const std::uint16_t value : logits) {
      const float converted = bf16_to_float(value);
      capture.write(reinterpret_cast<const char*>(&converted), sizeof(converted));
    }
    if (!capture.good()) return 1;
  }
  std::vector<std::size_t> prefill_greedy{best};
  std::vector<float> prefill_best_values{best_value};
  if (prefill_tokens.size() > 1) {
    const char* prefill_capture_path = std::getenv("SUPERINFER_QWEN38_PREFILL_LOGITS_F32");
    std::ofstream prefill_capture;
    if (prefill_capture_path != nullptr) {
      prefill_capture.open(prefill_capture_path, std::ios::binary | std::ios::trunc);
      if (!prefill_capture.good()) {
        std::cerr << "prefill capture open failed errno=" << errno << '\n';
        return 1;
      }
    }
    if (prefill_capture.is_open()) {
      std::vector<float> prefill_row;
      prefill_row.reserve(logits.size());
      for (const std::uint16_t value : logits) {
        prefill_row.push_back(bf16_to_float(value));
      }
      prefill_capture.write(reinterpret_cast<const char*>(prefill_row.data()),
                            static_cast<std::streamsize>(prefill_row.size() * sizeof(float)));
    }
    for (std::size_t step = 1; step < entry.outputs.size(); ++step) {
      const auto& prefill_output = specialized.value().plan.buffers()[entry.outputs[step].value()];
      if (prefill_output.tensor.dtype != superinfer::ir::physical::PhysicalDType::bf16 ||
          prefill_output.size != output.size) {
        std::cerr << "Qwen prefill logits buffer contract mismatch\n";
        return 1;
      }
      std::vector<std::uint16_t> prefill_logits(logits.size());
      const auto prefill_copy = session.copy_from_device(
          entry.outputs[step], {reinterpret_cast<std::byte*>(prefill_logits.data()), output.size});
      if (!prefill_copy.ok()) {
        std::cerr << "prefill logits download failed: " << prefill_copy.message() << '\n';
        return 1;
      }
      std::size_t prefill_best = 0;
      float prefill_best_value = bf16_to_float(prefill_logits.front());
      std::vector<float> prefill_row(prefill_logits.size());
      for (std::size_t index = 0; index < prefill_logits.size(); ++index) {
        const float value = bf16_to_float(prefill_logits[index]);
        if (!std::isfinite(value)) {
          std::cerr << "prefill logits contain non-finite value at " << index << '\n';
          return 1;
        }
        if (value > prefill_best_value) {
          prefill_best = index;
          prefill_best_value = value;
        }
        prefill_row[index] = value;
      }
      if (prefill_capture.is_open()) {
        prefill_capture.write(reinterpret_cast<const char*>(prefill_row.data()),
                              static_cast<std::streamsize>(prefill_row.size() * sizeof(float)));
      }
      prefill_greedy.push_back(prefill_best);
      prefill_best_values.push_back(prefill_best_value);
    }
    if (prefill_capture.is_open() && !prefill_capture.good()) return 1;
  }
  std::vector<std::size_t> continuation_greedy;
  std::vector<float> continuation_best_values;
  const char* continuation_capture_path =
      std::getenv("SUPERINFER_QWEN38_CONTINUATION_LOGITS_F32");
  if (std::getenv("SUPERINFER_QWEN38_CONTINUATION") != nullptr) {
    std::vector<std::uint32_t> continuation_tokens = parse_token_list(
        std::getenv("SUPERINFER_QWEN38_CONTINUATION_TOKENS"));
    if (continuation_tokens.empty()) continuation_tokens.push_back(static_cast<std::uint32_t>(best));
    std::ofstream continuation_capture;
    if (continuation_capture_path != nullptr) {
      continuation_capture.open(continuation_capture_path, std::ios::binary | std::ios::trunc);
      if (!continuation_capture.good()) return 1;
    }
    for (std::size_t step = 0; step < continuation_tokens.size(); ++step) {
      const std::uint32_t continuation_token = continuation_tokens[step];
      const auto token_status = session.copy_to_device(
          entry.inputs.front(), {reinterpret_cast<const std::byte*>(&continuation_token),
                                 sizeof(continuation_token)});
      if (!token_status.ok()) {
        std::cerr << "continuation token upload failed: " << token_status.message() << '\n';
        return 1;
      }
      const std::uint32_t position = static_cast<std::uint32_t>(step + 1);
      const auto continuation_status =
          layer_trace_step.has_value() && layer_trace_step.value() == position
              ? (layer_trace_requests.empty()
                     ? session.execute_at_position_for_test(position, layer_trace_commands,
                                                            layer_trace_captures)
                     : session.execute_at_position_for_test(position, layer_trace_requests,
                                                            layer_trace_captures))
              : session.execute_at_position_for_test(position);
      if (!continuation_status.ok()) {
        std::cerr << "Qwen continuation launch failed: " << continuation_status.message() << '\n';
        return 1;
      }
      const auto continuation_sync = session.synchronize_for_test();
      if (!continuation_sync.ok()) {
        std::cerr << "Qwen continuation synchronization failed: "
                  << continuation_sync.message() << '\n';
        return 1;
      }
      if (!capture_state_trace(static_cast<std::uint32_t>(step + 1))) return 1;
      if (hidden_capture_step.has_value() && hidden_capture_step.value() == step + 1U &&
          !capture_hidden(step + 1U)) return 1;
      if (!capture_kv(static_cast<std::uint32_t>(step + 1U))) return 1;
      if (!capture_gdn_state(static_cast<std::uint32_t>(step + 1U))) return 1;
      std::vector<std::uint16_t> continuation_logits(logits.size());
      const auto continuation_copy = session.copy_from_device(
          entry.outputs.front(), {reinterpret_cast<std::byte*>(continuation_logits.data()),
                                  output.size});
      if (!continuation_copy.ok()) {
        std::cerr << "continuation logits download failed: " << continuation_copy.message() << '\n';
        return 1;
      }
      std::size_t continuation_best = 0;
      float continuation_best_value = bf16_to_float(continuation_logits.front());
      for (std::size_t index = 0; index < continuation_logits.size(); ++index) {
        const float value = bf16_to_float(continuation_logits[index]);
        if (!std::isfinite(value)) {
          std::cerr << "continuation logits contain non-finite value at " << index << '\n';
          return 1;
        }
        if (value > continuation_best_value) {
          continuation_best = index;
          continuation_best_value = value;
        }
      }
      continuation_greedy.push_back(continuation_best);
      continuation_best_values.push_back(continuation_best_value);
      if (continuation_capture.is_open()) {
        for (const std::uint16_t value : continuation_logits) {
          const float converted = bf16_to_float(value);
          continuation_capture.write(reinterpret_cast<const char*>(&converted), sizeof(converted));
        }
        if (!continuation_capture.good()) return 1;
      }
    }
  }
  if (hidden_capture_path != nullptr && !hidden_capture_step.has_value() &&
      !capture_hidden(static_cast<std::uint32_t>(prefill_tokens.size() - 1U +
                                                  continuation_greedy.size()))) return 1;
  if (hidden_capture_step.has_value()) {
    const std::uint32_t completed_steps = static_cast<std::uint32_t>(
        prefill_tokens.size() == 1 ? continuation_greedy.size() + 1U : prefill_tokens.size());
    if (hidden_capture_step.value() >= completed_steps) {
      std::cerr << "hidden capture step was not executed: " << hidden_capture_step.value() << '\n';
      return 1;
    }
  }
  if (layer_trace_path != nullptr) {
    const std::size_t expected_captures =
        layer_trace_requests.empty() ? layer_trace_commands.size() : layer_trace_requests.size();
    if (!layer_trace_step.has_value() || layer_trace_captures.size() != expected_captures) {
      std::cerr << "layer trace step was not executed\n";
      return 1;
    }
    std::ofstream capture{layer_trace_path, std::ios::binary | std::ios::trunc};
    if (!capture.good()) return 1;
    for (const auto& bytes : layer_trace_captures) {
      if (command_trace_ids == nullptr && command_trace_outputs == nullptr &&
          bytes.size() != 5120U * sizeof(float)) {
        std::cerr << "layer trace output is not F32[5120]\n";
        return 1;
      }
      capture.write(reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
    }
    if (!capture.good()) return 1;
    std::ofstream metadata{std::string{layer_trace_path} + ".meta", std::ios::trunc};
    if (!metadata.good()) return 1;
    metadata << "step " << layer_trace_step.value() << " captures " << layer_trace_captures.size()
             << " contract " << ((layer_trace_stage != nullptr &&
                                    std::string_view{layer_trace_stage} == "post_attention")
                                       ? "post_token_mixer_residual"
                                       : (command_trace_ids != nullptr || command_trace_outputs != nullptr
                                              ? "command_outputs"
                                                                         : "post_mlp_residual"))
             << "\n";
    if (layer_trace_requests.empty()) {
      for (const auto command : layer_trace_commands) metadata << command.value() << '\n';
    } else {
      for (const auto& [command, buffer] : layer_trace_requests) {
        metadata << command.value() << ':' << buffer.value() << '\n';
      }
    }
    if (!metadata.good()) return 1;
  }
  std::cout << "qwen38 e2e token=" << token << " greedy=" << best << " logit=" << best_value
            << " checksum=" << checksum << " state_buffers=" << state_buffers
            << " commands=" << session.trace().commands_executed
            << " arena_bytes=" << session.device_arena_bytes();
  if (prefill_tokens.size() > 1) {
    std::cout << " prefill_steps=" << prefill_greedy.size();
    for (std::size_t step = 1; step < prefill_greedy.size(); ++step) {
      std::cout << " prefill_token=" << step << " greedy=" << prefill_greedy[step]
                << " logit=" << prefill_best_values[step];
    }
  }
  if (std::getenv("SUPERINFER_QWEN38_CONTINUATION") != nullptr) {
    std::cout << " continuation_steps=" << continuation_greedy.size();
    for (std::size_t step = 0; step < continuation_greedy.size(); ++step) {
      std::cout << " token=" << step + 1 << " greedy=" << continuation_greedy[step]
                << " logit=" << continuation_best_values[step];
    }
  }
  if (capacity_rejected) std::cout << " capacity_rejection=pass";
  if (boundary_executed) std::cout << " boundary_position=4095 boundary_allocations=stable";
  std::cout << '\n';
  return 0;
}
