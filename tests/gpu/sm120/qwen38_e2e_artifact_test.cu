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

  const char* layer_trace_path = std::getenv("SUPERINFER_QWEN38_LAYER_TRACE_F32");
  std::optional<std::uint32_t> layer_trace_step;
  std::vector<superinfer::ir::physical::CommandId> layer_trace_commands;
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
    std::size_t residual_index = 0;
    for (const auto& command : specialized.value().plan.commands()) {
      if (command.kernel.value() != 4) continue;
      // Kernel 4 is emitted once after attention and once after the MLP. Capture the latter.
      if ((residual_index++ % 2U) == 1U) layer_trace_commands.push_back(command.id);
    }
    if (layer_trace_commands.size() != 64) {
      std::cerr << "Qwen layer trace expected 64 post-MLP residual commands\n";
      return 1;
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
              ? session.execute_at_position_for_test(position, layer_trace_commands,
                                                      layer_trace_captures)
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
    if (!layer_trace_step.has_value() || layer_trace_captures.size() != layer_trace_commands.size()) {
      std::cerr << "layer trace step was not executed\n";
      return 1;
    }
    std::ofstream capture{layer_trace_path, std::ios::binary | std::ios::trunc};
    if (!capture.good()) return 1;
    for (const auto& bytes : layer_trace_captures) {
      if (bytes.size() != 5120U * sizeof(float)) {
        std::cerr << "layer trace output is not F32[5120]\n";
        return 1;
      }
      capture.write(reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
    }
    if (!capture.good()) return 1;
    std::ofstream metadata{std::string{layer_trace_path} + ".meta", std::ios::trunc};
    if (!metadata.good()) return 1;
    metadata << "step " << layer_trace_step.value() << " layers " << layer_trace_captures.size()
             << " contract post_mlp_residual\n";
    for (const auto command : layer_trace_commands) metadata << command.value() << '\n';
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
