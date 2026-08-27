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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
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

int skip(const char* message) {
  std::cerr << "SKIP: " << message << '\n';
  return 77;
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
  const auto frontend_status = frontend.validate(source);
  if (!frontend_status.ok()) {
    std::cerr << "frontend validation failed: " << frontend_status.message() << '\n';
    return 1;
  }
  const auto semantic = frontend.emit(source);
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
  const auto specialized = superinfer::sm120::Specializer{}.compile(
      lowered.value(),
      {superinfer::compiler::TargetProfile::offline_sm120a(32ULL << 30U, "baseline-v1"),
       0, 10000},
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
      specialized.value().plan.entry_points().front().inputs.size() != 1 ||
      specialized.value().plan.entry_points().front().outputs.size() != 1) {
    std::cerr << "Qwen decode entry point is not a single-token/single-logits contract\n";
    return 1;
  }
  const auto entry = specialized.value().plan.entry_points().front();
  const std::uint32_t token = 0;
  if (!session.copy_to_device(
          entry.inputs.front(),
          {reinterpret_cast<const std::byte*>(&token), sizeof(token)}).ok()) {
    std::cerr << "token upload failed\n";
    return 1;
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
  const char* hidden_capture_path = std::getenv("SUPERINFER_QWEN38_HIDDEN_F32");
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
    std::vector<float> hidden(hidden_buffer.size / sizeof(float));
    const auto hidden_status = session.copy_from_device(
        hidden_buffer.id, {reinterpret_cast<std::byte*>(hidden.data()), hidden_buffer.size});
    if (!hidden_status.ok()) {
      std::cerr << "final hidden download failed: " << hidden_status.message() << '\n';
      return 1;
    }
    std::ofstream capture{hidden_capture_path, std::ios::binary | std::ios::trunc};
    if (!capture.good()) return 1;
    capture.write(reinterpret_cast<const char*>(hidden.data()),
                  static_cast<std::streamsize>(hidden_buffer.size));
    if (!capture.good()) return 1;
  }
  std::cout << "qwen38 e2e token=0 greedy=" << best << " logit=" << best_value
            << " checksum=" << checksum << " state_buffers=" << state_buffers
            << " commands=" << session.trace().commands_executed
            << " arena_bytes=" << session.device_arena_bytes() << '\n';
  return 0;
}
