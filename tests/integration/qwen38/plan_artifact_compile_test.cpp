#include <frontends/qwen38/frontend.hpp>
#include <sm120/compiler/specializer.h>
#include <sm120/kernels/baseline/provider.h>
#include <superinfer/artifact/plan_binding.hpp>
#include <superinfer/artifact/sinf.hpp>
#include <superinfer/artifact/tensor_table.hpp>
#include <superinfer/compiler/model_frontend.hpp>
#include <superinfer/compiler/semantic_lowering.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
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

}  // namespace

int main() {
  const char* artifact_path = std::getenv("SUPERINFER_QWEN38_ARTIFACT");
  if (artifact_path == nullptr) {
    std::cerr << "SKIP: set SUPERINFER_QWEN38_ARTIFACT\n";
    return 77;
  }
  MappedFile file{artifact_path};
  if (!file.valid()) return 1;
  const auto artifact = superinfer::artifact::ArtifactReader::read(file.bytes());
  if (!artifact.has_value()) return 1;
  const auto integrity = artifact.value().validate_integrity();
  if (!integrity.ok()) return 1;
  const auto table = artifact.value().section(superinfer::artifact::SectionKind::tensor_table);
  if (!table.has_value()) return 1;
  const auto records = superinfer::artifact::parse_tensor_table(table.value());
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
      std::string{superinfer::frontends::qwen38::kSourceIdentity},
      source_records.size(),
      std::string{superinfer::frontends::qwen38::kTensorInventorySha256},
      std::move(source_records)};
  superinfer::frontends::qwen38::Frontend frontend;
  const auto frontend_status = frontend.validate(source);
  if (!frontend_status.ok()) {
    std::cerr << "frontend validation failed: " << frontend_status.message() << '\n';
    return 1;
  }
  std::uint32_t sequence_length = 1;
  if (const char* encoded = std::getenv("SUPERINFER_QWEN38_PREFILL_LENGTH"); encoded != nullptr) {
    sequence_length = static_cast<std::uint32_t>(std::strtoul(encoded, nullptr, 10));
  }
  const auto semantic = frontend.emit(source, sequence_length);
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
    std::cerr << "artifact plan binding failed: " << resolved.error().message() << '\n';
    return 1;
  }
  std::cout << "qwen38 artifact plan compile tensors=" << lowered.value().tensors().size()
            << " commands=" << specialized.value().plan.commands().size()
            << " artifact_buffers=" << resolved.value().size()
            << " arena_bytes=" << specialized.value().memory.device_arena_bytes
            << " kv_capacity=4096 sequence_length=" << sequence_length << '\n';
  return 0;
}
