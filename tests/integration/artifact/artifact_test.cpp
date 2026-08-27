#include <superinfer/artifact/sinf.hpp>
#include <superinfer/artifact/host_storage_policy.hpp>
#include <superinfer/artifact/tensor_table.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  using namespace superinfer;
  using artifact::ArtifactReader;
  using artifact::ArtifactSpec;
  using artifact::ArtifactWriter;

  static_assert(artifact::kMaximumArtifactBytes >= 32ULL * (1ULL << 30U));

  if (argc == 2) {
    std::ifstream input(argv[1], std::ios::binary);
    const std::vector<char> raw{std::istreambuf_iterator<char>{input}, {}};
    const auto* bytes = reinterpret_cast<const std::byte*>(raw.data());
    const auto parsed = ArtifactReader::read({bytes, raw.size()});
    return parsed.has_value() && parsed.value().validate_integrity().ok() ? 0 : 1;
  }

  ArtifactSpec spec;
  spec.manifest = "{\"model\":\"fixture\",\"revision\":\"r1\"}";
  spec.tensor_table = "[{\"name\":\"weight\",\"bytes\":4}]";
  spec.physical_plan = "physical-plan:v1 capability=120 catalog=fixture";
  spec.payload = {std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04}};

  const auto first = ArtifactWriter::write(spec);
  const auto second = ArtifactWriter::write(spec);
  assert(first.has_value() && second.has_value());
  assert(first.value() == second.value());

  const auto parsed = ArtifactReader::read({first.value().data(), first.value().size()});
  assert(parsed.has_value());
  assert(parsed.value().section(artifact::SectionKind::manifest).has_value());
  assert(parsed.value().section(artifact::SectionKind::payload).value().size() == 4);
  const auto payload_slice = parsed.value().payload_range(1, 2);
  assert(payload_slice.has_value());
  assert(std::to_integer<unsigned char>(payload_slice.value()[0]) == 0x02);
  assert(std::to_integer<unsigned char>(payload_slice.value()[1]) == 0x03);
  assert(!parsed.value().payload_range(3, 2).has_value());
  assert(parsed.value().validate_integrity().ok());

  ArtifactSpec typed_spec;
  typed_spec.manifest = "{}";
  typed_spec.tensor_table =
      "[{\"artifact_payload_end\":32,\"artifact_payload_offset\":8,"
      "\"dtype\":\"U8\",\"layout\":\"row_major\",\"logical_shape\":[4,16],"
      "\"name\":\"layer.weight\",\"physical_dtype\":\"u8\",\"role\":\"weight\","
      "\"shape\":[4,8],\"storage_encoding\":\"nvfp4_packed\"}]";
  typed_spec.physical_plan = "plan";
  typed_spec.payload.resize(32);
  const auto typed_bytes = ArtifactWriter::write(typed_spec);
  assert(typed_bytes.has_value());
  const auto typed_artifact = ArtifactReader::read({typed_bytes.value().data(), typed_bytes.value().size()});
  assert(typed_artifact.has_value());
  const auto table = artifact::parse_tensor_table(
      typed_artifact.value().section(artifact::SectionKind::tensor_table).value());
  assert(table.has_value() && table.value().size() == 1);
  assert(table.value().front().name == "layer.weight");
  assert(table.value().front().logical_shape == std::vector<std::uint64_t>({4, 16}));
  assert(table.value().front().payload_end == 32);

  const auto malformed_table = artifact::parse_tensor_table(
      {reinterpret_cast<const std::byte*>("[{\"name\":1}]"), 13});
  assert(!malformed_table.has_value());

  artifact::HostStoragePolicy storage;
  assert(storage.plan(4).has_value());
  assert(storage.package("fixture", parsed.value().section(artifact::SectionKind::payload).value()).ok());
  assert(artifact::HostStoragePolicy::materialize(
             parsed.value().section(artifact::SectionKind::payload).value())
             .has_value());

  auto truncated = first.value();
  truncated.pop_back();
  assert(!ArtifactReader::read({truncated.data(), truncated.size()}).has_value());

  auto corrupt = first.value();
  corrupt.back() ^= std::byte{0x01};
  const auto corrupt_result = ArtifactReader::read({corrupt.data(), corrupt.size()});
  assert(!corrupt_result.has_value());
  assert(corrupt_result.error().message().find("checksum") != std::string::npos);

  auto bad_version = first.value();
  bad_version[4] = std::byte{0x02};
  assert(!ArtifactReader::read({bad_version.data(), bad_version.size()}).has_value());

  auto unknown_required = first.value();
  unknown_required[32] = std::byte{0x7f};
  const auto unknown_result = ArtifactReader::read({unknown_required.data(), unknown_required.size()});
  assert(!unknown_result.has_value());
  assert(unknown_result.error().message().find("required section") != std::string::npos);
  return 0;
}
