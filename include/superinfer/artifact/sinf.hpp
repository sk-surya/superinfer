#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <superinfer/base/result.hpp>
#include <superinfer/base/views.hpp>

namespace superinfer::artifact {

enum class SectionKind : std::uint32_t {
  manifest = 1,
  tensor_table = 2,
  physical_plan = 3,
  payload = 4,
  integrity = 5,
};

constexpr std::uint32_t kRequiredSection = 1;
constexpr std::uint16_t kFormatMajor = 1;
constexpr std::uint16_t kFormatMinor = 0;
constexpr std::size_t kHeaderBytes = 32;
constexpr std::size_t kDirectoryEntryBytes = 32;
constexpr std::uint64_t kMaximumArtifactBytes = 1ULL << 35U;

/** Canonical CPU-side inputs to the deterministic `.sinf` writer. */
struct ArtifactSpec final {
  std::uint16_t format_major{kFormatMajor};
  std::uint16_t format_minor{kFormatMinor};
  std::string manifest;
  std::string tensor_table;
  std::string physical_plan;
  std::vector<std::byte> payload;
};

struct SectionRecord final {
  SectionKind kind;
  std::uint32_t flags;
  std::uint64_t offset;
  std::uint64_t size;
  std::uint64_t checksum;
};

class ArtifactView final {
 public:
  [[nodiscard]] std::uint16_t format_major() const noexcept { return major_; }
  [[nodiscard]] std::uint16_t format_minor() const noexcept { return minor_; }
  [[nodiscard]] const std::vector<SectionRecord>& sections() const noexcept { return sections_; }

  [[nodiscard]] base::Result<base::ConstByteView> section(SectionKind kind) const {
    for (const SectionRecord& record : sections_) {
      if (record.kind == kind) {
        return base::ConstByteView{bytes_.data() + record.offset, static_cast<std::size_t>(record.size)};
      }
    }
    return base::Status::out_of_range("requested artifact section is absent");
  }

  /** Returns a borrowed, bounds-checked view into the payload section; the artifact owns the bytes. */
  [[nodiscard]] base::Result<base::ConstByteView> payload_range(
      std::uint64_t offset, std::uint64_t size) const {
    const auto payload = section(SectionKind::payload);
    if (!payload.has_value()) return payload.error();
    if (offset > payload.value().size() || size > payload.value().size() - offset ||
        offset > std::numeric_limits<std::size_t>::max() ||
        size > std::numeric_limits<std::size_t>::max()) {
      return base::Status::out_of_range("artifact payload range is outside the payload section");
    }
    return base::ConstByteView{payload.value().data() + static_cast<std::size_t>(offset),
                               static_cast<std::size_t>(size)};
  }

  /** Rechecks section checksums and the integrity table after structural validation. */
  [[nodiscard]] base::Status validate_integrity() const;

 private:
  friend class ArtifactReader;
  ArtifactView(base::ConstByteView bytes, std::uint16_t major, std::uint16_t minor,
               std::vector<SectionRecord> sections)
      : bytes_(bytes.data(), bytes.size()), major_(major), minor_(minor), sections_(std::move(sections)) {}

  base::ConstByteView bytes_;
  std::uint16_t major_;
  std::uint16_t minor_;
  std::vector<SectionRecord> sections_;
};

namespace detail {

inline std::uint64_t checksum(base::ConstByteView bytes) noexcept {
  std::uint64_t result = 1469598103934665603ULL;
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    result ^= std::to_integer<std::uint8_t>(bytes[index]);
    result *= 1099511628211ULL;
  }
  return result;
}

inline void write_u16(std::vector<std::byte>& output, std::size_t offset, std::uint16_t value) {
  output[offset] = std::byte{static_cast<std::uint8_t>(value)};
  output[offset + 1] = std::byte{static_cast<std::uint8_t>(value >> 8U)};
}
inline void write_u32(std::vector<std::byte>& output, std::size_t offset, std::uint32_t value) {
  for (std::size_t index = 0; index < 4; ++index) {
    output[offset + index] = std::byte{static_cast<std::uint8_t>(value >> (index * 8U))};
  }
}
inline void write_u64(std::vector<std::byte>& output, std::size_t offset, std::uint64_t value) {
  for (std::size_t index = 0; index < 8; ++index) {
    output[offset + index] = std::byte{static_cast<std::uint8_t>(value >> (index * 8U))};
  }
}
inline std::uint16_t read_u16(base::ConstByteView bytes, std::size_t offset) noexcept {
  return static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[offset])) |
         static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[offset + 1]) << 8U);
}
inline std::uint32_t read_u32(base::ConstByteView bytes, std::size_t offset) noexcept {
  std::uint32_t result = 0;
  for (std::size_t index = 0; index < 4; ++index) {
    result |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[offset + index]))
              << (index * 8U);
  }
  return result;
}
inline std::uint64_t read_u64(base::ConstByteView bytes, std::size_t offset) noexcept {
  std::uint64_t result = 0;
  for (std::size_t index = 0; index < 8; ++index) {
    result |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(bytes[offset + index]))
              << (index * 8U);
  }
  return result;
}
inline std::size_t align8(std::size_t value) noexcept { return (value + 7U) & ~std::size_t{7U}; }
inline void append(std::vector<std::byte>& output, std::span<const std::byte> bytes) {
  output.insert(output.end(), bytes.begin(), bytes.end());
}
inline void append_string(std::vector<std::byte>& output, const std::string& value) {
  append(output, std::as_bytes(std::span{value.data(), value.size()}));
}

}  // namespace detail

/** Builds deterministic bytes and atomically writes complete artifacts. */
class ArtifactWriter final {
 public:
  static base::Result<std::vector<std::byte>> write(const ArtifactSpec& spec) {
    if (spec.format_major != kFormatMajor || spec.format_minor > kFormatMinor || spec.manifest.empty() ||
        spec.tensor_table.empty() || spec.physical_plan.empty()) {
      return base::Status::invalid_argument("artifact spec has unsupported version or empty required section");
    }
    struct Pending final {
      SectionKind kind;
      std::vector<std::byte> bytes;
    };
    std::vector<Pending> pending;
    pending.push_back({SectionKind::manifest, {}});
    detail::append_string(pending.back().bytes, spec.manifest);
    pending.push_back({SectionKind::tensor_table, {}});
    detail::append_string(pending.back().bytes, spec.tensor_table);
    pending.push_back({SectionKind::physical_plan, {}});
    detail::append_string(pending.back().bytes, spec.physical_plan);
    pending.push_back({SectionKind::payload, spec.payload});

    std::vector<std::byte> integrity;
    integrity.resize(pending.size() * 16U);
    for (std::size_t index = 0; index < pending.size(); ++index) {
      detail::write_u32(integrity, index * 16U, static_cast<std::uint32_t>(pending[index].kind));
      detail::write_u64(integrity, index * 16U + 8U,
                        detail::checksum({pending[index].bytes.data(), pending[index].bytes.size()}));
    }
    pending.push_back({SectionKind::integrity, std::move(integrity)});

    const std::size_t directory_end = kHeaderBytes + pending.size() * kDirectoryEntryBytes;
    std::vector<std::byte> output(directory_end, std::byte{0});
    std::vector<SectionRecord> records;
    records.reserve(pending.size());
    for (const Pending& section : pending) {
      const std::size_t offset = detail::align8(output.size());
      output.resize(offset, std::byte{0});
      const std::size_t section_offset = output.size();
      detail::append(output, {section.bytes.data(), section.bytes.size()});
      records.push_back({section.kind, kRequiredSection, section_offset, section.bytes.size(),
                         detail::checksum({section.bytes.data(), section.bytes.size()})});
    }
    detail::write_u16(output, 4, spec.format_major);
    detail::write_u16(output, 6, spec.format_minor);
    detail::write_u32(output, 8, static_cast<std::uint32_t>(kHeaderBytes));
    detail::write_u32(output, 12, static_cast<std::uint32_t>(records.size()));
    detail::write_u64(output, 16, kHeaderBytes);
    detail::write_u64(output, 24, output.size());
    output[0] = std::byte{'S'};
    output[1] = std::byte{'I'};
    output[2] = std::byte{'N'};
    output[3] = std::byte{'F'};
    for (std::size_t index = 0; index < records.size(); ++index) {
      const std::size_t offset = kHeaderBytes + index * kDirectoryEntryBytes;
      const SectionRecord& record = records[index];
      detail::write_u32(output, offset, static_cast<std::uint32_t>(record.kind));
      detail::write_u32(output, offset + 4U, record.flags);
      detail::write_u64(output, offset + 8U, record.offset);
      detail::write_u64(output, offset + 16U, record.size);
      detail::write_u64(output, offset + 24U, record.checksum);
    }
    return output;
  }

  static base::Status write_file(const std::filesystem::path& path, const ArtifactSpec& spec) {
    const auto bytes = write(spec);
    if (!bytes.has_value()) {
      base::Status error = bytes.error();
      return error.with_context("artifact writer");
    }
    const std::filesystem::path temporary = path.string() + ".partial";
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) return base::Status::unavailable("cannot open temporary artifact output");
    stream.write(reinterpret_cast<const char*>(bytes.value().data()),
                 static_cast<std::streamsize>(bytes.value().size()));
    stream.flush();
    stream.close();
    if (!stream) return base::Status::data_loss("failed writing temporary artifact output");
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) return base::Status::unavailable("failed to atomically install artifact");
    return {};
  }
};

/** Defensive CPU-only parser; no storage or device materialization occurs during read. */
class ArtifactReader final {
 public:
  static base::Result<ArtifactView> read(base::ConstByteView bytes) {
    constexpr std::size_t kMaximumSections = 1024;
    if (bytes.size() < kHeaderBytes || bytes[0] != std::byte{'S'} || bytes[1] != std::byte{'I'} ||
        bytes[2] != std::byte{'N'} || bytes[3] != std::byte{'F'}) {
      return base::Status::data_loss("invalid artifact magic or truncated header");
    }
    const std::uint16_t major = detail::read_u16(bytes, 4);
    const std::uint16_t minor = detail::read_u16(bytes, 6);
    const std::uint32_t header_size = detail::read_u32(bytes, 8);
    const std::uint32_t section_count = detail::read_u32(bytes, 12);
    const std::uint64_t directory_offset = detail::read_u64(bytes, 16);
    const std::uint64_t total_size = detail::read_u64(bytes, 24);
    if (major != kFormatMajor || minor > kFormatMinor) {
      return base::Status::unsupported("unsupported artifact format version");
    }
    if (header_size != kHeaderBytes || section_count == 0 || section_count > kMaximumSections ||
        directory_offset != kHeaderBytes || total_size != bytes.size() || total_size > kMaximumArtifactBytes) {
      return base::Status::data_loss("invalid artifact header bounds");
    }
    const std::uint64_t directory_end = directory_offset +
                                        static_cast<std::uint64_t>(section_count) * kDirectoryEntryBytes;
    if (directory_end > bytes.size()) return base::Status::data_loss("truncated artifact section directory");
    std::vector<SectionRecord> records;
    records.reserve(section_count);
    for (std::uint32_t index = 0; index < section_count; ++index) {
      const std::size_t offset = static_cast<std::size_t>(directory_offset) + index * kDirectoryEntryBytes;
      const std::uint32_t kind_value = detail::read_u32(bytes, offset);
      const std::uint32_t flags = detail::read_u32(bytes, offset + 4U);
      const std::uint64_t section_offset = detail::read_u64(bytes, offset + 8U);
      const std::uint64_t section_size = detail::read_u64(bytes, offset + 16U);
      const std::uint64_t section_checksum = detail::read_u64(bytes, offset + 24U);
      const bool known = kind_value >= static_cast<std::uint32_t>(SectionKind::manifest) &&
                         kind_value <= static_cast<std::uint32_t>(SectionKind::integrity);
      if (!known && (flags & kRequiredSection) != 0) {
        return base::Status::unsupported("unknown required section");
      }
      if (section_offset % 8U != 0 || section_offset < directory_end || section_offset > bytes.size() ||
          section_size > bytes.size() - section_offset) {
        return base::Status::data_loss("artifact section offset or size is invalid");
      }
      if (known) {
        records.push_back({static_cast<SectionKind>(kind_value), flags, section_offset, section_size,
                           section_checksum});
      }
    }
    std::vector<SectionRecord> sorted = records;
    std::sort(sorted.begin(), sorted.end(), [](const SectionRecord& left, const SectionRecord& right) {
      return left.offset < right.offset;
    });
    for (std::size_t index = 1; index < sorted.size(); ++index) {
      if (sorted[index - 1].offset + sorted[index - 1].size > sorted[index].offset) {
        return base::Status::data_loss("artifact sections overlap");
      }
    }
    for (SectionKind required : {SectionKind::manifest, SectionKind::tensor_table,
                                 SectionKind::physical_plan, SectionKind::payload, SectionKind::integrity}) {
      const auto count = std::count_if(records.begin(), records.end(), [required](const SectionRecord& record) {
        return record.kind == required && (record.flags & kRequiredSection) != 0;
      });
      if (count != 1) return base::Status::data_loss("artifact required section is missing or duplicated");
    }
    ArtifactView view{bytes, major, minor, std::move(records)};
    const base::Status integrity = view.validate_integrity();
    if (!integrity.ok()) return integrity;
    return view;
  }
};

inline base::Status ArtifactView::validate_integrity() const {
  for (const SectionRecord& record : sections_) {
    const auto bytes = base::ConstByteView{bytes_.data() + record.offset, static_cast<std::size_t>(record.size)};
    if (detail::checksum(bytes) != record.checksum) {
      return base::Status::data_loss("artifact section checksum mismatch");
    }
  }
  const auto integrity_result = section(SectionKind::integrity);
  if (!integrity_result.has_value()) return integrity_result.error();
  const base::ConstByteView integrity = integrity_result.value();
  const std::size_t expected_size = (sections_.size() - 1U) * 16U;
  if (integrity.size() != expected_size) return base::Status::data_loss("artifact integrity table size mismatch");
  for (std::size_t index = 0; index < expected_size / 16U; ++index) {
    const SectionKind kind = static_cast<SectionKind>(detail::read_u32(integrity, index * 16U));
    const std::uint64_t expected_checksum = detail::read_u64(integrity, index * 16U + 8U);
    const auto match = std::find_if(sections_.begin(), sections_.end(), [kind](const SectionRecord& record) {
      return record.kind == kind;
    });
    if (match == sections_.end() || match->kind == SectionKind::integrity ||
        match->checksum != expected_checksum) {
      return base::Status::data_loss("artifact integrity table mismatch");
    }
  }
  return {};
}

}  // namespace superinfer::artifact
