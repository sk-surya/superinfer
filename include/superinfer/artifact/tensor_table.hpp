#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <superinfer/artifact/sinf.hpp>

namespace superinfer::artifact {

/** The bounded metadata needed to bind one tensor payload into a physical buffer. */
struct TensorTableRecord final {
  std::string name;
  std::string role;
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::vector<std::uint64_t> logical_shape;
  std::string physical_dtype;
  std::string layout;
  std::string storage_encoding;
  std::uint64_t payload_offset{0};
  std::uint64_t payload_end{0};
};

namespace detail {

/** Small, non-allocating JSON cursor for the converter's canonical tensor-table schema. */
class JsonCursor final {
 public:
  explicit JsonCursor(base::ConstByteView bytes)
      : input_(reinterpret_cast<const char*>(bytes.data()), bytes.size()) {}

  [[nodiscard]] base::Status expect(char value) {
    whitespace();
    if (position_ >= input_.size() || input_[position_] != value) {
      return base::Status::data_loss("tensor table JSON has an unexpected token");
    }
    ++position_;
    return {};
  }

  [[nodiscard]] base::Result<std::string> string() {
    whitespace();
    if (position_ >= input_.size() || input_[position_] != '"') {
      return base::Status::data_loss("tensor table JSON string is missing");
    }
    ++position_;
    std::string result;
    while (position_ < input_.size()) {
      const char value = input_[position_++];
      if (value == '"') return result;
      if (value == '\\') {
        if (position_ >= input_.size()) return base::Status::data_loss("truncated JSON escape");
        const char escaped = input_[position_++];
        switch (escaped) {
          case '"': result.push_back('"'); break;
          case '\\': result.push_back('\\'); break;
          case '/': result.push_back('/'); break;
          case 'b': result.push_back('\b'); break;
          case 'f': result.push_back('\f'); break;
          case 'n': result.push_back('\n'); break;
          case 'r': result.push_back('\r'); break;
          case 't': result.push_back('\t'); break;
          default: return base::Status::data_loss("unsupported JSON escape in tensor table");
        }
      } else {
        if (static_cast<unsigned char>(value) < 0x20U) {
          return base::Status::data_loss("control character in tensor table JSON string");
        }
        result.push_back(value);
      }
    }
    return base::Status::data_loss("unterminated tensor table JSON string");
  }

  [[nodiscard]] base::Result<std::uint64_t> unsigned_integer() {
    whitespace();
    const char* begin = input_.data() + position_;
    const char* end = input_.data() + input_.size();
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(begin, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr == begin) {
      return base::Status::data_loss("tensor table JSON integer is invalid");
    }
    position_ = static_cast<std::size_t>(parsed.ptr - input_.data());
    return value;
  }

  [[nodiscard]] base::Status array(std::vector<std::uint64_t>& values) {
    auto status = expect('[');
    if (!status.ok()) return status;
    whitespace();
    if (position_ < input_.size() && input_[position_] == ']') {
      ++position_;
      return {};
    }
    while (true) {
      const auto value = unsigned_integer();
      if (!value.has_value()) return value.error();
      values.push_back(value.value());
      whitespace();
      if (position_ >= input_.size()) return base::Status::data_loss("truncated tensor table array");
      if (input_[position_] == ']') {
        ++position_;
        return {};
      }
      status = expect(',');
      if (!status.ok()) return status;
    }
  }

  [[nodiscard]] base::Status skip_value() {
    whitespace();
    if (position_ >= input_.size()) return base::Status::data_loss("missing JSON value");
    if (input_[position_] == '"') {
      const auto ignored = string();
      return ignored.has_value() ? base::Status{} : ignored.error();
    }
    if (input_[position_] == '[') {
      ++position_;
      whitespace();
      if (position_ < input_.size() && input_[position_] == ']') {
        ++position_;
        return {};
      }
      while (true) {
        const auto status = skip_value();
        if (!status.ok()) return status;
        whitespace();
        if (position_ < input_.size() && input_[position_] == ']') {
          ++position_;
          return {};
        }
        const auto comma = expect(',');
        if (!comma.ok()) return comma;
      }
    }
    if (input_[position_] == '{') {
      ++position_;
      whitespace();
      if (position_ < input_.size() && input_[position_] == '}') {
        ++position_;
        return {};
      }
      while (true) {
        const auto key = string();
        if (!key.has_value()) return key.error();
        const auto colon = expect(':');
        if (!colon.ok()) return colon;
        const auto value = skip_value();
        if (!value.ok()) return value;
        whitespace();
        if (position_ < input_.size() && input_[position_] == '}') {
          ++position_;
          return {};
        }
        const auto comma = expect(',');
        if (!comma.ok()) return comma;
      }
    }
    while (position_ < input_.size() && input_[position_] != ',' && input_[position_] != ']' &&
           input_[position_] != '}') {
      ++position_;
    }
    return {};
  }

  void whitespace() noexcept {
    while (position_ < input_.size() &&
           (input_[position_] == ' ' || input_[position_] == '\n' || input_[position_] == '\r' ||
            input_[position_] == '\t')) {
      ++position_;
    }
  }

  [[nodiscard]] bool at_end() {
    whitespace();
    return position_ == input_.size();
  }

 private:
  std::string_view input_;
  std::size_t position_{0};
};

inline base::Status parse_tensor_record(JsonCursor& cursor, TensorTableRecord& record) {
  auto status = cursor.expect('{');
  if (!status.ok()) return status;
  cursor.whitespace();
  if (cursor.at_end()) return base::Status::data_loss("truncated tensor table object");
  while (true) {
    const auto key = cursor.string();
    if (!key.has_value()) return key.error();
    status = cursor.expect(':');
    if (!status.ok()) return status;
    if (key.value() == "name") {
      const auto value = cursor.string();
      if (!value.has_value()) return value.error();
      record.name = std::move(value).value();
    } else if (key.value() == "role") {
      const auto value = cursor.string();
      if (!value.has_value()) return value.error();
      record.role = std::move(value).value();
    } else if (key.value() == "dtype") {
      const auto value = cursor.string();
      if (!value.has_value()) return value.error();
      record.dtype = std::move(value).value();
    } else if (key.value() == "physical_dtype") {
      const auto value = cursor.string();
      if (!value.has_value()) return value.error();
      record.physical_dtype = std::move(value).value();
    } else if (key.value() == "layout") {
      const auto value = cursor.string();
      if (!value.has_value()) return value.error();
      record.layout = std::move(value).value();
    } else if (key.value() == "storage_encoding") {
      const auto value = cursor.string();
      if (!value.has_value()) return value.error();
      record.storage_encoding = std::move(value).value();
    } else if (key.value() == "shape") {
      status = cursor.array(record.shape);
      if (!status.ok()) return status;
    } else if (key.value() == "logical_shape") {
      status = cursor.array(record.logical_shape);
      if (!status.ok()) return status;
    } else if (key.value() == "artifact_payload_offset") {
      const auto value = cursor.unsigned_integer();
      if (!value.has_value()) return value.error();
      record.payload_offset = value.value();
    } else if (key.value() == "artifact_payload_end") {
      const auto value = cursor.unsigned_integer();
      if (!value.has_value()) return value.error();
      record.payload_end = value.value();
    } else {
      status = cursor.skip_value();
      if (!status.ok()) return status;
    }
    cursor.whitespace();
    if (cursor.at_end()) return base::Status::data_loss("truncated tensor table object");
    if (cursor.expect('}').ok()) return {};
    status = cursor.expect(',');
    if (!status.ok()) return status;
  }
}

}  // namespace detail

/** Parses and validates the canonical tensor-table section without reading payload bytes. */
inline base::Result<std::vector<TensorTableRecord>> parse_tensor_table(base::ConstByteView bytes) {
  detail::JsonCursor cursor{bytes};
  auto status = cursor.expect('[');
  if (!status.ok()) return status;
  std::vector<TensorTableRecord> records;
  cursor.whitespace();
  if (cursor.expect(']').ok()) return records;
  while (true) {
    TensorTableRecord record;
    status = detail::parse_tensor_record(cursor, record);
    if (!status.ok()) return status;
    // Accept pre-metadata v1 payload tables as well. Current converters emit these fields, while
    // deriving them here keeps older authenticated artifacts inspectable without weakening the
    // physical binding: the source dtype/name still determine the only legal representation.
    if (record.physical_dtype.empty()) {
      if (record.dtype == "BF16") record.physical_dtype = "bf16";
      if (record.dtype == "F32") record.physical_dtype = "f32";
      if (record.dtype == "F16") record.physical_dtype = "f16";
      if (record.dtype == "I32") record.physical_dtype = "int32";
      if (record.dtype == "U8" || record.dtype == "F8_E4M3") record.physical_dtype = "u8";
    }
    if (record.layout.empty()) record.layout = "row_major";
    if (record.storage_encoding.empty()) {
      record.storage_encoding = record.dtype == "F8_E4M3" ? "fp8_e4m3_group_scale" : "none";
      if (record.dtype == "U8" && record.name.ends_with(".weight")) {
        record.storage_encoding = "nvfp4_packed";
      }
    }
    if (record.logical_shape.empty()) {
      record.logical_shape = record.shape.empty() ? std::vector<std::uint64_t>{1} : record.shape;
      if (record.storage_encoding == "nvfp4_packed") {
        if (record.logical_shape.size() != 2 || record.logical_shape[1] >
            std::numeric_limits<std::uint64_t>::max() / 2U) {
          return base::Status::data_loss("packed tensor logical shape is invalid");
        }
        record.logical_shape[1] *= 2U;
      }
    }
    if (record.name.empty() || record.physical_dtype.empty() || record.layout.empty() ||
        record.storage_encoding.empty() || record.logical_shape.empty() ||
        record.payload_end <= record.payload_offset) {
      return base::Status::data_loss("tensor table record is incomplete");
    }
    records.push_back(std::move(record));
    cursor.whitespace();
    if (cursor.expect(']').ok()) break;
    status = cursor.expect(',');
    if (!status.ok()) return status;
  }
  if (!cursor.at_end()) return base::Status::data_loss("tensor table JSON has trailing data");
  return records;
}

}  // namespace superinfer::artifact
