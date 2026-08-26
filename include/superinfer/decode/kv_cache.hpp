#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <utility>
#include <initializer_list>

#include <superinfer/base/result.hpp>

namespace superinfer::decode {

/**
 * Physical layout for a two-plane key/value cache.
 *
 * Bytes are ordered as [plane][layer][position][kv_head][feature]. The view is non-owning;
 * callers allocate the returned byte count once while constructing a decode session. The layout
 * contains no model identity and is valid for both full and recurrent attention state adapters.
 */
struct KvCacheLayout final {
  std::uint32_t layer_count{0};
  std::uint32_t max_sequence_length{0};
  std::uint32_t kv_head_count{0};
  std::uint32_t head_dimension{0};
  std::uint32_t element_bytes{0};
  std::uint64_t alignment{0};

  [[nodiscard]] base::Status validate() const noexcept {
    if (layer_count == 0 || layer_count > 64 || max_sequence_length == 0 || kv_head_count == 0 ||
        head_dimension == 0 || element_bytes == 0 || alignment == 0 ||
        (alignment & (alignment - 1)) != 0) {
      return base::Status::invalid_argument("KV cache layout has invalid dimensions or alignment");
    }
    if (!checked_product_fits({layer_count, max_sequence_length, kv_head_count, head_dimension,
                               element_bytes, 2})) {
      return base::Status::overflow("KV cache layout byte count overflows uint64");
    }
    return {};
  }

  [[nodiscard]] base::Result<std::uint64_t> storage_bytes() const noexcept {
    const base::Status status = validate();
    if (!status.ok()) return status;
    return checked_product({layer_count, max_sequence_length, kv_head_count, head_dimension,
                            element_bytes, 2});
  }

  [[nodiscard]] base::Result<std::uint64_t> layer_bytes() const noexcept {
    const base::Status status = validate();
    if (!status.ok()) return status;
    return checked_product({max_sequence_length, kv_head_count, head_dimension, element_bytes, 2});
  }

  [[nodiscard]] base::Result<std::uint64_t> row_bytes() const noexcept {
    const base::Status status = validate();
    if (!status.ok()) return status;
    return checked_product({kv_head_count, head_dimension, element_bytes});
  }

  /** Returns the byte offset of one complete plane row in the non-owning storage. */
  [[nodiscard]] base::Result<std::uint64_t> offset(bool value_plane, std::uint32_t layer,
                                                   std::uint32_t position) const noexcept {
    const base::Status status = validate();
    if (!status.ok()) return status;
    if (layer >= layer_count || position >= max_sequence_length) {
      return base::Status::out_of_range("KV cache coordinate is outside its declared capacity");
    }
    const std::uint64_t row = static_cast<std::uint64_t>(layer) * max_sequence_length + position;
    const std::uint64_t plane_rows = static_cast<std::uint64_t>(layer_count) * max_sequence_length;
    return (static_cast<std::uint64_t>(value_plane) * plane_rows + row) * row_bytes().value();
  }

 private:
  static bool checked_product_fits(std::initializer_list<std::uint64_t> factors) noexcept {
    std::uint64_t result = 1;
    for (const std::uint64_t factor : factors) {
      if (factor != 0 && result > std::numeric_limits<std::uint64_t>::max() / factor) return false;
      result *= factor;
    }
    return true;
  }

  static base::Result<std::uint64_t> checked_product(
      std::initializer_list<std::uint64_t> factors) noexcept {
    std::uint64_t result = 1;
    for (const std::uint64_t factor : factors) {
      if (factor != 0 && result > std::numeric_limits<std::uint64_t>::max() / factor) {
        return base::Status::overflow("KV cache size overflows uint64");
      }
      result *= factor;
    }
    return result;
  }
};

/**
 * Allocation-free state machine over a caller-owned KV cache.
 *
 * A step is committed only after every layer has been written. Rollback invalidates the in-flight
 * step and clears its partial rows, so a failed decode cannot expose mixed-position state.
 */
class KvCacheState final {
 public:
  static base::Result<KvCacheState> create(const KvCacheLayout& layout,
                                           std::span<std::byte> storage) noexcept {
    const auto bytes = layout.storage_bytes();
    if (!bytes.has_value()) return bytes.error();
    if (storage.size() < bytes.value()) {
      return base::Status::resource_exhausted("KV cache storage is smaller than its layout");
    }
    return KvCacheState{layout, storage};
  }

  [[nodiscard]] std::uint32_t next_position() const noexcept { return next_position_; }

  base::Status begin_step(std::uint32_t position) noexcept {
    if (in_flight_) return base::Status::failed_precondition("KV cache step is already in flight");
    if (position != next_position_ || position >= layout_.max_sequence_length) {
      return base::Status::out_of_range("KV cache step must append at its next valid position");
    }
    in_flight_ = true;
    written_layers_ = 0;
    return {};
  }

  base::Status write_layer(std::uint32_t layer, std::span<const std::byte> key,
                           std::span<const std::byte> value) noexcept {
    if (!in_flight_) return base::Status::failed_precondition("KV cache step is not in flight");
    if (layer >= layout_.layer_count) return base::Status::out_of_range("KV cache layer is undefined");
    const auto row_bytes = layout_.row_bytes();
    if (!row_bytes.has_value()) return row_bytes.error();
    if (key.size() != row_bytes.value() || value.size() != row_bytes.value()) {
      return base::Status::invalid_argument("KV cache row size does not match its layout");
    }
    const std::uint64_t layer_bit = std::uint64_t{1} << layer;
    if ((written_layers_ & layer_bit) != 0) {
      return base::Status::failed_precondition("KV cache layer was written twice in one step");
    }
    const auto key_offset = layout_.offset(false, layer, next_position_);
    const auto value_offset = layout_.offset(true, layer, next_position_);
    if (!key_offset.has_value() || !value_offset.has_value()) {
      return base::Status::internal("KV cache layout rejected a validated coordinate");
    }
    std::memcpy(storage_.data() + key_offset.value(), key.data(), key.size());
    std::memcpy(storage_.data() + value_offset.value(), value.data(), value.size());
    written_layers_ |= layer_bit;
    return {};
  }

  base::Status commit_step() noexcept {
    if (!in_flight_) return base::Status::failed_precondition("KV cache step is not in flight");
    const std::uint64_t expected = layout_.layer_count == 64
                                       ? std::numeric_limits<std::uint64_t>::max()
                                       : (std::uint64_t{1} << layout_.layer_count) - 1;
    if (written_layers_ != expected) {
      return base::Status::failed_precondition("KV cache step is missing one or more layers");
    }
    in_flight_ = false;
    written_layers_ = 0;
    ++next_position_;
    return {};
  }

  base::Status rollback_step() noexcept {
    if (!in_flight_) return base::Status::failed_precondition("KV cache step is not in flight");
    const auto row_bytes = layout_.row_bytes();
    if (!row_bytes.has_value()) return row_bytes.error();
    for (std::uint32_t layer = 0; layer < layout_.layer_count; ++layer) {
      if ((written_layers_ & (std::uint64_t{1} << layer)) == 0) continue;
      const auto key_offset = layout_.offset(false, layer, next_position_);
      const auto value_offset = layout_.offset(true, layer, next_position_);
      if (!key_offset.has_value() || !value_offset.has_value()) {
        return base::Status::internal("KV cache layout rejected a validated rollback coordinate");
      }
      std::memset(storage_.data() + key_offset.value(), 0, row_bytes.value());
      std::memset(storage_.data() + value_offset.value(), 0, row_bytes.value());
    }
    in_flight_ = false;
    written_layers_ = 0;
    return {};
  }

  base::Status read_layer(std::uint32_t layer, std::uint32_t position, std::span<std::byte> key,
                          std::span<std::byte> value) const noexcept {
    if (in_flight_ || position >= next_position_) {
      return base::Status::failed_precondition("KV cache position is not committed");
    }
    if (layer >= layout_.layer_count) return base::Status::out_of_range("KV cache layer is undefined");
    const auto row_bytes = layout_.row_bytes();
    if (!row_bytes.has_value()) return row_bytes.error();
    if (key.size() != row_bytes.value() || value.size() != row_bytes.value()) {
      return base::Status::invalid_argument("KV cache output row size does not match its layout");
    }
    const auto key_offset = layout_.offset(false, layer, position);
    const auto value_offset = layout_.offset(true, layer, position);
    if (!key_offset.has_value() || !value_offset.has_value()) {
      return base::Status::internal("KV cache layout rejected a committed coordinate");
    }
    std::memcpy(key.data(), storage_.data() + key_offset.value(), key.size());
    std::memcpy(value.data(), storage_.data() + value_offset.value(), value.size());
    return {};
  }

  void reset() noexcept {
    std::fill(storage_.begin(), storage_.end(), std::byte{0});
    next_position_ = 0;
    in_flight_ = false;
    written_layers_ = 0;
  }

 private:
  KvCacheState(KvCacheLayout layout, std::span<std::byte> storage)
      : layout_(layout), storage_(storage) {}

  KvCacheLayout layout_;
  std::span<std::byte> storage_;
  std::uint32_t next_position_{0};
  std::uint64_t written_layers_{0};
  bool in_flight_{false};
};

}  // namespace superinfer::decode
