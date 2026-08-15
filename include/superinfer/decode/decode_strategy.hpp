#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <superinfer/base/result.hpp>

namespace superinfer::decode {

/** Compile-time resources requested by a token policy. */
struct DecodeRequirements final {
  std::uint64_t workspace_bytes;
  bool needs_verification_graph;
};

/** Bounded opaque state view owned and sized by the materialized physical plan. */
struct DecodeStateView final {
  std::byte* data;
  std::uint64_t size;
};

/** Owns proposal/acceptance policy while attention remains a kernel concern. */
class DecodeStrategy {
 public:
  virtual ~DecodeStrategy() = default;
  virtual DecodeRequirements requirements() const noexcept = 0;
  virtual base::Status initialize(DecodeStateView) const = 0;
};

template <typename T>
inline constexpr bool is_decode_strategy = std::is_base_of_v<DecodeStrategy, T>;

}  // namespace superinfer::decode
