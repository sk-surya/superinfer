#pragma once

#include <cstdint>
#include <limits>

#include <superinfer/base/result.hpp>

namespace superinfer::base {

/** Performs unsigned addition and rejects overflow instead of wrapping. */
inline Result<std::uint64_t> checked_add(std::uint64_t left, std::uint64_t right) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return Status::overflow("unsigned addition overflow");
  }
  return left + right;
}

/** Performs unsigned multiplication and rejects overflow instead of wrapping. */
inline Result<std::uint64_t> checked_mul(std::uint64_t left, std::uint64_t right) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return Status::overflow("unsigned multiplication overflow");
  }
  return left * right;
}

/** Rounds an offset up to a non-zero alignment and rejects arithmetic overflow. */
inline Result<std::uint64_t> checked_align_up(std::uint64_t value, std::uint64_t alignment) {
  if (alignment == 0) {
    return Status::invalid_argument("alignment must be non-zero");
  }
  const std::uint64_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  return checked_add(value, alignment - remainder);
}

}  // namespace superinfer::base

