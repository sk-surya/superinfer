#pragma once

#include <string_view>

namespace superinfer::base {

/** Identifies where an owning allocation or a view is valid. */
enum class MemorySpace { host, pinned_host, device, managed };

constexpr std::string_view memory_space_name(MemorySpace space) noexcept {
  switch (space) {
    case MemorySpace::host:
      return "host";
    case MemorySpace::pinned_host:
      return "pinned_host";
    case MemorySpace::device:
      return "device";
    case MemorySpace::managed:
      return "managed";
  }
  return "unknown";
}

}  // namespace superinfer::base

