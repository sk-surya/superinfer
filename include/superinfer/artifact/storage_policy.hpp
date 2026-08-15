#pragma once

#include <cstdint>
#include <string_view>
#include <type_traits>

#include <superinfer/base/memory_space.hpp>
#include <superinfer/base/result.hpp>
#include <superinfer/base/views.hpp>

namespace superinfer::artifact {

/** Format-independent tensor storage facts presented to kernels as typed views later. */
struct StorageDescriptor final {
  base::MemorySpace space;
  std::uint64_t alignment;
  std::uint64_t bytes;
};

/** Packages and materializes storage without exposing artifact offsets to kernels. */
class StoragePolicy {
 public:
  virtual ~StoragePolicy() = default;
  virtual base::Result<StorageDescriptor> plan(std::uint64_t bytes) const = 0;
  virtual base::Status package(std::string_view identity, base::ConstByteView payload) const = 0;
};

template <typename T>
inline constexpr bool is_storage_policy = std::is_base_of_v<StoragePolicy, T>;

}  // namespace superinfer::artifact

