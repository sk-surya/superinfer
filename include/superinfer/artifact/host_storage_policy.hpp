#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include <superinfer/artifact/storage_policy.hpp>

namespace superinfer::artifact {

/**
 * CPU baseline storage policy with aligned host descriptors and explicit copied ownership.
 * Device materialization is intentionally deferred to the S02 backend.
 */
class HostStoragePolicy final : public StoragePolicy {
 public:
  base::Result<StorageDescriptor> plan(std::uint64_t bytes) const override {
    return StorageDescriptor{base::MemorySpace::host, 64, bytes};
  }

  base::Status package(std::string_view identity, base::ConstByteView payload) const override {
    if (identity.empty()) return base::Status::invalid_argument("storage identity is empty");
    if (payload.data() == nullptr && payload.size() != 0) {
      return base::Status::invalid_argument("storage payload view is invalid");
    }
    return {};
  }

  /** Copies validated host bytes into one visible owner for later device materialization. */
  static base::Result<std::vector<std::byte>> materialize(base::ConstByteView payload) {
    if (payload.data() == nullptr && payload.size() != 0) {
      return base::Status::invalid_argument("storage payload view is invalid");
    }
    if (payload.size() == 0) return std::vector<std::byte>{};
    return std::vector<std::byte>{payload.data(), payload.data() + payload.size()};
  }
};

}  // namespace superinfer::artifact
