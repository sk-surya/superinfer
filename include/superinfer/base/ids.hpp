#pragma once

#include <compare>
#include <cstdint>

namespace superinfer::base {

/** A strongly typed non-owning identity used for physical buffers. */
template <typename Tag>
class StrongId final {
 public:
  constexpr explicit StrongId(std::uint64_t value = 0) noexcept : value_(value) {}
  [[nodiscard]] constexpr std::uint64_t value() const noexcept { return value_; }
  constexpr auto operator<=>(const StrongId&) const = default;

 private:
  std::uint64_t value_;
};

struct TensorIdTag;
struct DeviceBufferIdTag;
struct KernelIdTag;

using TensorId = StrongId<TensorIdTag>;
using DeviceBufferId = StrongId<DeviceBufferIdTag>;
using KernelId = StrongId<KernelIdTag>;

}  // namespace superinfer::base

