#include <superinfer/base/checked_math.hpp>
#include <superinfer/base/ids.hpp>
#include <superinfer/base/memory_space.hpp>
#include <superinfer/base/result.hpp>
#include <superinfer/base/status.hpp>
#include <superinfer/base/views.hpp>

#include <cassert>
#include <cstdint>
#include <type_traits>

int main() {
  using namespace superinfer::base;

  const auto overflow = checked_add(std::uint64_t{7}, UINT64_MAX);
  assert(!overflow.has_value());
  assert(overflow.error().code() == StatusCode::overflow);

  const auto product = checked_mul(std::uint64_t{7}, std::uint64_t{9});
  assert(product.has_value());
  assert(product.value() == 63);

  const auto alignment = checked_align_up(std::uint64_t{17}, 16);
  assert(alignment.has_value());
  assert(alignment.value() == 32);
  assert(!checked_align_up(17, 0).has_value());

  Status status = Status::invalid_argument("bad shape").with_context("tensor");
  assert(status.code() == StatusCode::invalid_argument);
  assert(status.message() == "bad shape");
  assert(status.context().size() == 1);

  DeviceBufferId first{1};
  DeviceBufferId second{2};
  assert(first != second);
  static_assert(!std::is_convertible_v<DeviceBufferId, TensorId>);

  const std::uint32_t values[] = {1, 2, 3};
  ConstView<std::uint32_t> view{values, 3};
  assert(view.size() == 3);
  assert(view[1] == 2);
  assert(view.subview(1, 2)[0] == 2);

  assert(memory_space_name(MemorySpace::host) == "host");
  assert(memory_space_name(MemorySpace::device) == "device");
  return 0;
}

