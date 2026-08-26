#include <superinfer/decode/kv_cache.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

int main() {
  using superinfer::decode::KvCacheLayout;
  using superinfer::decode::KvCacheState;

  const KvCacheLayout layout{2, 3, 2, 4, 1, 16};
  assert(layout.validate().ok());
  assert(layout.storage_bytes().has_value());
  std::vector<std::byte> storage(layout.storage_bytes().value());
  const auto state_result = KvCacheState::create(layout, storage);
  assert(state_result.has_value());
  auto state = std::move(state_result).value();

  assert(state.next_position() == 0);
  assert(state.begin_step(0).ok());
  const std::array<std::byte, 8> key0{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4},
                                      std::byte{5}, std::byte{6}, std::byte{7}, std::byte{8}};
  const std::array<std::byte, 8> value0{std::byte{9}, std::byte{10}, std::byte{11}, std::byte{12},
                                        std::byte{13}, std::byte{14}, std::byte{15}, std::byte{16}};
  assert(state.write_layer(0, key0, value0).ok());
  assert(state.write_layer(1, key0, value0).ok());
  assert(state.commit_step().ok());
  assert(state.next_position() == 1);

  std::array<std::byte, 8> key_read{};
  std::array<std::byte, 8> value_read{};
  assert(state.read_layer(0, 0, key_read, value_read).ok());
  assert(key_read == key0);
  assert(value_read == value0);

  assert(state.begin_step(1).ok());
  assert(state.write_layer(0, key0, value0).ok());
  assert(!state.commit_step().ok());
  assert(state.rollback_step().ok());
  assert(state.next_position() == 1);
  assert(!state.begin_step(3).ok());
  state.reset();
  assert(state.next_position() == 0);
  assert(!state.read_layer(0, 0, key_read, value_read).ok());

  const KvCacheLayout max_layers{64, 1, 1, 1, 1, 16};
  std::vector<std::byte> max_storage(max_layers.storage_bytes().value());
  auto max_state = KvCacheState::create(max_layers, max_storage).value();
  assert(max_state.begin_step(0).ok());
  const std::array<std::byte, 1> one{std::byte{1}};
  for (std::uint32_t layer = 0; layer < 64; ++layer) {
    assert(max_state.write_layer(layer, one, one).ok());
  }
  assert(max_state.commit_step().ok());
  return 0;
}
