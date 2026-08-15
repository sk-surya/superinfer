#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace superinfer::base {

/** Non-owning mutable contiguous view; the caller owns and must outlive the view. */
template <typename T>
class View final {
 public:
  constexpr View(T* data, std::size_t size) noexcept : data_(data), size_(size) {}
  [[nodiscard]] constexpr T* data() const noexcept { return data_; }
  [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
  [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
  constexpr T& operator[](std::size_t index) const noexcept {
    assert(index < size_);
    return data_[index];
  }

 private:
  T* data_;
  std::size_t size_;
};

/** Non-owning immutable contiguous view; the caller owns and must outlive the view. */
template <typename T>
class ConstView final {
 public:
  constexpr ConstView(const T* data, std::size_t size) noexcept : data_(data), size_(size) {}
  [[nodiscard]] constexpr const T* data() const noexcept { return data_; }
  [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
  [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
  constexpr const T& operator[](std::size_t index) const noexcept {
    assert(index < size_);
    return data_[index];
  }
  constexpr ConstView subview(std::size_t offset, std::size_t length) const noexcept {
    assert(offset <= size_ && length <= size_ - offset);
    return {data_ + offset, length};
  }

 private:
  const T* data_;
  std::size_t size_;
};

using ConstByteView = ConstView<std::byte>;
using ByteView = View<std::byte>;

}  // namespace superinfer::base

