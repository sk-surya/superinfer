#pragma once

#include <cassert>
#include <optional>
#include <utility>

#include <superinfer/base/status.hpp>

namespace superinfer::base {

/** Owns either a value or a typed failure; the value is moved, never borrowed. */
template <typename T>
class Result final {
 public:
  Result(T value) : value_(std::move(value)) {}
  Result(Status error) : error_(std::move(error)) { assert(!error_.ok()); }

  [[nodiscard]] bool has_value() const noexcept { return value_.has_value(); }
  [[nodiscard]] explicit operator bool() const noexcept { return has_value(); }
  [[nodiscard]] const T& value() const& {
    assert(value_.has_value());
    return *value_;
  }
  [[nodiscard]] T& value() & {
    assert(value_.has_value());
    return *value_;
  }
  [[nodiscard]] T&& value() && {
    assert(value_.has_value());
    return std::move(*value_);
  }
  [[nodiscard]] const Status& error() const noexcept { return error_; }

 private:
  std::optional<T> value_;
  Status error_;
};

template <>
class Result<void> final {
 public:
  Result() = default;
  Result(Status error) : error_(std::move(error)) { assert(!error_.ok()); }
  [[nodiscard]] bool has_value() const noexcept { return error_.ok(); }
  [[nodiscard]] explicit operator bool() const noexcept { return has_value(); }
  [[nodiscard]] const Status& error() const noexcept { return error_; }

 private:
  Status error_;
};

}  // namespace superinfer::base

