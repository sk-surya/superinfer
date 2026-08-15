#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace superinfer::base {

/** Classifies failures returned across compiler, artifact, and runtime boundaries. */
enum class StatusCode {
  ok,
  invalid_argument,
  out_of_range,
  overflow,
  failed_precondition,
  unsupported,
  resource_exhausted,
  unavailable,
  data_loss,
  internal,
};

/**
 * Owns a typed failure and its boundary context.
 *
 * Status values are immutable after construction except for context enrichment. They are
 * thread-safe when not concurrently mutated. No exception is used to report an error.
 */
class Status final {
 public:
  Status() = default;
  Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {}

  static Status invalid_argument(std::string message) {
    return {StatusCode::invalid_argument, std::move(message)};
  }
  static Status out_of_range(std::string message) {
    return {StatusCode::out_of_range, std::move(message)};
  }
  static Status overflow(std::string message) { return {StatusCode::overflow, std::move(message)}; }
  static Status failed_precondition(std::string message) {
    return {StatusCode::failed_precondition, std::move(message)};
  }
  static Status unsupported(std::string message) {
    return {StatusCode::unsupported, std::move(message)};
  }
  static Status resource_exhausted(std::string message) {
    return {StatusCode::resource_exhausted, std::move(message)};
  }
  static Status unavailable(std::string message) {
    return {StatusCode::unavailable, std::move(message)};
  }
  static Status data_loss(std::string message) {
    return {StatusCode::data_loss, std::move(message)};
  }
  static Status internal(std::string message) { return {StatusCode::internal, std::move(message)}; }

  [[nodiscard]] bool ok() const noexcept { return code_ == StatusCode::ok; }
  [[nodiscard]] StatusCode code() const noexcept { return code_; }
  [[nodiscard]] const std::string& message() const noexcept { return message_; }
  [[nodiscard]] const std::vector<std::string>& context() const noexcept { return context_; }

  /** Adds a caller-owned boundary label by copying it into this status. */
  Status& with_context(std::string_view context) {
    context_.emplace_back(context);
    return *this;
  }

 private:
  StatusCode code_{StatusCode::ok};
  std::string message_;
  std::vector<std::string> context_;
};

}  // namespace superinfer::base
