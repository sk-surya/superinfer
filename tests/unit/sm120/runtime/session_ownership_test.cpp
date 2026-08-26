#include <superinfer/runtime/session.hpp>

#include <cassert>
#include <type_traits>

namespace {

superinfer::ir::physical::Plan make_plan(superinfer::base::KernelId kernel) {
  superinfer::ir::physical::PlanBuilder builder;
  builder.set_resource_bounds({64, 32, 2});
  assert(builder.add_buffer(0, 16, 16).has_value());
  assert(builder.add_command(kernel, {{superinfer::ir::physical::BufferId{0}}}, {}, 0, 0, 8)
             .has_value());
  const auto plan = std::move(builder).finalize({120, "baseline-v1"});
  assert(plan.has_value());
  return std::move(plan).value();
}

}  // namespace

int main() {
  using namespace superinfer;
  static_assert(!std::is_copy_constructible_v<runtime::DeviceBuffer>);
  static_assert(std::is_move_constructible_v<runtime::DeviceBuffer>);
  static_assert(!std::is_copy_constructible_v<runtime::StreamOwner>);
  static_assert(!std::is_copy_constructible_v<runtime::EventOwner>);
  static_assert(!std::is_copy_constructible_v<runtime::RuntimeSession>);

  const auto plan = make_plan(base::KernelId{9});
  runtime::SessionOptions options;
  options.target_capability = 120;
  options.kernel_catalog = "baseline-v1";
  options.backend = runtime::BackendKind::host_reference;
  auto session = runtime::RuntimeSession::create(plan, options);
  assert(session.has_value());
  assert(session.value().device_arena_bytes() == 64);
  assert(session.value().workspace_bytes() == 32);
  assert(session.value().stream_count() == 1);
  assert(session.value().event_count() == 0);
  assert(session.value().execute().ok());
  assert(session.value().trace().commands_executed == 1);

  options.backend = runtime::BackendKind::cuda;
  const auto unavailable = runtime::RuntimeSession::create(plan, options);
  assert(!unavailable.has_value());
  assert(unavailable.error().code() == base::StatusCode::unavailable);

  options.backend = runtime::BackendKind::host_reference;
  const auto invalid = make_plan(base::KernelId{});
  const auto rejected = runtime::RuntimeSession::create(invalid, options);
  assert(!rejected.has_value());
  assert(rejected.error().code() == base::StatusCode::failed_precondition);
  return 0;
}
