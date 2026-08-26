#include <superinfer/artifact/storage_policy.hpp>
#include <superinfer/compiler/graph_pass.hpp>
#include <superinfer/compiler/model_frontend.hpp>
#include <superinfer/decode/decode_strategy.hpp>
#include <superinfer/ir/lowered_ir.hpp>
#include <superinfer/ir/physical_plan.hpp>
#include <superinfer/ir/semantic_ir.hpp>
#include <superinfer/kernels/kernel_provider.hpp>
#include <superinfer/runtime/executor_contract.hpp>

#include <cassert>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

class FakeFrontend final : public superinfer::compiler::ModelFrontend {
 public:
  superinfer::base::Status validate(const superinfer::compiler::SourceInventory&) const override {
    return {};
  }
  superinfer::base::Result<superinfer::ir::semantic::Module> emit(
      const superinfer::compiler::SourceInventory&) const override {
    return superinfer::ir::semantic::Module::empty();
  }
};

class FakePass final : public superinfer::compiler::GraphPass {
 public:
  superinfer::compiler::PassDescriptor descriptor() const noexcept override {
    return {superinfer::compiler::Representation::semantic_ir, "fixture", "none", "none"};
  }
  superinfer::base::Status apply() const override { return {}; }
};

class FakeProvider final : public superinfer::kernels::KernelProvider {
 public:
  superinfer::base::Result<std::vector<superinfer::kernels::KernelCandidate>> enumerate(
      const superinfer::kernels::KernelQuery&) const override {
    return std::vector<superinfer::kernels::KernelCandidate>{};
  }
};

class FakeDecode final : public superinfer::decode::DecodeStrategy {
 public:
  superinfer::decode::DecodeRequirements requirements() const noexcept override { return {0, false}; }
  superinfer::base::Status initialize(superinfer::decode::DecodeStateView) const override {
    return {};
  }
};

class FakeStorage final : public superinfer::artifact::StoragePolicy {
 public:
  superinfer::base::Result<superinfer::artifact::StorageDescriptor> plan(
      std::uint64_t bytes) const override {
    return superinfer::artifact::StorageDescriptor{superinfer::base::MemorySpace::host, 1, bytes};
  }
  superinfer::base::Status package(std::string_view,
                                   superinfer::base::ConstByteView) const override {
    return {};
  }
};

}  // namespace

int main() {
  using namespace superinfer;
  static_assert(!std::is_same_v<ir::semantic::Module, ir::lowered::Module>);
  static_assert(!std::is_same_v<ir::lowered::Module, ir::physical::Plan>);
  static_assert(!std::is_constructible_v<ir::physical::Plan, ir::semantic::Module>);

  static_assert(compiler::is_model_frontend<compiler::ModelFrontend>);
  static_assert(compiler::is_graph_pass<compiler::GraphPass>);
  static_assert(kernels::is_kernel_provider<kernels::KernelProvider>);
  static_assert(decode::is_decode_strategy<decode::DecodeStrategy>);
  static_assert(artifact::is_storage_policy<artifact::StoragePolicy>);

  const FakeFrontend frontend;
  const FakePass pass;
  const FakeProvider provider;
  const FakeDecode decode;
  const FakeStorage storage;
  assert(frontend.validate({"fixture", 0, {}}).ok());
  assert(pass.apply().ok());
  assert(provider.enumerate({"embedding", 0}).has_value());
  assert(decode.requirements().workspace_bytes == 0);
  assert(storage.plan(32).has_value());

  const ir::physical::Plan plan = ir::physical::Plan::empty();
  const runtime::ExecutorContract executor{plan};
  assert(executor.plan_version() == plan.version());

  const std::filesystem::path source_dir{SUPERINFER_SOURCE_DIR};
  const auto runtime_header = source_dir / "include/superinfer/runtime/executor_contract.hpp";
  std::ifstream input(runtime_header);
  assert(input.good());
  const std::string contents{std::istreambuf_iterator<char>{input}, {}};
  assert(contents.find("Qwen") == std::string::npos);
  assert(contents.find("Gemma") == std::string::npos);
  assert(contents.find("ModelFrontend") == std::string::npos);
  return 0;
}
