#include <superinfer/compiler/pass_manager.hpp>

#include <cassert>
#include <string>

namespace {

class FixturePass final : public superinfer::compiler::GraphPass {
 public:
  FixturePass(std::string id, bool deterministic, int* counter)
      : id_(std::move(id)), deterministic_(deterministic), counter_(counter) {}

  superinfer::compiler::PassDescriptor descriptor() const noexcept override {
    return {superinfer::compiler::Representation::semantic_ir, id_, "test-effects", "test-analysis",
            1, "fixture-config", "fixture-preconditions", "fixture-postconditions", deterministic_};
  }
  superinfer::base::Status apply() const override {
    ++(*counter_);
    return {};
  }

 private:
  std::string id_;
  bool deterministic_;
  int* counter_;
};

}  // namespace

int main() {
  int counter = 0;
  FixturePass canonicalize{"canonicalize", true, &counter};
  FixturePass normalize{"normalize", true, &counter};
  FixturePass nondeterministic{"random", false, &counter};

  superinfer::compiler::PassManager manager;
  assert(manager.add(canonicalize).ok());
  assert(manager.add(normalize).ok());
  assert(!manager.add(canonicalize).ok());
  assert(manager.run().ok());
  assert(counter == 2);
  assert(manager.provenance().find("canonicalize@1") != std::string::npos);
  assert(manager.provenance().find("normalize@1") != std::string::npos);
  assert(manager.provenance().find("{fixture-config}") != std::string::npos);

  superinfer::compiler::PassManager rejected;
  const auto rejected_status = rejected.add(nondeterministic);
  assert(!rejected_status.ok());
  assert(rejected_status.message().find("deterministic") != std::string::npos);
  return 0;
}
