#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <superinfer/compiler/graph_pass.hpp>

namespace superinfer::compiler {

/** Ordered pass provenance retained as part of deterministic compilation evidence. */
struct PassRecord final {
  const GraphPass* pass;
  PassDescriptor descriptor;
};

/**
 * Validates and executes an explicit deterministic pass sequence. Pass objects are non-owning and
 * must outlive this manager; callers own pass lifetime and the manager owns only provenance.
 */
class PassManager final {
 public:
  base::Status add(const GraphPass& pass) {
    const PassDescriptor descriptor = pass.descriptor();
    if (descriptor.name.empty()) return base::Status::invalid_argument("graph pass ID is empty");
    if (!descriptor.deterministic) {
      return base::Status::failed_precondition("graph pass must declare deterministic behavior");
    }
    for (const PassRecord& record : passes_) {
      if (record.descriptor.name == descriptor.name) {
        return base::Status::invalid_argument("duplicate graph pass ID: " + std::string(descriptor.name));
      }
    }
    passes_.push_back({&pass, descriptor});
    return {};
  }

  base::Status run() const {
    for (const PassRecord& record : passes_) {
      base::Status status = record.pass->apply();
      if (!status.ok()) return status.with_context(record.descriptor.name);
    }
    return {};
  }

  [[nodiscard]] std::string provenance() const {
    std::string result;
    for (const PassRecord& record : passes_) {
      if (!result.empty()) result += ";";
      result += std::string(record.descriptor.name) + "@" + std::to_string(record.descriptor.version);
      if (!record.descriptor.configuration.empty()) {
        result += "{" + std::string(record.descriptor.configuration) + "}";
      }
    }
    return result;
  }

 private:
  std::vector<PassRecord> passes_;
};

}  // namespace superinfer::compiler
