#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <superinfer/artifact/sinf.hpp>
#include <superinfer/artifact/tensor_table.hpp>
#include <superinfer/base/result.hpp>
#include <superinfer/ir/physical_plan.hpp>

namespace superinfer::artifact {

/** One validated, borrowed artifact payload range bound to a Physical Plan buffer. */
struct ResolvedArtifactBuffer final {
  ir::physical::BufferId buffer;
  std::string artifact_name;
  base::ConstByteView payload;
};

/**
 * Resolves offline artifact names carried by typed Physical Plan buffers.
 *
 * The returned views borrow the immutable ArtifactView and remain valid only while that view's
 * backing bytes remain alive. This class performs no device work, allocation, or kernel choice;
 * callers own device materialization and must preserve the one-buffer/one-range mapping.
 */
class ArtifactPlanBinding final {
 public:
  [[nodiscard]] static base::Result<std::vector<ResolvedArtifactBuffer>> resolve(
      const ir::physical::Plan& plan, const ArtifactView& artifact,
      const std::vector<TensorTableRecord>& records) {
    const base::Status plan_status = plan.verify();
    if (!plan_status.ok()) {
      base::Status error = plan_status;
      return error.with_context("artifact physical plan");
    }
    const base::Status integrity_status = artifact.validate_integrity();
    if (!integrity_status.ok()) {
      base::Status error = integrity_status;
      return error.with_context("artifact integrity");
    }

    std::vector<ResolvedArtifactBuffer> result;
    for (const ir::physical::BufferDescriptor& buffer : plan.buffers()) {
      if (buffer.tensor.artifact_name.empty()) continue;
      const TensorTableRecord* record = nullptr;
      for (const TensorTableRecord& candidate : records) {
        if (candidate.name == buffer.tensor.artifact_name) {
          record = &candidate;
          break;
        }
      }
      if (record == nullptr) {
        return base::Status::failed_precondition(
            "physical buffer artifact tensor is absent: " + buffer.tensor.artifact_name);
      }
      if (!descriptor_matches(buffer.tensor, *record)) {
        return base::Status::failed_precondition(
            "physical buffer descriptor disagrees with artifact tensor: " +
            buffer.tensor.artifact_name);
      }
      if (record->payload_end < record->payload_offset ||
          record->payload_end - record->payload_offset != buffer.size) {
        return base::Status::failed_precondition(
            "physical buffer size disagrees with artifact tensor: " +
            buffer.tensor.artifact_name);
      }
      const auto payload = artifact.payload_range(
          record->payload_offset, record->payload_end - record->payload_offset);
      if (!payload.has_value()) {
        base::Status error = payload.error();
        return error.with_context("artifact payload range");
      }
      result.push_back({buffer.id, record->name, payload.value()});
    }
    return result;
  }

 private:
  [[nodiscard]] static bool descriptor_matches(
      const ir::physical::PhysicalTensorDescriptor& descriptor,
      const TensorTableRecord& record) noexcept {
    const auto dtype = [&]() {
      if (record.physical_dtype == "f32") return ir::physical::PhysicalDType::f32;
      if (record.physical_dtype == "f16") return ir::physical::PhysicalDType::f16;
      if (record.physical_dtype == "bf16") return ir::physical::PhysicalDType::bf16;
      if (record.physical_dtype == "int32") return ir::physical::PhysicalDType::int32;
      if (record.physical_dtype == "u8") return ir::physical::PhysicalDType::u8;
      return ir::physical::PhysicalDType::unknown;
    }();
    const auto encoding = [&]() {
      if (record.storage_encoding == "nvfp4_packed") {
        return ir::physical::StorageEncoding::nvfp4_packed;
      }
      if (record.storage_encoding == "fp8_e4m3_group_scale") {
        return ir::physical::StorageEncoding::fp8_e4m3_group_scale;
      }
      return ir::physical::StorageEncoding::none;
    }();
    return descriptor.dtype == dtype && descriptor.shape == record.logical_shape &&
           descriptor.layout == ir::physical::PhysicalLayout::row_major &&
           descriptor.encoding == encoding && descriptor.alignment >= 256;
  }
};

}  // namespace superinfer::artifact
