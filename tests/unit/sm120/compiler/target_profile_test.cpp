#include <superinfer/compiler/target.h>
#include <sm120/target/profile.h>

#include <cassert>
#include <cstdint>
#include <string>

int main() {
  using superinfer::base::StatusCode;
  using superinfer::compiler::TargetProfile;

  const TargetProfile offline = TargetProfile::offline_sm120a(32ULL << 30U, "baseline-v1");
  assert(offline.validate().ok());
  assert(offline.target_name == "sm_120a");
  assert(offline.compute_capability == 120);
  assert(offline.device_memory_bytes == (32ULL << 30U));
  assert(offline.compatible_with(offline).ok());
  assert(offline.fingerprint() == "sm_120a:120:baseline-v1");
  assert(superinfer::sm120::declared_profile(offline.device_memory_bytes, offline.kernel_catalog)
             .fingerprint() == offline.fingerprint());

  TargetProfile wrong_target = offline;
  wrong_target.target_name = "sm_89";
  assert(wrong_target.validate().code() == StatusCode::unsupported);

  TargetProfile wrong_capability = offline;
  wrong_capability.compute_capability = 89;
  assert(wrong_capability.validate().code() == StatusCode::unsupported);

  TargetProfile missing_catalog = offline;
  missing_catalog.kernel_catalog.clear();
  assert(missing_catalog.validate().code() == StatusCode::invalid_argument);

  TargetProfile smaller_device = offline;
  smaller_device.device_memory_bytes = 16ULL << 30U;
  assert(offline.compatible_with(smaller_device).code() == StatusCode::resource_exhausted);

  TargetProfile different_catalog = offline;
  different_catalog.kernel_catalog = "other-v1";
  assert(offline.compatible_with(different_catalog).code() == StatusCode::failed_precondition);
  return 0;
}
