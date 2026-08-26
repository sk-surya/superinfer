#pragma once

#include <string>

#include <superinfer/compiler/target.h>

namespace superinfer::sm120 {

/** Constructs the declared offline profile used when CUDA probing is unavailable. */
inline compiler::TargetProfile declared_profile(std::uint64_t device_memory_bytes,
                                                std::string kernel_catalog) {
  return compiler::TargetProfile::offline_sm120a(device_memory_bytes, std::move(kernel_catalog));
}

}  // namespace superinfer::sm120
