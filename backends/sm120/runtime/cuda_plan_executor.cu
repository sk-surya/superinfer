#include <sm120/runtime/cuda_plan_executor.cuh>

// Keep the CUDA executor in the backend target so the packaged runtime compiles the same
// registry and launch contracts that GPU consumers include for their public session interface.
