// P8-R1: execute mma.sync.m16n8k64.mxf4nvf4.block_scale.scale_vec::4X on sm_120a.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

__device__ __forceinline__ void mma_mxf4nvf4_m16n8k64(
    float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4],
    uint32_t scale_a, uint32_t scale_b) {
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
      "%14, {%15, %16}, %17, {%18, %19};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(scale_a),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)), "r"(scale_b),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)));
}

__global__ void probe_kernel(float* out) {
  if ((threadIdx.x / 32) != 0) return;
  float d[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  const float c[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  // E2M1 code 0x33 -> low=3 (1.5), high=3 (1.5); A and B all 1.5.
  uint32_t a[4] = {0x33333333U, 0x33333333U, 0x33333333U, 0x33333333U};
  uint32_t b[2] = {0x33333333U, 0x33333333U};
  const uint32_t scale = 0x38383838U;  // four UE4M3 1.0 scales.
  mma_mxf4nvf4_m16n8k64(d, a, b, c, scale, scale);
  const int lane = threadIdx.x % 32;
  for (int i = 0; i < 4; ++i) out[lane * 4 + i] = d[i];
}

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
  if (cudaSetDevice(0) != cudaSuccess) return 77;
  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, 0) != cudaSuccess) return 77;
  if (props.major != 12 || props.minor != 0) return 77;
  float* out = nullptr;
  if (cudaMalloc(&out, 32 * 4 * sizeof(float)) != cudaSuccess) return 1;
  probe_kernel<<<1, 32>>>(out);
  const cudaError_t launch = cudaGetLastError();
  if (launch != cudaSuccess) {
    std::printf("launch failed: %s\n", cudaGetErrorString(launch));
    return 1;
  }
  const cudaError_t sync = cudaDeviceSynchronize();
  if (sync != cudaSuccess) {
    std::printf("sync failed: %s\n", cudaGetErrorString(sync));
    return 1;
  }
  float host[32 * 4];
  if (cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
  std::printf("EXECUTED on %s (sm_%d%d); d[0..3] of lane0 = %g %g %g %g\n", props.name,
              props.major, props.minor, host[0], host[1], host[2], host[3]);
  bool finite = true;
  for (int i = 0; i < 32 * 4; ++i) {
    if (!(host[i] == host[i]) || host[i] > 1e30F || host[i] < -1e30F) finite = false;
  }
  std::printf("finite=%s\n", finite ? "yes" : "no");
  cudaFree(out);
  return finite ? 0 : 2;
}
