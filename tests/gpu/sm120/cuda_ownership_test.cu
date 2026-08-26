#include <sm120/runtime/cuda_ownership.cuh>

#include <cassert>
#include <cstdint>

namespace {

__global__ void write_canary(std::uint32_t* output) {
  if (threadIdx.x == 0) output[0] = 0x509A120U;
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_error = cudaGetDeviceCount(&device_count);
  if (count_error == cudaErrorNoDevice || count_error == cudaErrorInsufficientDriver) return 77;
  assert(count_error == cudaSuccess);
  if (device_count == 0) return 77;
  const cudaError_t set_error = cudaSetDevice(0);
  if (set_error == cudaErrorNoDevice || set_error == cudaErrorInsufficientDriver) return 77;
  assert(set_error == cudaSuccess);

  cudaDeviceProp properties{};
  const cudaError_t property_error = cudaGetDeviceProperties(&properties, 0);
  if (property_error == cudaErrorNoDevice || property_error == cudaErrorInsufficientDriver) return 77;
  assert(property_error == cudaSuccess);
  if (properties.major != 12 || properties.minor != 0) return 77;

  using namespace superinfer::sm120::cuda_runtime;
  auto buffer = DeviceBuffer::allocate(sizeof(std::uint32_t));
  assert(buffer.has_value());
  auto stream = StreamOwner::create();
  assert(stream.has_value());
  auto event = EventOwner::create();
  assert(event.has_value());

  write_canary<<<1, 1, 0, stream.value().get()>>>(
      static_cast<std::uint32_t*>(buffer.value().data()));
  assert(cudaGetLastError() == cudaSuccess);
  assert(cudaEventRecord(event.value().get(), stream.value().get()) == cudaSuccess);
  std::uint32_t host_value = 0;
  assert(cudaMemcpyAsync(&host_value, buffer.value().data(), sizeof(host_value),
                         cudaMemcpyDeviceToHost, stream.value().get()) == cudaSuccess);
  assert(cudaEventRecord(event.value().get(), stream.value().get()) == cudaSuccess);
  assert(cudaEventSynchronize(event.value().get()) == cudaSuccess);
  assert(host_value == 0x509A120U);
  assert(buffer.value().reset().ok());
  assert(stream.value().reset().ok());
  assert(event.value().reset().ok());
  return 0;
}
