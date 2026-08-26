# RTX 5090 validation

The local `sm_120a` lane is configured explicitly because the default CMake discovery may retain a
stale compiler probe:

```sh
cmake -S . -B build/cuda-sm120a-13.1 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSUPERINFER_ENABLE_CUDA=ON \
  -DSUPERINFER_BUILD_TESTS=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build/cuda-sm120a-13.1 -j2
ctest --test-dir build/cuda-sm120a-13.1 --output-on-failure
```

The GPU tests preflight the active device and return CTest skip code 77 for no device, an
insufficient driver, or a non-`sm_120a` target. They do not reset devices or terminate unrelated
model processes. The test lane captures correctness and lifecycle evidence only; performance
measurements belong to the benchmark protocol.

For sanitizer qualification, run the installed tool explicitly:

```sh
/usr/local/cuda-13.1/bin/compute-sanitizer --tool memcheck \
  --error-exitcode=1 build/cuda-sm120a-13.1/tests/superinfer_sm120_cuda_plan_executor
```

Compute Sanitizer requires additional device memory. If the target is occupied by model-serving
processes, preserve the process state and record the sanitizer result as unavailable rather than
evicting workloads.
