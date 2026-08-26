# Runtime trace envelope

The CUDA baseline exposes `CudaLifecycleTrace` as a bounded, allocation-free summary owned by a
`CudaPlanSession`. Construction counters cover device arenas, streams, events, and kernel binding;
the execution boundary records explicit `cudaDeviceSynchronize` calls made by test/copy APIs.

`execute()` does not update any lifecycle counter. A steady-state decode qualification therefore
captures the counter before warmup, executes the prebound plan repeatedly, and asserts that the
allocation/free/registry/synchronization envelope is unchanged. Host/device copy helpers are setup
boundaries and are intentionally excluded from that steady-state assertion.

The counters are intentionally numeric and bounded. Formatting, filesystem access, and event-log
allocation belong in an offline harness, never in the token loop.
