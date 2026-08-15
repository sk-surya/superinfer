# Build and toolchain baseline

The foundation build is C++20 with CMake 3.25 or newer and has no required third-party
dependencies. The CPU path is the default and does not require a CUDA installation or GPU.

The checked-in presets are:

- `cpu-dev`: Debug C++20 build with the CPU unit and architecture tests.
- `cpu-sanitize`: Debug build with AddressSanitizer and UndefinedBehaviorSanitizer.
- `cuda-sm120a`: optional CUDA C++20 probe/configuration targeting `sm_120a`; it warns and
  disables CUDA targets when no CUDA compiler is installed.

The development environment used to author S00 is:

| Tool | Observed version |
| --- | --- |
| CMake | 4.2.3 |
| GNU C++ | 15.2.0 |
| Ninja | 1.13.2 |
| Python | 3.14.4 (the package contract remains Python 3.12+) |

Use `cmake --preset cpu-dev`, `cmake --build --preset cpu-dev`, and
`ctest --preset cpu-dev`. The CUDA preset is a capability check at this phase; it does not
claim that a runtime or kernel exists.

