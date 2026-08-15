# Development workflow

SuperInfer is intentionally useful on a CPU-only machine. The core Python package imports no
Torch, CUDA, Hugging Face, or safetensors module. Model and GPU dependencies are optional
extras in `pyproject.toml` and are not needed for the foundation tests.

## Fast validation

From the repository root:

```bash
python3 tools/validate.py
```

This runs the Python syntax/import boundary check, deterministic unit/property/golden tests,
C++ CPU configure/build/CTest, an install-tree consumer compile/run, and writes structured
failure context under `build/cpu-dev/artifacts/S00/validation/` when a step fails.

The full CPU profile adds sanitizer CTest and a wheel build:

```bash
python3 tools/validate.py --full
```

The equivalent individual CMake commands are documented in `docs/toolchain.md`. CUDA is an
optional later lane; `cmake --preset cuda-sm120a` only probes the compiler in this phase.

## Tests and evidence

- `tests/unit/` owns contract behavior for support utilities.
- `tests/property/` owns bounded shape/property checks and persists future failure seeds.
- `tests/golden/` owns canonical serialized fixture expectations.
- C++ tests live beside their owning layer and use CTest's failure output.
- Test helpers retain seed, command, stdout, stderr, and return code in JSON when a subprocess
  fails. Do not update a golden without reviewing the semantic diff.

The repository guide, `.planning/QUALITY.md`, and the active phase plan define the required
evidence for new tests and implementation changes.

