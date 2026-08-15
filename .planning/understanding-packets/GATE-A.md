# Gate A — Semantic IR and compiler boundaries

## 1. What changed

S01 added a model-independent Semantic IR, a target-aware Lowered IR, an immutable Physical Plan,
deterministic pass provenance, and a versioned `.sinf` container. The CPU path now verifies each
boundary and can convert normalized synthetic input into an inspectable artifact.

## 2. Why it exists and which boundary it protects

The three representations prevent model meaning, target layout, and runtime execution policy from
collapsing into one dynamic interpreter. Semantic IR owns meaning; Lowered IR owns target/layout
decisions; Physical Plan owns executable resources. The runtime receives only the validated physical
contract, so model-family logic cannot leak into the token path.

## 3. One execution/data path

The current synthetic path is:

`normalized source JSON -> semantic::Builder -> semantic::Module -> GraphPass metadata -> lowered::ModuleBuilder -> physical::PlanBuilder -> ArtifactWriter -> ArtifactReader -> runtime::PhysicalPlan alias`

The converter shell accepts normalized frontend output rather than Hugging Face tensors; Qwen and
real frontend semantics remain later-phase work. The reader validates all sections and checksums
before returning read-only views.

## 4. Important shapes and data structures

- Semantic fixture: activation `hidden` `[2,4]` f32, weight `norm_weight` `[4]` f32, RMSNorm output `[2,4]`.
- Attention attributes: query heads, KV heads, head dimension, and even RoPE dimension; GQA requires query-head divisibility.
- Lowered fixture: physical shape `[2,4]`, row-major, device memory, alignment 16, f16 storage/f32 accumulation.
- Physical fixture: arena 128 bytes, workspace 64 bytes, two commands, two 32-byte aligned buffers, dependency `command 1 -> command 0`.
- `.sinf` sections: manifest, tensor table, physical plan, payload, integrity; every section is 8-byte aligned and checksummed.

## 5. Core invariants

1. A Semantic IR dump contains no physical target/storage decisions.
2. A Lowered IR tensor retains a semantic-origin ID but does not contain device addresses or executable handles.
3. A Physical Plan is immutable after builder finalization and all buffer/command references are validated before runtime use.
4. Required artifact versions/capabilities fail closed; unknown optional sections may be skipped.
5. Identical normalized inputs and tool version produce identical dumps/artifact bytes.

## 6. Performance model

S01 work is compile/load control-plane work, not GPU throughput. Canonical dumps are dominated by
sorting names and serializing metadata. Artifact validation is O(bytes + section count) with bounded
section count and no device allocation. Runtime construction can materialize already-validated views;
the token loop does not parse the artifact or select model behavior.

## 7. Likely failure modes and diagnosis

- A semantic graph that uses a later-produced tensor: inspect `Module::verify()` and the semantic property test.
- GQA or RoPE rejection: inspect operation attributes and the diagnostic message before changing tolerances.
- Physical buffer overlap or dependency cycle: inspect the Physical Plan dump and `Plan::verify()`.
- Artifact corruption/version mismatch: run `superinfer inspect artifact.sinf --json` and check the failing section/checksum.
- Model-specific behavior in runtime: run `superinfer.architecture.dependencies` and inspect the runtime header only.

## 8. Exactly three files to read

1. `include/superinfer/ir/semantic/module.hpp`
2. `include/superinfer/ir/lowered/module.hpp`
3. `include/superinfer/ir/physical_plan.hpp`

## 9. Hands-on experiment

In `tests/unit/ir/semantic_ir_test.cpp`, change the synthetic `norm` operation kind from
`OperationKind::rms_norm` to `OperationKind::layer_norm`, then run:

```bash
cmake --build --preset cpu-dev
ctest --preset cpu-dev -R superinfer.ir.semantic --output-on-failure
```

Prediction: the semantic dump test fails only on its golden `kind=rms_norm` line; tensor shapes and
the independent Lowered/Physical fixtures remain unchanged. Restore the operation kind after the
experiment (or update the golden only if you intentionally choose the new semantic fixture).

## 10. Five ownership questions

1. Explain why Semantic IR, Lowered IR, and Physical Plan are three representations rather than three names for one mutable graph.
2. Predict which dump changes if a tensor’s symbolic dimension changes from `seq` to `context`, and which dump must not change.
3. Trace `hidden [2,4]` from the semantic builder through the lowered descriptor and into a physical command buffer reference.
4. If a runtime source file starts branching on `Qwen`, which boundary invariant is violated and which test should fail first?
5. Perform the operation-kind experiment above: what failed, why did it fail, and what remained invariant?

