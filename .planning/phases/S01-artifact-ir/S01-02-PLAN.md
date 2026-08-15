---
phase: "S01-artifact-ir"
plan: "S01-02"
type: "feature"
wave: 2
depends_on: [S01-01]
files_modified:
  - include/superinfer/ir/lowered/**
  - include/superinfer/runtime/physical_plan.h
  - include/superinfer/compiler/pass_manager.h
  - src/{ir/lowered,compiler}/**
  - tests/{unit,property,golden}/{ir,compiler,runtime}/**
autonomous: true
requirements_addressed: [ARCH-001, ARCH-003, ARCH-004, ARCH-007, BCK-003]
must_haves:
  truths:
    - "Lowered IR exposes target/layout/fusion/resource decisions without becoming executable runtime policy."
    - "GraphPass ordering and analysis invalidation are explicit."
    - "Physical Plans are immutable and reject all invalid references/resources."
  artifacts:
    - "Lowered IR and verifier"
    - "Deterministic pass manager and pass provenance"
    - "Versioned Physical Plan schema, builder, verifier and golden dumps"
---

# S01-02 — Lowering, Pass Manager, and Physical Plan Schema

## Objective

Create the compiler-side target representation and the complete runtime execution schema without implementing `sm120` kernels.

## Tasks

1. **Define Lowered IR**
   - Represent physical shapes/padding, storage/accumulation dtypes, quantization parameters, layouts, memory spaces, fused regions, KV layout, workspaces, scheduling dependencies, and kernel capability requirements.
   - Keep provider implementation handles, device addresses, and concrete arena offsets out until Physical Plan construction.
   - Implement checked builders, verifier, deterministic traversal/dump, and semantic-origin links.

2. **Implement GraphPass contracts and manager**
   - Require stable pass ID/version, input representation, preconditions, configuration schema, preserved/invalidated analyses, and stated postconditions.
   - Validate after each pass in debug/test modes; report failing pass and minimized context.
   - Serialize the ordered pipeline and configuration into compilation provenance.
   - Test duplicate pass IDs, invalid order, nondeterministic output detection, and analysis invalidation.

3. **Implement representative synthetic passes**
   - Add deterministic canonicalization, constant folding, explicit layout assignment, and simple fusion marker passes over synthetic fixtures.
   - Use them to prove semantic-to-lowered transitions and pass failure rollback.
   - Avoid committing target policy that belongs to S02.

4. **Define Physical Plan schema**
   - Represent target/capability fingerprint, stable buffer/arena descriptors, constants, command DAG/streams/events, kernel IDs and launch blobs, entry schedules, decode state, and resource maxima.
   - Use checked offset/size/index types. Make plan immutable after successful builder finalization.
   - Version the schema independently from container format where appropriate.

5. **Implement Physical Plan verifier**
   - Check schema/capabilities, buffer bounds/alignment/non-overlap rules, command references, DAG acyclicity, stream/event ordering, launch/workspace bounds, entry points, decode state and resource caps.
   - Ensure verifier needs no CUDA driver and performs no allocation based on unvalidated values.

6. **Add representation transition/golden tests**
   - Produce canonical Semantic -> Lowered -> Physical dumps for tiny fixtures.
   - Test that every lowering decision is traceable to semantic origin/pass provenance.
   - Fuzz/property-mutate plan indexes, counts, offsets, DAG edges, and resource declarations.

## Verification

- Run all IR/pass/plan tests under ASan/UBSan where supported.
- Confirm byte-identical dumps across two clean processes.
- Confirm Physical Plan public consumer API exposes only immutable views.
- Confirm adversarial counts/offsets fail before allocation and without excessive CPU/memory use.

## Completion Evidence

- Three-representation golden pipeline.
- Pass pipeline provenance example.
- Physical Plan negative-test matrix mapped to BCK-003.
