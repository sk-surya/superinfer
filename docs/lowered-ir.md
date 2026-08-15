# Lowered IR and Physical Plan

Lowered IR is the target-aware but still inspectable representation. It records physical shapes,
layout, memory space, alignment, storage/accumulation dtype intent, fused regions, and kernel
capability requirements. It does not contain device addresses, file offsets, function pointers,
or executable handles.

The pass manager retains an ordered, deterministic provenance record containing each pass ID,
version, configuration, representation input, effects, invalidated analyses, and postconditions.
Pass IDs must be unique and a pass that does not declare deterministic behavior is rejected before
execution. Pass objects are non-owning and must outlive the manager.

Physical Plan is the immutable execution contract. Its builder records capability fingerprint,
arena/workspace bounds, aligned buffer descriptors, kernel IDs, command streams, dependencies, and
workspace slices. Finalization checks all IDs, checked bounds, buffer overlap, dependency references,
acyclicity, command caps, and capability metadata before exposing the plan. The runtime sees only
const views and performs no graph transformation or allocation as part of validation.

