# S08: Hardening and V0 Release — Context

**Status:** Planned
**Depends on:** S07
**Critical-path role:** Converts the verified research system into a trustworthy, installable V0 release and audits every milestone claim.

<domain>
## Phase Boundary

Complete negative/security/compatibility hardening, stable GPU CI tiers, packaging/install/release documentation, clean-machine rehearsals, evidence/requirement audit, license/support/limitations, and the V0 release packet. Do not add major features or new models.
</domain>

<decisions>
## Locked Decisions

- Release scope is the exact Qwen/Gemma/RTX 5090 configurations proven in prior phases.
- Unsupported artifacts/models/hardware/options fail safely and clearly.
- Required CI, compatibility, clean-machine model demos and benchmark reproduction are release blockers.
- Large/model-licensed data are referenced by checksums/acquisition instructions, not casually redistributed.
- Known limitations and negative results are published honestly.
- No last-minute optimization bypasses the same correctness/evidence gates.

### Executor Discretion

- Packaging/release hosting mechanism and exact documentation structure.
- Security scanning/fuzz duration appropriate to available infrastructure.
- Version numbering consistent with a pre-1.0 artifact/ABI policy.
</decisions>

<canonical_refs>
## Canonical References

- `.planning/QUALITY.md` — release gate and CI lanes.
- `.planning/BENCHMARKS.md` — reproduction contract.
- `.planning/REQUIREMENTS.md` — QUA-001–005, REL-001/002, FMT-002, BCK-003.
- `.planning/RISKS.md` — complete register, especially Critical/High release risks.
- `AGENTS.md` — project definition of done and change discipline.
</canonical_refs>

<deferred>
## Deferred

New kernels/models/backends, feature redesign, production server/security boundary, multi-GPU, stable 1.0 ABI, and broad packaging ecosystems.
</deferred>
