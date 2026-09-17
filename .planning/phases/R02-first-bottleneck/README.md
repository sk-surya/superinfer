# R02 — Profiler-Selected First Bottleneck

R02 is intentionally not preplanned to a kernel or provider before R01 profiling. After R01 selects exactly one dominant actionable steady-state decode bottleneck, the autonomous agent must create `R02-PLAN.md` in this directory naming the exact measured source files/functions, differential test, optimization hypothesis, Amdahl ceiling, benchmark command, and rollback boundary, self-review it, then execute it without waiting for user approval.

Preselecting a target before R01 is a planning error.