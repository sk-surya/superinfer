# S04 Autoresearch Runner — Minimum Automation of the Proven Loop

**Status:** implemented. `tools/autoresearch_runner.py`, `schemas/autoresearch-experiment.schema.json`,
`tests/unit/test_autoresearch_runner.py`.

This runner automates the **mechanics** of the loop that seven manual loops already proved:

```
profile -> isolated candidate -> local differential -> model correctness
        -> fixed benchmark (before/after) -> fresh-session reproduction -> promote/reject
```

It automates no technical judgment. It does not choose a target, form a hypothesis, design a
kernel, or decide what a good tolerance is. It executes a declarative manifest, captures evidence,
and applies declared thresholds exactly as written.

## Why a manifest, not a search

The manual loops varied in *hypothesis* (occupancy vs algorithmic redundancy vs instruction/latency)
and in *evidence source* (share table, roofline arithmetic, source inspection). They did not vary in
*steps*. The unstable part is human, so it was left outside the runner; the stable part is
mechanical, so it is encoded in data. The manifest is the experimenter's written commitment made
before the run, which is what makes a promotion auditable.

The runner is extracted from concrete instances. It is explicitly **not** a general search system.

## Manifest fields

See `schemas/autoresearch-experiment.schema.json` (JSON Schema draft 2020-12) for the authoritative
contract. Fields:

| field | meaning |
|---|---|
| `id` | stable experiment id, recorded in result and report |
| `incumbent_commit` | git commit of the retained incumbent; provenance only, never reset |
| `candidate_commit` | git commit under test |
| `candidate_worktree` | optional isolated candidate directory; resolved relative to the manifest directory. Absent means the incumbent directory is used |
| `benchmark_manifest` | path to the fixed benchmark manifest that defines the workload; recorded and hashed |
| `local_differential_command` | required argv; bit-exact/trusted differential. Must exit 0 |
| `correctness_command` | required for GPU experiments (e.g. the D-021 `tools/qwen38_s03r_acceptance.py ... --d021-contract ...`); `null` for CPU-only |
| `profile_command` | optional profiler argv; stdout captured to the evidence dir |
| `benchmark_command` | required fixed benchmark argv, run once in the incumbent context and once in the candidate context |
| `reproduction_command` | required argv that runs the benchmark again in the candidate context |
| `metrics` | `name -> {regex, direction, stage?, compare?}`; parsed from stage stdout |
| `thresholds` | `{min_local_speedup, min_e2e_gain}`, each number or `null` |
| `evidence_dir` | where `result.json`, `REPORT.md`, and captured stage output are written |

Commands are argv arrays executed directly (`subprocess.run`, no shell), so nothing is
word-split or shell-interpreted.

### Metric semantics

- `regex` must contain **exactly one capture group**; the group is parsed as a float. A declared
  metric that does not appear in stdout is *missing evidence*, never zero.
- `direction` (`higher_is_better` / `lower_is_better`) tells the runner how to turn a raw timing
  into a gain: `after/before` or `before/after`, so a gain `>= 1` always means improvement.
- `stage` defaults to `benchmark`; valid values are `local`, `benchmark`, `reproduction`.
  `benchmark` metrics are also parsed from the reproduction run because the reproduction command is
  the benchmark run again.
- `compare` is `direct` or `context`:
  - `direct` — the parsed value already is the gated gain (e.g. a local microbenchmark reports its
    own speedup). Used for the local differential, which runs once.
  - `context` — the raw value is compared between the incumbent and candidate benchmark runs
    (the before/after delta). Default for `benchmark`/`reproduction` metrics.

## Stage order and contexts

1. `profile` (optional) — candidate dir; stdout captured to evidence.
2. `local` (required) — candidate dir; must exit 0.
3. `correctness` (required unless `null`) — candidate dir; must exit 0.
4. `benchmark` (required) — incumbent dir, then candidate dir; both must exit 0.
5. `reproduction` (required) — candidate dir; must exit 0.

The incumbent context defaults to the runner's working directory or `--incumbent-dir`. The candidate
context defaults to `candidate_worktree` or `--candidate-dir`.

## Fail-closed semantics

- A manifest that is absent, unreadable, or missing a required field is rejected **before any
  execution** (exit code 1, no evidence directory created).
- The first stage that exits non-zero rejects the candidate and aborts the run; every later stage is
  recorded as `not_run` and its command is **not** executed.
- Missing thresholds can never promote: the verdict is `inconclusive`.
- A declared threshold with no parsed evidence is a reject, not a zero.
- A gain below its declared threshold is a reject.
- If a context benchmark metric's direction does not reproduce in the fresh session, the candidate is
  rejected even when the threshold passed.
- No code path invents, widens, or loosens a tolerance. The runner never edits the manifest, runs
  `git reset`, force-pushes, or deletes user files.

Exit codes: `0` promote (or dry-run completed), `1` usage/manifest/setup error, `2` reject,
`3` inconclusive.

## Evidence produced

Inside `evidence_dir`:

- `result.json` — schema, manifest path + sha256, incumbent/candidate commits, per-stage
  command/status/returncode/duration, truncated stdout/stderr, parsed metrics, before/after deltas,
  reproduction check, verdict, reasons.
- `REPORT.md` — the same content rendered for humans.
- `<stage>-<context>.stdout.txt` / `.stderr.txt` — full untruncated captures.

`--dry-run` validates the manifest and prints the exact resolved command sequence (stage, context,
cwd, argv) without executing anything and without creating the evidence directory.

## CLI

```
python3 tools/autoresearch_runner.py --manifest <experiment.json> \
    [--incumbent-dir DIR] [--candidate-dir DIR] [--evidence-dir DIR] [--dry-run]
```

## Out of scope (explicitly deferred)

- LLM or any automated **candidate generation**; hypotheses stay human.
- **CUDA synthesis** or code rewriting.
- **Bayesian / generic / parameter search**; the runner evaluates exactly one declared candidate.
- **Distributed scheduling** or multi-GPU orchestration.
- **Automatic tolerance invention** or statistical promotion bands; thresholds are declared by the
  experimenter and applied verbatim.
- Choosing the target kernel from a profile. `profile_command` output is evidence for the human; the
  runner never selects a bottleneck or a hypothesis.

The next manual loop is still run manually; the runner only removes the repeated bookkeeping around
an already-formed hypothesis.
