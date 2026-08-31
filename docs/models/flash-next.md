# Flash-Next research status

The official identity is now pinned from concrete upstream metadata:

| Input | Revision / hash |
|---|---|
| Hugging Face source, `Qwen/Qwen3.8-Flash-Next` | `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| Source `config.json` SHA-256 | `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b` |
| Source `model.safetensors.index.json` SHA-256 | `99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de` |
| Official repository, `QwenLM/Qwen3.8-Flash-Next` | `69885871a64393807d988b27b1b5e380e8f28526` |

The source index reports 1,658 tensor entries across 131 safetensors shards and a BF16 payload
size of 359,999,963,128 bytes. The pinned config establishes `qwen4_exp`, 48 layers, 512 routed
experts, top-10 routing, hybrid GDN/QSA layers, 4-way gated residual streams, a 2,048-entry QSA
indexer budget, a 262,144-token maximum position count, one MTP layer, and PLE metadata. These
are source-contract facts, not a claim that the full payload is locally available.

S03F-01 remains capacity-blocked. The complete 131-shard checkpoint is not present in the
workspace, and the official repository currently supplies a technical report rather than an
executable reference implementation. The local `/srv/models/hf/Qwen3.8-Flash-Next-NVFP4` tree is
a private RadixArk candidate containing only layers 0–22 (with one missing layer-21 expert-range
shard); it is not substituted for the official checkpoint. Its reported NVFP4 evaluations are not
used as SuperInfer quality evidence.

Therefore the memory ledger intentionally leaves packed category bytes, state/workspace totals,
and total bytes unknown. The residency ledger contains no feasible candidate and makes no quality
equivalence claim. The validator in `superinfer.convert.flash_next` reads config/index/header
metadata only, requires exact revisions when provenance is supplied, and fails closed on a partial
or mismatched source. Re-run it after acquiring all pinned shards and an executable reference
revision; S03F-02 must not add expert staging or paging before that evidence exists.

## Runtime-state formula evidence

The research tool now exposes a deterministic formula-only estimate that is independent of
checkpoint completeness. For the local candidate's pinned 48-layer text configuration, with
batch 1, BF16 KV/convolution storage, FP32 recurrent state, and a 2-byte activation workspace,
the estimate at the declared 262,144-token context is:

| Component | Bytes |
|---|---:|
| Full-attention KV state | 6,442,450,944 |
| Linear-attention recurrent state | 113,246,208 |
| Linear-attention convolution state | 2,949,120 |
| Reusable decode workspace lower bound | 24,576 |
| Formula total | 6,558,670,848 |

At context 4,096, the formula total is 216,883,200 bytes. These are state/workspace formulas,
not complete model residency numbers: routed experts, shared experts, PLE, non-expert weights,
and quality qualification remain blocked on the complete authenticated artifact and executable
reference.
