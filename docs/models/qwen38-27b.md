# Qwen3.8-27B support identity

SuperInfer’s first model target is the NVFP4 RTX 5090 derivative of Qwen3.8-27B:

- Base: `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Derivative: `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` at
  `0cc27958cefbbe231782ec8511de8c4eb5233348`.
- Local acceptance source: `/srv/models/hf/Qwen3.8-27B-NVFP4-RTX5090-LMHead4`.
- Local variant identity: `62abd1d060bd801005f47754f01619054cc248d3417699ecea414c7ede1b3a4a`, the
  SHA-256 of the canonical validated metadata manifest and complete tensor/file hash inventory.
- License: Apache-2.0.

The config uses the `qwen3_5` Transformers architecture because Qwen3.8 is built on Qwen3.5’s
architecture. It is a native vision-language model, not a plain decoder-only text model. The V0
critical path initially validates text-only token execution while preserving the source’s semantic
architecture: 64 layers, 48 linear-attention layers, 16 full-attention layers, hidden size 5120,
24 query heads, 4 KV heads, head dimension 256, rotary dimension 64, intermediate size 17408,
vocabulary 248320, and native context 262144.

The linear path uses 16 key heads and 48 value heads, both with 128-wide heads, plus a four-token
convolution state. Its recurrent state is therefore distinct from the full-attention KV cache.

The derivative uses ModelOpt NVFP4 W4A4 weights with group size 16 and FP8 KV storage. The checked-in
[source manifest](../../frontends/qwen38/manifest.json) records metadata, tensor-inventory hash,
and complete local input hashes without checking model weights into the repository.
The converter classifies packed weights separately from scale records; only semantic parameter
records are handed to the frontend, while scale payloads remain storage-policy metadata.

## Validation

Run the provenance validator before any bulk conversion:

```sh
PYTHONPATH=python python3 - <<'PY'
from pathlib import Path
from superinfer.convert.qwen38 import validate_source

inventory = validate_source(
    Path('/srv/models/hf/Qwen3.8-27B-NVFP4-RTX5090-LMHead4'),
    upstream_revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
    derivative_revision='0cc27958cefbbe231782ec8511de8c4eb5233348',
)
print(inventory.manifest()['tensor_inventory_sha256'])
PY
```

The validator checks required config/tokenizer fields, the index/header tensor-name bijection,
positive shapes, safe offsets, the pinned layer schedule, and immutable source/file hashes. It parses
safetensors headers before streaming shard bytes for authentication; a mismatch fails before payload
materialization.

## Payload artifact checkpoint

The deterministic payload recipe streams the authenticated safetensors shards into a `.sinf` file
without holding the model in host memory:

```sh
PYTHONPATH=python python3 - <<'PY'
from pathlib import Path
from superinfer.convert.qwen38 import write_qwen38_payload_artifact

write_qwen38_payload_artifact(
    Path('/srv/models/hf/Qwen3.8-27B-NVFP4-RTX5090-LMHead4'),
    Path('build/evidence/qwen38-payload-v1.sinf'),
)
PY
PYTHONPATH=python python3 -m superinfer inspect build/evidence/qwen38-payload-v1-final-a.sinf --json
```

The final current-recipe checked local artifact is 18,766,778,520 bytes with SHA-256
`a8d4b2b398cc3458349cd6daee09a6f8e3776bc729b893291d30f28f1fba1573`; this regeneration includes
401 deterministic NVFP4 sidecar bindings in the manifest. Large-artifact inspection is
section-streamed and uses bounded memory.
Its manifest includes the validated NVFP4/FP8 quantization contract and reports layer-level
physical differential evidence below; full-model token generation remains pending.

## Layer differential checkpoints

The current branch has real-artifact CUDA evidence for both mixed attention families:

- layer 3 full attention plus gated MLP: `artifacts/S03/qwen38-layer3-artifact-differential.json`;
- layer 0 Gated-DeltaNet plus MLP across two state-continuing segments:
  `artifacts/S03/qwen38-gdn-layer0-artifact-differential.json`.

These are layer correctness checkpoints, not full-model or token-generation acceptance. The
fixtures intentionally retain explicit command boundaries and use GPU 0 only; full Qwen prefill,
logits, and greedy continuation remain S03-03 work.
