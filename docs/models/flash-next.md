# Flash-Next research status

S03F-01 is currently blocked at source qualification. The checkout contains the approved
architecture amendment and plan, but no exact Flash-Next `config.json`, safetensors index/shards,
or pinned reference implementation/revisions. Nearby Qwen3.8-DFlash2 and Qwen3-Coder-Next GGUF
files are explicitly excluded because they do not establish the named model contract.

The research utility in `superinfer.convert.flash_next` is intentionally fail-closed. A caller must
supply the exact model/reference repositories and 40-character revisions, and the source must
provide model type, layer/expert/top-k fields, PLE metadata, QSA metadata, and a consistent
safetensors index/header inventory. It reads headers and hashes metadata; it does not load tensor
payloads.

No semantic or storage fact is pinned by this document. Consequently there are no exact tensor
bytes, category totals, quantization quality results, or full-expert-residency claims. The canonical
blocked evidence is `artifacts/S03F/flash-next-source-evidence.json`. S03F-02 remains blocked until
S03 closes and the missing source/reference evidence is supplied.
