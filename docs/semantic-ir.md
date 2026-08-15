# Semantic IR contract

Semantic IR is the model-meaning boundary. Its durable values are tensor specifications,
symbolic/static shapes, operation kinds and attributes, entry-point signatures, and explicit state
edges. The vocabulary is model-independent: attention variants, gated dense FFN, MoE route/top-k/
expert/combine, embedding, normalization, RoPE, residual, LM head, decode logits, and sampling
inputs are represented as operations rather than model-family names.

The builder is transactional. It rejects empty or duplicate names and caps attacker-controlled
object counts. `build()` verifies tensor definitions/uses, producer topology, entry-point/state
references, attention head divisibility and RoPE constraints, and MoE expert/top-k constraints
before returning an immutable module. Failures are typed `Status` values with a semantic-builder
context; no partial graph escapes.

Canonical dumps sort tensors, operations, and entry points by stable names and refer to tensors by
those names. They do not contain source paths, addresses, provider identifiers, or physical
storage decisions. This makes construction-order changes reviewable and keeps Semantic IR
independent of Lowered IR and Physical Plan.

