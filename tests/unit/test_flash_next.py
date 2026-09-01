import hashlib
import json
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from superinfer.convert.flash_next import (
    FlashNextContract,
    FlashNextValidationError,
    TensorRecord,
    build_residency_options,
    blocked_source_evidence,
    classify_tensor_bytes,
    estimate_runtime_state_and_workspace,
    inspect_gguf_source,
    official_contract,
    validate_source,
)


def _contract(**overrides: object) -> FlashNextContract:
    values = {
        "model_type": "flash_next",
        "layer_count": 48,
        "expert_count": 128,
        "top_k": 8,
        "upstream_repository": "example/upstream",
        "upstream_revision": "a" * 40,
        "reference_repository": "example/reference",
        "reference_revision": "b" * 40,
        "qsa_config": {"block_size": 128, "indexer": "qsa.index"},
    }
    values.update(overrides)
    return FlashNextContract(**values)


def _write_source(root: Path, *, config_overrides: dict[str, object] | None = None,
                  include_ple: bool = True, include_qsa: bool = True) -> None:
    config: dict[str, object] = {
        "model_type": "flash_next", "num_hidden_layers": 48,
        "num_experts": 128, "num_experts_per_tok": 8,
        "ple": {"table": "ple.table"} if include_ple else {},
        "qsa": {"block_size": 128, "indexer": "qsa.index"} if include_qsa else {},
    }
    config.update(config_overrides or {})
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "provenance.json").write_text(json.dumps({
        "upstream_repository": "example/upstream", "upstream_revision": "a" * 40,
        "reference_repository": "example/reference", "reference_revision": "b" * 40,
    }), encoding="utf-8")
    tensors = {
        "model.layers.0.mlp.experts.0.w":
            {"dtype": "U8", "shape": [2, 8], "data_offsets": [0, 16]},
        "ple.table": {"dtype": "U8", "shape": [4, 8], "data_offsets": [16, 48]},
        "model.embed_tokens.weight":
            {"dtype": "BF16", "shape": [2, 8], "data_offsets": [48, 80]},
    }
    header = json.dumps(tensors, separators=(",", ":")).encode()
    (root / "model-00001-of-00001.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0" * 80
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model-00001-of-00001.safetensors" for name in tensors}}),
        encoding="utf-8",
    )


def _write_gguf(path: Path, tensors: list[tuple[str, tuple[int, ...], int]], *,
                split_no: int, split_count: int) -> None:
    def string(value: str) -> bytes:
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    def value(value: object) -> bytes:
        if isinstance(value, str):
            return struct.pack("<I", 8) + string(value)
        if isinstance(value, int):
            return struct.pack("<I", 4) + struct.pack("<i", value)
        raise TypeError(value)

    metadata = [
        ("general.architecture", "qwen4exp"),
        ("general.alignment", 32),
        ("split.no", split_no),
        ("split.count", split_count),
    ]
    tensor_info = bytearray()
    payload_size = 0
    for name, shape, tensor_type in tensors:
        tensor_info += string(name)
        tensor_info += struct.pack("<I", len(shape))
        tensor_info += struct.pack("<" + "Q" * len(shape), *shape)
        tensor_info += struct.pack("<I Q", tensor_type, payload_size)
        payload_size += {0: 4 * 4, 7: 24, 22: 82}[tensor_type]
    kv = b"".join(string(name) + value(item) for name, item in metadata)
    header = struct.pack("<4s I Q Q", b"GGUF", 3, len(tensors), len(metadata)) + kv + tensor_info
    data_offset = (len(header) + 31) // 32 * 32
    path.write_bytes(header + b"\0" * (data_offset - len(header)) + b"\0" * payload_size)


class FlashNextContractTests(unittest.TestCase):
    def test_rejects_wrong_model_type_layer_expert_and_top_k(self) -> None:
        for key, value, field in (("model_type", "other", "model_type"),
                                  ("num_hidden_layers", 47, "layer_count"),
                                  ("num_experts", 64, "expert_count"),
                                  ("num_experts_per_tok", 4, "top_k")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_source(root, config_overrides={key: value})
                with self.assertRaisesRegex(FlashNextValidationError, field):
                    validate_source(root, _contract())

    def test_rejects_missing_ple_unexpected_qsa_and_changed_revision(self) -> None:
        cases = [({"include_ple": False}, "ple"), ({"include_qsa": False}, "qsa"),
                 ({"config_overrides": {"qsa": {"block_size": 64, "indexer": "qsa.index"}}}, "qsa")]
        for kwargs, field in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_source(root, **kwargs)
                with self.assertRaisesRegex(FlashNextValidationError, field):
                    validate_source(root, _contract())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root)
            with self.assertRaisesRegex(FlashNextValidationError, "revision"):
                validate_source(root, _contract(upstream_revision="c" * 40))

    def test_inventory_is_metadata_only_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root)
            first = validate_source(root, _contract())
            second = validate_source(root, _contract())
            self.assertEqual(first.canonical_json(), second.canonical_json())
            self.assertEqual(len(first.tensors), 3)
            self.assertEqual(first.tensors[0].nbytes, 32)

    def test_rejects_shard_path_traversal_and_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root)
            index_path = root / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            index["weight_map"]["ple.table"] = "../outside.safetensors"
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(FlashNextValidationError, "path"):
                validate_source(root, _contract())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root)
            with self.assertRaisesRegex(FlashNextValidationError, "hash"):
                validate_source(root, _contract(expected_config_sha256="0" * 64,
                                                expected_index_sha256=hashlib.sha256(
                                                    (root / "model.safetensors.index.json").read_bytes()).hexdigest(),
                                                expected_shard_count=1,
                                                expected_tensor_count=3))


class FlashNextLedgerTests(unittest.TestCase):
    def test_inspects_complete_gguf_shards_and_classifies_packed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_gguf(root / "model-00001-of-00002.gguf", [
                ("token_embd.weight", (4,), 0),
                ("blk.0.ffn_gate_exps.weight", (256,), 22),
            ], split_no=0, split_count=2)
            _write_gguf(root / "model-00002-of-00002.gguf", [
                ("per_layer_token_embd.weight", (32,), 7),
                ("blk.0.ffn_gate_shexp.weight", (4,), 0),
            ], split_no=1, split_count=2)
            inventory = inspect_gguf_source(root)
            self.assertEqual(inventory["shard_count"], 2)
            self.assertEqual(inventory["tensor_count"], 4)
            self.assertEqual(inventory["category_bytes"]["embedding_lm_head"], 16)
            self.assertEqual(inventory["category_bytes"]["ple"], 24)
            self.assertEqual(inventory["category_bytes"]["routed_experts"], 82)
            self.assertEqual(inventory["category_bytes"]["shared_experts"], 16)
            self.assertEqual(inventory["tensor_payload_bytes"], 138)
            self.assertEqual(inventory["split_numbers"], [0, 1])
            self.assertEqual(inventory["metadata"]["general.architecture"], "qwen4exp")

    def test_gguf_inventory_rejects_missing_shard_and_unknown_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_gguf(root / "model-00001-of-00002.gguf", [("mystery", (4,), 0)],
                        split_no=0, split_count=2)
            with self.assertRaisesRegex(FlashNextValidationError, "shards"):
                inspect_gguf_source(root)
            _write_gguf(root / "model-00002-of-00002.gguf", [], split_no=1, split_count=2)
            with self.assertRaisesRegex(FlashNextValidationError, "unclassified"):
                inspect_gguf_source(root)

    def test_runtime_state_and_workspace_formula_is_deterministic(self) -> None:
        config = {
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "full_attention",
                             "linear_attention", "full_attention"],
            "hidden_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 3,
            "linear_conv_kernel_dim": 5,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 6,
        }
        first = estimate_runtime_state_and_workspace(
            config, context_length=10, batch_size=2, activation_dtype_bytes=2,
            kv_dtype_bytes=2, state_dtype_bytes=4,
        )
        second = estimate_runtime_state_and_workspace(
            config, context_length=10, batch_size=2, activation_dtype_bytes=2,
            kv_dtype_bytes=2, state_dtype_bytes=4,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["kv_state_bytes"], 2 * 10 * 2 * 2 * 2 * 4 * 2)
        self.assertEqual(first["recurrent_state_bytes"], 2 * 2 * 4 * 2 * 3 * 4)
        self.assertEqual(first["convolution_state_bytes"], 2 * 2 * (2 * 2 * 2 + 4 * 3) * 5 * 2)
        self.assertGreater(first["workspace_bytes"], 0)
        self.assertEqual(first["total_bytes"], sum(
            first[key] for key in ("kv_state_bytes", "recurrent_state_bytes",
                                   "convolution_state_bytes", "workspace_bytes")
        ))

    def test_runtime_state_formula_rejects_invalid_context_and_batch(self) -> None:
        config = {"num_hidden_layers": 1, "layer_types": ["full_attention"]}
        for kwargs in ({"context_length": 0}, {"context_length": 4, "batch_size": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                    FlashNextValidationError, "runtime_shape"):
                estimate_runtime_state_and_workspace(config, **kwargs)

    def test_categories_reconcile_and_order_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root)
            inventory = validate_source(root, _contract())
            ledger = classify_tensor_bytes(inventory)
            self.assertEqual(list(ledger), [
                "embedding_lm_head", "mtp", "non_expert_text", "ple",
                "routed_experts", "router_indexer", "shared_experts", "vision",
            ])
            self.assertEqual(sum(ledger.values()), sum(t.nbytes for t in inventory.tensors))

    def test_unknown_tensor_family_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root)
            source = validate_source(root, _contract())
            unknown = replace(source, tensors=source.tensors + (
                TensorRecord("mystery.tensor", "U8", (1,), "model-00001-of-00001.safetensors", 0, 1),
            ))
            with self.assertRaisesRegex(FlashNextValidationError, "unclassified"):
                classify_tensor_bytes(unknown)

    def test_official_contract_is_pinned_to_observed_metadata(self) -> None:
        contract = official_contract()
        self.assertEqual(contract.model_type, "qwen4_exp")
        self.assertEqual((contract.layer_count, contract.expert_count, contract.top_k), (48, 512, 10))
        self.assertEqual(len(contract.upstream_revision), 40)
        self.assertEqual(contract.qsa_config["indexer_budget"], 2048)

    def test_blocker_preserves_official_identity_without_capacity_claims(self) -> None:
        evidence = blocked_source_evidence()
        self.assertEqual(evidence["status"], "blocked")
        self.assertEqual(evidence["official_identity"]["expected_safetensors_shards"], 131)
        self.assertEqual(evidence["official_identity"]["expected_tensor_entries"], 1658)
        self.assertFalse(evidence["local_candidate"]["usable_for_full_ledger"])

    def test_residency_rejects_over_budget_and_partitions_contiguously(self) -> None:
        layers = [("layer.0", 60), ("layer.1", 40), ("layer.2", 40)]
        options = build_residency_options(
            layers, {"routed_experts": 100, "ple": 10, "non_expert_text": 30},
            recipes={"fidelity_first": {"routed_experts": 1.0, "ple": 1.0, "non_expert_text": 1.0}},
            device_budget_bytes=100, headroom_bytes=0,
        )
        self.assertEqual(
            [(item["device"], item["first_layer"], item["last_layer"], item["projected_bytes"])
             for item in options[0]["partition"]],
            [(0, 0, 1, 100), (1, 2, 2, 40)],
        )
        with self.assertRaisesRegex(FlashNextValidationError, "budget"):
            build_residency_options(
                layers, {"routed_experts": 100, "ple": 10, "non_expert_text": 30},
                recipes={"too_big": {"routed_experts": 2.0, "ple": 1.0, "non_expert_text": 1.0}},
                device_budget_bytes=100, headroom_bytes=0,
            )
        with self.assertRaisesRegex(FlashNextValidationError, "layer"):
            build_residency_options(
                [("layer.0", 150), ("layer.1", 50)],
                {"routed_experts": 200},
                recipes={"oversized_layer": {"routed_experts": 1.0}},
                device_budget_bytes=100, headroom_bytes=0,
            )
