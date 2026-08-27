import json
import struct
import tempfile
import unittest
from pathlib import Path

from superinfer.convert.flash_next import (
    FlashNextContract,
    FlashNextValidationError,
    build_residency_options,
    blocked_source_evidence,
    classify_tensor_bytes,
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


class FlashNextLedgerTests(unittest.TestCase):
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
        self.assertFalse(evidence["local_candidate"]["usable_for_full_ledger"])

    def test_residency_rejects_over_budget_and_partitions_contiguously(self) -> None:
        layers = [("layer.0", 60), ("layer.1", 40), ("layer.2", 40)]
        options = build_residency_options(
            layers, {"routed_experts": 100, "ple": 10, "non_expert_text": 30},
            recipes={"fidelity_first": {"routed_experts": 1.0, "ple": 1.0, "non_expert_text": 1.0}},
            device_budget_bytes=100, headroom_bytes=0,
        )
        self.assertEqual(options[0]["partition"], [{"device": 0, "first_layer": 0, "last_layer": 1},
                                                       {"device": 1, "first_layer": 2, "last_layer": 2}])
        with self.assertRaisesRegex(FlashNextValidationError, "budget"):
            build_residency_options(
                layers, {"routed_experts": 100, "ple": 10, "non_expert_text": 30},
                recipes={"too_big": {"routed_experts": 2.0, "ple": 1.0, "non_expert_text": 1.0}},
                device_budget_bytes=100, headroom_bytes=0,
            )
