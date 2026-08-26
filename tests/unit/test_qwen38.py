import json
import struct
import tempfile
import unittest
from pathlib import Path

from superinfer.artifact import inspect_artifact
from superinfer.convert.qwen38 import (
    Qwen38ValidationError,
    validate_source,
    write_qwen38_payload_artifact,
    write_qwen38_metadata_artifact,
)


def _config() -> dict[str, object]:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 16,
            "full_attention_interval": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "rms_norm_eps": 1e-6,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
    }


def _write_source(root: Path, *, indexed_name: str = "weight") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    (root / "generation_config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer.json").write_text(
        json.dumps({
            "model": {"type": "BPE", "vocab": {"x": 0}, "merges": []},
            "added_tokens": [{"content": "<|im_end|>", "id": 2}],
        }),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer", "model_max_length": 262144, "eos_token": "<|im_end|>"}),
        encoding="utf-8",
    )
    (root / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    (root / "vocab.json").write_text(json.dumps({"x": 0}), encoding="utf-8")
    (root / "merges.txt").write_text("", encoding="utf-8")
    (root / "hf_quant_config.json").write_text(
        json.dumps({
            "producer": {"name": "modelopt", "version": "fixture"},
            "quantization": {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "kv_cache_quant_algo": "FP8",
                "exclude_modules": [],
            },
        }),
        encoding="utf-8",
    )
    header = json.dumps(
        {indexed_name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode("utf-8")
    (root / "model-00001-of-00001.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0\0\0\0"
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {indexed_name: "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )


class Qwen38SourceTests(unittest.TestCase):
    def test_inventory_is_deterministic_and_authenticates_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root)
            first = validate_source(root, enforce_pinned=False)
            second = validate_source(root, enforce_pinned=False)
            self.assertEqual(first.manifest(), second.manifest())
            self.assertEqual(first.tensors[0].shape, (1,))
            self.assertEqual(first.tensors[0].data_end, 4)
            self.assertEqual(first.normalized_tensor_mapping()[0]["role"], "weight")
            self.assertIn("model-00001-of-00001.safetensors", first.file_hashes)
            self.assertEqual(first.manifest()["quantization"]["algorithm"], "NVFP4")

    def test_config_mismatch_fails_before_tensor_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root)
            config = _config()
            assert isinstance(config["text_config"], dict)
            config["text_config"]["hidden_size"] = 1
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(Qwen38ValidationError, r"config_mismatch \[text_config.hidden_size\]"):
                validate_source(root, enforce_pinned=False)

    def test_index_and_header_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root, indexed_name="index_name")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"different_name": "model-00001-of-00001.safetensors"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Qwen38ValidationError, "tensor_index_mismatch"):
                validate_source(root, enforce_pinned=False)

    def test_pinned_revision_gate_rejects_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root)
            with self.assertRaisesRegex(Qwen38ValidationError, "source_identity_mismatch"):
                validate_source(root, upstream_revision="1" * 40, derivative_revision="2" * 40)

    def test_invalid_metadata_encoding_has_stable_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root)
            (root / "config.json").write_bytes(b"{\xff")
            with self.assertRaisesRegex(Qwen38ValidationError, r"invalid_encoding \[config\]"):
                validate_source(root, enforce_pinned=False)

    def test_tokenizer_eos_contract_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root)
            (root / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "Qwen2Tokenizer", "model_max_length": 262144, "eos_token": "bad"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Qwen38ValidationError, r"tokenizer_mismatch \[tokenizer_config.eos_token\]"):
                validate_source(root, enforce_pinned=False)

    def test_quantization_contract_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root)
            quantization = {
                "producer": {"name": "modelopt", "version": "fixture"},
                "quantization": {
                    "quant_algo": "FP8",
                    "group_size": 16,
                    "kv_cache_quant_algo": "FP8",
                    "exclude_modules": [],
                },
            }
            (root / "hf_quant_config.json").write_text(json.dumps(quantization), encoding="utf-8")
            with self.assertRaisesRegex(Qwen38ValidationError, r"quantization_mismatch \[hf_quant_config.quantization.quant_algo\]"):
                validate_source(root, enforce_pinned=False)

    def test_quantization_scale_records_are_not_semantic_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root, indexed_name="layer.weight_scale")
            header = json.dumps(
                {"layer.weight_scale": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]}},
                separators=(",", ":"),
            ).encode("utf-8")
            (root / "model-00001-of-00001.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0")
            inventory = validate_source(root, enforce_pinned=False)
            self.assertEqual(inventory.tensors[0].role, "scale")

    def test_shard_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root)
            outside = root.parent / "outside.safetensors"
            outside.write_bytes((root / "model-00001-of-00001.safetensors").read_bytes())
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "../outside.safetensors"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Qwen38ValidationError, "invalid_tensor_index"):
                validate_source(root, enforce_pinned=False)

    def test_metadata_artifact_is_deterministic_and_declares_unavailable_kernels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root)
            first_path = root / "first.sinf"
            second_path = root / "second.sinf"
            write_qwen38_metadata_artifact(root, first_path, max_context=64, enforce_pinned=False)
            write_qwen38_metadata_artifact(root, second_path, max_context=64, enforce_pinned=False)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            summary = inspect_artifact(first_path)
            conversion = summary["manifest"]["conversion"]
            self.assertFalse(conversion["payload_included"])
            self.assertEqual(conversion["max_context"], 64)
            statuses = {entry["baseline_status"] for entry in conversion["operation_capabilities"]}
            self.assertEqual(statuses, {"executable", "unavailable"})
            self.assertGreater(conversion["memory_ledger_bytes"]["margin"], 0)

    def test_payload_artifact_streams_shards_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root)
            first_path = root / "first-payload.sinf"
            second_path = root / "second-payload.sinf"
            write_qwen38_payload_artifact(root, first_path, enforce_pinned=False)
            write_qwen38_payload_artifact(root, second_path, enforce_pinned=False)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            summary = inspect_artifact(first_path)
            self.assertEqual(summary["payload_bytes"], (root / "model-00001-of-00001.safetensors").stat().st_size)
            conversion = summary["manifest"]["conversion"]
            self.assertTrue(conversion["payload_included"])
            self.assertEqual(
                {entry["baseline_status"] for entry in conversion["operation_capabilities"]},
                {"executable", "unavailable"},
            )
            self.assertGreater(conversion["memory_ledger_bytes"]["margin"], 0)


if __name__ == "__main__":
    unittest.main()
