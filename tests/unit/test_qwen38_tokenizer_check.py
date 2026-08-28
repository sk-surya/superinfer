import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class Qwen38TokenizerCheckTests(unittest.TestCase):
    def test_input_failure_writes_bounded_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tokenizer.json"
            process = subprocess.run(
                [sys.executable, "tools/qwen38_tokenizer_check.py", "--model-dir", "/missing/model",
                 "--corpus", "/missing/corpus.json", "--output", str(output)],
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(process.returncode, 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "fail")
            self.assertIn("error", report)


if __name__ == "__main__":
    unittest.main()
