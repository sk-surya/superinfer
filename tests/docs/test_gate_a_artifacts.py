from pathlib import Path
import unittest


ROOT = Path(__file__).parents[2]
ARTIFACTS = {
    "studio": ROOT / "docs/understanding/gate-a/index.html",
    "field_guide": ROOT / "docs/understanding/gate-a/field-guide.html",
}


class GateAArtifactTest(unittest.TestCase):
    def test_both_gate_a_artifacts_exist_and_are_self_contained(self):
        for name, path in ARTIFACTS.items():
            with self.subTest(artifact=name):
                self.assertTrue(path.is_file(), path)
                html = path.read_text(encoding="utf-8")
                self.assertIn("Gate A", html)
                self.assertIn("Semantic IR", html)
                self.assertIn("Lowered IR", html)
                self.assertIn("Physical Plan", html)
                self.assertNotIn("https://", html)
                self.assertNotIn("http://", html)

    def test_studio_contains_required_ownership_exercises(self):
        html = ARTIFACTS["studio"].read_text(encoding="utf-8")
        for marker in (
            "semantic operation experiment",
            "hidden [2,4]",
            "five ownership questions",
            "Qwen",
            "Export my answers",
        ):
            self.assertIn(marker, html)

    def test_field_guide_is_print_ready(self):
        html = ARTIFACTS["field_guide"].read_text(encoding="utf-8")
        self.assertIn("@media print", html)
        self.assertIn("Exactly three files", html)
        self.assertIn("Exactly five questions", html)


if __name__ == "__main__":
    unittest.main()
