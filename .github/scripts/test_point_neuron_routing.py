"""Check the published Codex routes for focused and end-to-end tasks."""

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
LEGACY = r"(?i)\bbrainpy[- ._]state\b"


class PointNeuronRoutingTests(unittest.TestCase):
    def test_skill_discovery_uses_brainspike(self):
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        reviewer = json.loads(
            (ROOT / "mcp-servers/codex/skills.json").read_text(encoding="utf-8")
        )
        for name, skills in (("installer", manifest["skills"]), ("reviewer", reviewer)):
            with self.subTest(discovery=name):
                self.assertIn("brainspike", skills)
                self.assertNotIn("brainpy-state", skills)

    def test_active_point_neuron_routes_use_brainspike(self):
        routes = {
            "brainx-general-guard": (
                r"(?im)^\|\s*Point neurons\b[^|\n]*\|\s*`?BrainSpike\b"
            ),
            "brainx-modeling-loop": (
                r"(?im)^\s*-\s+`brainspike`\s+for\s+(?:all\s+)?point[- ]neurons?\b"
            ),
        }
        for name, route in routes.items():
            with self.subTest(skill=name):
                skill = (ROOT / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
                self.assertRegex(skill, route)
                self.assertNotRegex(skill, LEGACY)


if __name__ == "__main__":
    unittest.main()
