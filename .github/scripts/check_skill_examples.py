"""Run focused Markdown examples with BrainX 2026.10.25 on CPU."""

import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ROOT / "skills" / "package-skills"
PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)
FENCED_BLOCK = re.compile(r"^```[^\n]*\n.*?^```[^\n]*", re.MULTILINE | re.DOTALL)


def run_examples(path, scope, heading=None):
    content = path.read_text(encoding="utf-8")
    line_offset = 0
    if heading:
        headings = FENCED_BLOCK.sub(lambda block: re.sub(r"[^\n]", " ", block[0]), content)
        start = re.search(r"^(#+) " + re.escape(heading) + "$", headings, re.MULTILINE)
        assert start, f"Missing section {heading!r} in {path}"
        line_offset = content.count("\n", 0, start.end())
        end = re.search(r"^#{1," + str(len(start[1])) + r"} ", headings[start.end():], re.MULTILINE)
        content = content[start.end():]
        if end:
            content = content[:end.start()]
    blocks = list(PYTHON_BLOCK.finditer(content))
    assert blocks, f"No Python examples in {path}"
    for block in blocks:
        preceding_lines = line_offset + content.count("\n", 0, block.start(1))
        exec(compile("\n" * preceding_lines + block[1], str(path), "exec"), scope)


class SkillExamples(unittest.TestCase):
    def test_brainstate_training(self):
        import brainstate
        import jax.numpy as jnp

        scope = {"brainstate": brainstate, "nn": brainstate.nn, "jnp": jnp}
        run_examples(PACKAGES / "brainstate/SKILL.md", scope, "5. State-aware transforms")
        params = next(iter(scope["params"].values()))
        before = params.value
        loss = scope["train_step"]()
        self.assertTrue(jnp.isfinite(loss))
        for name in ("weight", "bias"):
            self.assertFalse(jnp.array_equal(params.value[name], before[name]), name)

    def test_braincell_recording(self):
        import brainunit as u
        import numpy as np

        scope = {}
        run_examples(PACKAGES / "braincell/SKILL.md", scope, "Build and run a multicompartment cell")
        block = scope["result"].samples["soma_v"]
        self.assertEqual(block.values.shape, (1000, 1))
        self.assertTrue(np.isfinite(block.values.to_decimal(u.mV)).all())
        np.testing.assert_allclose(np.diff(block.time.to_decimal(u.ms)), 0.1, atol=1e-5)

    def test_braincell_topology(self):
        import matplotlib.pyplot as plt

        path = PACKAGES / "braincell/references/multicompartment/topology-building-and-visualization.md"
        with tempfile.TemporaryDirectory() as temp:
            swc = Path(temp) / "neuron.swc"
            swc.write_text("1 1 0 0 0 10 -1\n2 3 0 12 0 1 1\n3 3 0 32 0 1 2\n", encoding="ascii")
            # The visualization reference takes a supplied SWC reconstruction.
            previous = Path.cwd()
            try:
                os.chdir(temp)
                with patch.object(plt, "show"):
                    run_examples(path, {})
                for number in plt.get_fignums():
                    plt.figure(number).canvas.draw()
                self.assertGreaterEqual(len(plt.get_fignums()), 4)
            finally:
                plt.close("all")
                os.chdir(previous)

    def test_braincell_network(self):
        path = PACKAGES / "braincell/references/multicompartment/network-workflow.md"
        run_examples(path, {})

    def test_braincell_sampling_schedule(self):
        import braincell
        from braincell.filter import RootLocation
        import brainunit as u

        branch = braincell.Branch.from_lengths(lengths=[20.] * u.um, radii=[5., 5.] * u.um)
        cell = braincell.Cell(braincell.Morphology.from_root(branch))
        path = PACKAGES / "braincell/references/multicompartment/probe-reference.md"
        run_examples(path, {"braincell": braincell, "cell": cell, "RootLocation": RootLocation, "u": u})

    def test_brainstate_environment(self):
        from concurrent.futures import ThreadPoolExecutor
        import brainstate
        import brainunit as u

        env = brainstate.environ.EnvironmentState()
        brainstate.environ.set(dt=0.5 * u.ms, env=env)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with brainstate.environ.context(dt=2.0 * u.ms, env=env):
                self.assertEqual(brainstate.environ.get_dt(env=env), 2.0 * u.ms)
                observed = pool.submit(brainstate.environ.get_dt, env=env).result()
                self.assertEqual(observed, 0.5 * u.ms)
        self.assertEqual(brainstate.environ.get_dt(env=env), 0.5 * u.ms)


if __name__ == "__main__":
    unittest.main()
