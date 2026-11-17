"""Direct Simulator experiments, including explicit negative-control boundaries."""
import importlib.metadata
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "skills/package-skills/brainspike/scripts"))
from train_nest_simulator import experiment, rollout
import jax
import jax.numpy as jnp
import numpy as np


class SimulatorTrainingTests(unittest.TestCase):
    def test_current_gradient_matches_finite_difference(self):
        f = lambda current: rollout(jnp.asarray(0.5), current=current)[0][-1]
        value, gradient = jax.value_and_grad(f)(jnp.asarray(0.5))
        finite_difference = (f(0.5001)-f(0.4999))/0.0002
        self.assertGreater(abs(float(gradient)), 0)
        np.testing.assert_allclose(gradient, finite_difference, rtol=1e-6, atol=1e-8)
        print("Current gradient:", float(gradient), "finite difference:", float(finite_difference))

    def test_dense_and_sparse_gradients_and_fresh_reset(self):
        outcomes = []
        for comm in ("dense", "sparse"):
            f = lambda weight: rollout(weight, comm)[0][-1]
            value, gradient = jax.value_and_grad(f)(jnp.asarray(0.4))
            finite_difference = (f(0.4001)-f(0.3999))/0.0002
            self.assertGreater(abs(float(gradient)), 0)
            np.testing.assert_allclose(gradient, finite_difference, rtol=1e-6, atol=1e-8)
            np.testing.assert_array_equal(rollout(0.4, comm)[0], rollout(0.4, comm)[0])
            outcomes.append((float(value), float(gradient)))
        np.testing.assert_allclose(outcomes[0], outcomes[1], rtol=1e-10, atol=1e-10)
        print("Dense/sparse voltage and weight gradient:", outcomes)

    def test_dense_and_sparse_weight_learning(self):
        for comm in ("dense", "sparse"):
            with self.subTest(comm=comm):
                result = experiment(comm)
                print("TRAINING", json.dumps(result), flush=True)
                self.assertTrue(all(np.isfinite(result[k]) for k in
                    ["initial_loss", "final_loss", "learned_weight", "autodiff_gradient"]))
                self.assertLess(result["final_loss"], result["initial_loss"]*1e-4)
                self.assertLess(abs(result["learned_weight"]-result["target_weight"]), 0.01)
                np.testing.assert_allclose(result["autodiff_gradient"], result["finite_difference_gradient"], rtol=1e-6)

    def test_hard_recorded_spike_count_has_zero_gradient(self):
        # Forward firing alone must not be misreported as surrogate-gradient learning.
        count = lambda current: jnp.sum(rollout(jnp.asarray(.5), current=current, duration=20)[1].astype(jnp.float64))
        value, gradient = jax.value_and_grad(count)(jnp.asarray(10.))
        self.assertGreater(float(value), 0)
        self.assertEqual(float(gradient), 0)
        print("Boundary: hard recorded spike count",float(value),"has gradient",float(gradient))

    def test_whole_builder_outer_jit_rejects_host_time_validation(self):
        with self.assertRaises(jax.errors.TracerBoolConversionError):
            jax.jit(lambda w: rollout(w)[0][-1])(jnp.asarray(0.4))
        print("Boundary: keep multi-spike-generator graph construction outside outer JIT")


if __name__ == "__main__":
    print(json.dumps({p:importlib.metadata.version(p) for p in ["brainspikes", "brainpy", "brainstate", "jax"]}))
    unittest.main(verbosity=2)
