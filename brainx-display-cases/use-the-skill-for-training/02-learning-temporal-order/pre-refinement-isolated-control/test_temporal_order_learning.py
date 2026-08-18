import tempfile
import unittest
from pathlib import Path

import numpy as np

import brainunit as u

import temporal_order_learning as model


class TemporalOrderLearningTest(unittest.TestCase):
    def test_tone_templates_reverse_onsets(self):
        times, templates = model.make_order_templates()
        times_ms = np.asarray(times.to_decimal(u.ms))
        events = np.asarray(templates)

        ab_a = times_ms[np.flatnonzero(events[model.A_FIRST, :, 0])[0]]
        ab_b = times_ms[np.flatnonzero(events[model.A_FIRST, :, 1])[0]]
        ba_a = times_ms[np.flatnonzero(events[model.B_FIRST, :, 0])[0]]
        ba_b = times_ms[np.flatnonzero(events[model.B_FIRST, :, 1])[0]]

        self.assertLess(ab_a, ab_b)
        self.assertLess(ba_b, ba_a)
        self.assertTrue(
            np.isclose(ab_b - ab_a, model.INTER_TONE_DELAY.to_decimal(u.ms))
        )

    def test_circuit_learns_then_handles_reversed_order(self):
        result = model.run_experiment(n_trials=18)

        after_ab = np.asarray(model.predicted_orders(result["after_ab_outputs"]))
        after_ba = np.asarray(model.predicted_orders(result["after_ba_outputs"]))
        weights = np.asarray(result["final_weights"])

        self.assertEqual(after_ab[model.A_FIRST], model.A_FIRST)
        self.assertEqual(after_ba[model.B_FIRST], model.B_FIRST)
        self.assertGreater(
            weights[model.A_FIRST, model.A_FIRST],
            weights[model.A_FIRST, model.B_FIRST],
        )
        self.assertGreater(
            weights[model.B_FIRST, model.B_FIRST],
            weights[model.B_FIRST, model.A_FIRST],
        )
        self.assertGreaterEqual(weights.min(), model.WEIGHT_MIN)
        self.assertLessEqual(weights.max(), model.WEIGHT_MAX)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = model.plot_experiment(
                result, Path(temp_dir) / "relearning.png"
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
