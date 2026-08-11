import unittest

import numpy as np

from temporal_order_learning import run_experiment


class TemporalOrderLearningTest(unittest.TestCase):
    def test_circuit_acquires_and_reverses_temporal_order(self):
        results = run_experiment(trials_per_phase=24, seed=3)

        self.assertLessEqual(results["initial_accuracy"], 0.6)
        self.assertGreaterEqual(results["acquired_accuracy"], 0.9)
        self.assertLessEqual(results["switch_accuracy"], 0.2)
        self.assertGreaterEqual(results["reversed_accuracy"], 0.9)

        acquired = results["acquired_weights"]
        reversed_weights = results["reversed_weights"]
        self.assertGreater(acquired[0, 0], acquired[0, 1])
        self.assertGreater(acquired[1, 1], acquired[1, 0])
        self.assertGreater(reversed_weights[0, 1], reversed_weights[0, 0])
        self.assertGreater(reversed_weights[1, 0], reversed_weights[1, 1])

        detector_counts = results["acquired_evaluation"].detector_spikes.sum(axis=1)
        evaluation_orders = np.arange(detector_counts.shape[0]) % 2
        self.assertTrue(np.all(detector_counts.argmax(axis=1) == evaluation_orders))


if __name__ == "__main__":
    unittest.main()
