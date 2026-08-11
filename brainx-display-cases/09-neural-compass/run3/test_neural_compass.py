import unittest

import numpy as np

from neural_compass import (
    ALIGNMENT_THRESHOLD_RAD,
    MIN_BUMP_STRENGTH,
    circular_error,
    classify_lesion_outcomes,
    decode_activity,
    wrap_angle,
)


class CircularAnalysisTests(unittest.TestCase):
    def test_wrap_and_error_cross_branch_cut(self):
        self.assertAlmostEqual(wrap_angle(np.deg2rad(190.0)), np.deg2rad(-170.0))
        error = circular_error(np.deg2rad(-179.0), np.deg2rad(179.0))
        self.assertAlmostEqual(error, np.deg2rad(2.0))

    def test_population_vector_decodes_known_heading(self):
        preferred = np.linspace(-np.pi, np.pi, 48, endpoint=False)
        target = np.deg2rad(60.0)
        activity = np.exp(-0.5 * (wrap_angle(preferred - target) / 0.3) ** 2)
        decoded, strength = decode_activity(activity, preferred)
        self.assertLess(circular_error(decoded, target), np.deg2rad(0.5))
        self.assertGreater(strength, MIN_BUMP_STRENGTH)

    def test_recovery_requires_departure_then_sustained_return(self):
        control = np.zeros((8, 3))
        lesion = np.zeros((8, 3))
        lesion[2:5, 1] = 2.0 * ALIGNMENT_THRESHOLD_RAD
        lesion[:, 2] = 2.0 * ALIGNMENT_THRESHOLD_RAD
        strength = np.ones((8, 3))

        labels, departure, sustained, _ = classify_lesion_outcomes(
            control,
            lesion,
            strength,
            lesion_on_index=2,
            sustained_steps=3,
        )
        self.assertEqual(labels.tolist(), ["spared", "recovered", "failed"])
        self.assertEqual(departure.tolist(), [False, True, True])
        self.assertEqual(sustained.tolist(), [True, True, False])

    def test_low_relative_activity_is_a_failure(self):
        control = np.zeros((6, 2))
        lesion = np.zeros((6, 2))
        strength = np.ones((6, 2))
        relative_activity = np.ones((6, 2))
        relative_activity[2:, 1] = 0.1

        labels, departure, sustained, _ = classify_lesion_outcomes(
            control,
            lesion,
            strength,
            lesion_on_index=2,
            sustained_steps=2,
            relative_activity=relative_activity,
        )
        self.assertEqual(labels.tolist(), ["spared", "failed"])
        self.assertEqual(departure.tolist(), [False, True])
        self.assertEqual(sustained.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
