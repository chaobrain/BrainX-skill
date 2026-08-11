import unittest

import numpy as np

import brainstate
import brainunit as u

import head_direction_compass as compass


class CompassTests(unittest.TestCase):
    def test_circular_error_wraps_at_north(self):
        error = compass.circular_error_deg(np.array([355.0, 5.0, 180.0]), 0.0)
        np.testing.assert_allclose(error, [5.0, 5.0, 180.0])


    def test_ring_weights_are_rotation_equivariant_and_unitful(self):
        hold, positive, negative = compass.ring_weights(12)
        self.assertIs(hold.unit.dim, u.mS.dim)
        np.testing.assert_allclose(
            np.asarray(hold.to_decimal(u.mS))[1],
            np.roll(np.asarray(hold.to_decimal(u.mS))[0], 1),
            atol=1e-7,
        )
        self.assertEqual(positive.shape, (12, 12))
        self.assertEqual(negative.shape, (12, 12))
        np.testing.assert_allclose(
            np.asarray(positive.to_decimal(u.mS)),
            -np.asarray(negative.to_decimal(u.mS)),
            atol=1e-7,
        )


    def test_protocol_integrates_to_quarter_turn(self):
        with brainstate.environ.context(dt=compass.DT):
            times, cue, velocity, lesion = compass.build_protocols()
        turn_deg = np.degrees(
            np.asarray(velocity.to_decimal(u.radian / u.second)).sum()
            * compass.DT.to_decimal(u.second)
        )
        self.assertEqual(times.shape, cue.shape)
        self.assertEqual(times.shape, velocity.shape)
        self.assertEqual(times.shape, lesion.shape)
        self.assertTrue(np.isclose(turn_deg, 90.0, atol=0.2))
        self.assertTrue(
            np.all(np.asarray(lesion[: int(compass.LESION_ONSET / compass.DT)]) == 0.0)
        )


    def test_population_decoder_recovers_cardinal_headings(self):
        activity = np.eye(4)
        headings, resultant, total = compass.decode_population(
            activity,
            np.asarray(compass.preferred_angles(4)),
        )
        np.testing.assert_allclose(headings, [0.0, 90.0, 180.0, 270.0], atol=1e-5)
        np.testing.assert_allclose(resultant, 1.0)
        np.testing.assert_allclose(total, 1.0)

    def test_default_sweep_tracks_and_has_mixed_lesion_outcomes(self):
        analysis = compass.analyze(compass.run_sweep())
        self.assertTrue(np.all(analysis["control_valid"]))
        self.assertLessEqual(
            max(row["control_turn_rmse_deg"] for row in analysis["rows"]),
            compass.MAX_CONTROL_TURN_RMSE_DEG,
        )
        recovered = int(np.sum(analysis["recovered"]))
        self.assertGreater(recovered, 0)
        self.assertLess(recovered, len(analysis["rows"]))


if __name__ == "__main__":
    unittest.main()
