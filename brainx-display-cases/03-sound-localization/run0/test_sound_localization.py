import unittest

import brainunit as u
import numpy as np

import sound_localization as model


class SoundLocalizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ear, cls.detectors, cls.readout = model.simulate_itds()
        cls.labels = model.decode_directions(cls.readout)

    def test_every_ear_emits_one_spike(self):
        np.testing.assert_array_equal(np.asarray(self.ear), 1)

    def test_each_itd_selects_its_matching_detector(self):
        detector_counts = np.asarray(self.detectors)
        num_detectors = model.PREFERRED_DELAY_STEPS.shape[0]
        expected = np.eye(num_detectors, dtype=np.int32)
        np.testing.assert_array_equal(detector_counts, expected)

    def test_direction_follows_itd_sign(self):
        expected = np.array(
            ["RIGHT"] * 6 + ["CENTER"] + ["LEFT"] * 6
        )
        np.testing.assert_array_equal(self.labels, expected)

    def test_readout_is_left_right_symmetric(self):
        counts = np.asarray(self.readout)
        np.testing.assert_array_equal(counts[:, 0], counts[::-1, 1])
        self.assertEqual(counts[6].sum(), 0)

    def test_itds_are_sub_millisecond_time_quantities(self):
        max_itd_ms = np.max(np.abs(model.PREFERRED_ITDS.to_decimal(u.ms)))
        self.assertLess(max_itd_ms, 1.0)

    def test_non_time_itd_is_rejected(self):
        with self.assertRaises(u.UnitMismatchError):
            model.simulate_itds([1.0 * u.mV])


if __name__ == "__main__":
    unittest.main()
