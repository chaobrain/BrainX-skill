import unittest

import brainunit as u
import jax.numpy as jnp
import numpy as np

from sound_localization import (
    EVALUATION_ITDS,
    assert_delay_convention,
    decode_directions,
    simulate_itds,
)


class SoundLocalizationTest(unittest.TestCase):
    def test_delay_line_uses_exact_integer_taps(self):
        assert_delay_convention()

    def test_localizer_classifies_both_sides_and_center(self):
        ear_counts, detector_counts, readout_counts = simulate_itds()
        directions = decode_directions(detector_counts, readout_counts)
        itds_ms = np.asarray(EVALUATION_ITDS.to_decimal(u.ms))
        expected = np.where(
            itds_ms < 0.0,
            "RIGHT",
            np.where(itds_ms > 0.0, "LEFT", "CENTER"),
        )

        np.testing.assert_array_equal(np.asarray(ear_counts), 1)
        np.testing.assert_array_equal(directions, expected)

    def test_itds_require_time_units_and_supported_range(self):
        with self.assertRaises(u.UnitMismatchError):
            simulate_itds(jnp.asarray([1.0]) * u.mV)
        with self.assertRaisesRegex(ValueError, "within"):
            simulate_itds(jnp.asarray([0.61]) * u.ms)


if __name__ == "__main__":
    unittest.main()
