import unittest

import numpy as np
import jax.numpy as jnp
import brainunit as u

from temporal_order_learning import (
    FIRST_ONSET_STEP,
    SECOND_ONSET_STEP,
    TRIAL_STEPS,
    build_trial_batch,
    run_experiment,
)


class TemporalOrderLearningTest(unittest.TestCase):
    def test_vmapped_trial_encoding_reverses_the_sensory_channels(self):
        orders = jnp.array([0, 1])
        batch = build_trial_batch(
            orders,
            learning=jnp.array([True, True]),
            teacher_enabled=jnp.array([True, True]),
        )
        current = np.asarray(batch.tone_current.to_decimal(u.mA)).reshape(
            2, TRIAL_STEPS, 2
        )

        self.assertGreater(current[0, FIRST_ONSET_STEP, 0], 0.0)
        self.assertGreater(current[0, SECOND_ONSET_STEP, 1], 0.0)
        self.assertGreater(current[1, FIRST_ONSET_STEP, 1], 0.0)
        self.assertGreater(current[1, SECOND_ONSET_STEP, 0], 0.0)

    def test_teacher_free_output_relearns_after_order_reversal(self):
        result = run_experiment(phase_trials=10)

        # After the first phase AB evokes only the A-first population.
        self.assertGreater(result.phase1_probe_spikes[0, 0], 0)
        self.assertEqual(result.phase1_probe_spikes[0, 1], 0)
        self.assertEqual(result.phase1_probe_spikes[1].sum(), 0)

        # After reversing the tones, BA independently evokes B-first spikes.
        self.assertGreater(result.phase2_probe_spikes[1, 1], 0)
        self.assertEqual(result.phase2_probe_spikes[1, 0], 0)


if __name__ == "__main__":
    unittest.main()
