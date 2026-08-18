import unittest

import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp

import seizure_recruitment as model


class SeizureRecruitmentTest(unittest.TestCase):
    def test_delay_phase_convention(self):
        with brainstate.environ.context(dt=1.0 * u.ms):
            delay = brainstate.nn.Delay(
                jnp.zeros(()),
                time=3.0 * u.ms,
                init=braintools.init.Constant(0.0),
            )
            brainstate.nn.init_all_states(delay)

            def delayed_impulse(value):
                delay.update(value)
                return delay.retrieve_at_step(jnp.asarray(3, dtype=jnp.int32))

            observed = brainstate.transform.for_loop(
                delayed_impulse,
                jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
            )

        expected = jnp.asarray([0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertTrue(bool(jnp.array_equal(observed, expected)))

    def test_sweep_contains_local_and_ordered_recruitment(self):
        brainstate.random.seed(0)
        brainstate.environ.set(dt=model.DT)
        _, _, _, activity, recruited, onsets = model.sweep_conditions()
        local, spread = model.choose_examples(recruited, onsets)

        self.assertEqual(activity.shape, (5, 4, 4, 800, model.N_REGIONS))
        self.assertEqual(int(jnp.sum(recruited[local])), 1)
        self.assertEqual(int(jnp.sum(recruited[spread])), model.N_REGIONS)
        self.assertTrue(bool(jnp.all(u.math.diff(onsets[spread]) > 0.0 * u.ms)))
        neighbor_peaks = jnp.max(activity[local], axis=0)[1:]
        self.assertTrue(bool(jnp.all(neighbor_peaks < model.RECRUITMENT_THRESHOLD)))


if __name__ == "__main__":
    unittest.main()
