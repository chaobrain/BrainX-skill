import brainevent
import jax.numpy as jnp
import numpy as np

import internal_neural_compass as compass


def test_circular_difference_crosses_wrap_boundary():
    difference = compass.circular_difference(
        jnp.deg2rad(-179.0), jnp.deg2rad(179.0)
    )
    assert np.isclose(np.rad2deg(float(difference)), 2.0, atol=1e-4)


def test_population_vector_decodes_known_bump():
    angles = compass.ring_angles()
    target = jnp.deg2rad(70.0)
    rates = jnp.exp(6.0 * (jnp.cos(angles - target) - 1.0))
    decoded, strength = compass.decode_bump(rates, angles)
    assert abs(np.rad2deg(float(compass.circular_difference(decoded, target)))) < 0.1
    assert float(strength) > 0.8


def test_brainevent_ring_communication_preserves_target_shape():
    spikes = jnp.zeros(compass.N_NEURONS, dtype=bool).at[0].set(True)
    current = brainevent.BinaryArray(spikes) @ compass.recurrent_kernel(
        compass.ring_angles()
    )
    assert current.shape == (compass.N_NEURONS,)
    assert current.unit == compass.LOCAL_WEIGHT.unit


def test_delay_tap_arrives_after_exact_completed_steps():
    history = jnp.zeros((3, 1), dtype=bool)
    observed = []
    for event in (True, False, False, False):
        history, delayed = compass.advance_spike_history(
            history, jnp.array([event]), delay_steps=2
        )
        observed.append(bool(delayed[0]))
    assert observed == [False, False, True, False]


def test_outcome_classifier_requires_departure_then_sustained_return():
    error = jnp.array(
        [
            [2.0, 2.0, 2.0],
            [3.0, 20.0, 20.0],
            [2.0, 8.0, 18.0],
            [2.0, 7.0, 17.0],
        ]
    )
    strength = jnp.ones_like(error) * 0.8
    rate_ratio = jnp.ones_like(error)
    outcomes = compass.classify_lesion_outcomes(
        error,
        strength,
        rate_ratio,
        recovery_steps=2,
    )
    assert np.array_equal(np.asarray(outcomes), np.array([0, 1, 2]))


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} focused tests passed")
