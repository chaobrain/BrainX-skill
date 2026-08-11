import brainevent
import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from cortical_wave import (
    BENDS,
    CROSSES,
    DIES,
    SPLITS,
    ExperimentConfig,
    circular_lesion_mask,
    classify_outcome,
    make_local_csr,
    sheet_coordinates,
)


def test_circular_lesion_is_nested_and_unit_aware():
    config = ExperimentConfig(nx=9, ny=7, obstacle_x=4 * u.mm, obstacle_y=3 * u.mm)
    x, y = sheet_coordinates(config)
    small = circular_lesion_mask(x, y, 1.0 * u.mm, config.obstacle_x, config.obstacle_y)
    large = circular_lesion_mask(x, y, 2.0 * u.mm, config.obstacle_x, config.obstacle_y)

    assert int(jnp.sum(small)) == 5
    assert bool(jnp.all(~small | large))


def test_local_csr_communicates_only_to_neighboring_sites():
    with brainstate.environ.context(precision=32):
        conn = make_local_csr(3, 2, 1.0, 2.0 * u.siemens, include_self=False)
    spike = brainevent.BinaryArray(jnp.asarray([True, False, False, False, False, False]))
    received = spike @ conn

    assert received.shape == (6,)
    np.testing.assert_allclose(
        received.to_decimal(u.siemens),
        np.asarray([0.0, 2.0, 0.0, 2.0, 0.0, 0.0]),
    )


def test_outcome_classifier_uses_reach_and_two_bypass_routes():
    assert classify_outcome(0.0, 0.8, 1.0, 1.0) == CROSSES
    assert classify_outcome(4.0, 0.8, 0.8, 0.05) == BENDS
    assert classify_outcome(4.0, 0.8, 0.8, 0.7) == SPLITS
    assert classify_outcome(4.0, 0.05, 0.8, 0.8) == DIES
