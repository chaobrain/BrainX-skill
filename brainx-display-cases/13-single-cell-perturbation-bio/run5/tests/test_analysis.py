import numpy as np

from v1_influence import (
    correlation_rows,
    orientation_difference_deg,
    photostim_waveform,
    piecewise_distance_design,
    Config,
)


def test_orientation_difference_is_180_degree_periodic():
    actual = orientation_difference_deg(np.array([0, 45, 90, 135, 180, 315]), 0)
    np.testing.assert_allclose(actual, [0, 45, 90, 45, 0, 45])


def test_row_correlations_handle_constant_tuning():
    a = np.array([[1, 2, 3], [1, 1, 1], [1, 2, 3]], dtype=float)
    b = np.array([[2, 4, 6], [2, 3, 4], [3, 2, 1]], dtype=float)
    np.testing.assert_allclose(correlation_rows(a, b), [1, 0, -1], atol=1e-12)


def test_piecewise_distance_basis_has_expected_knots():
    x = piecewise_distance_design(np.array([25, 100, 300, 500]))
    np.testing.assert_allclose(
        x,
        [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
    )


def test_photostim_protocol_has_four_32_ms_sweeps():
    cfg = Config()
    wave = photostim_waveform(cfg)
    assert np.count_nonzero(wave) == cfg.photo_sweeps * round(cfg.photo_sweep_ms / cfg.dt_ms)
    transitions = np.flatnonzero(np.diff(np.r_[0, wave > 0, 0]))
    assert transitions.size == 2 * cfg.photo_sweeps
    np.testing.assert_allclose(np.diff(transitions.reshape(-1, 2), axis=1)[:, 0] * cfg.dt_ms, 32.0)
