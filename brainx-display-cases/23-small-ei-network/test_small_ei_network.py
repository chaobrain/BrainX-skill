from __future__ import annotations

import numpy as np
import pytest

brainpy = pytest.importorskip("brainpy")
brainstate = pytest.importorskip("brainstate")
brainunit = pytest.importorskip("brainunit")

import small_ei_network as model


def test_network_constants_and_split():
    assert model.N_NEURONS == 20
    net = model.EINetwork()
    assert net.n_exc == 16
    assert net.n_inh == 4
    assert net.num_neurons == 20


def test_short_rollout_returns_time_major_spikes():
    times, spikes = model.run(duration=1.0 * brainunit.ms)
    assert times.shape == (10,)
    assert spikes.shape == (10, 20)
    values = np.asarray(spikes)
    assert np.all((values == 0) | (values == 1))
