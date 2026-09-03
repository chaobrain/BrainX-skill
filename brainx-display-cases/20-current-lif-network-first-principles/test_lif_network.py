from __future__ import annotations

import json
from pathlib import Path

import brainevent
import brainpy
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import lif_network


def test_external_rate_reduces_to_eta_events_per_step():
    config = lif_network.load_config()
    model = config["model"]
    for eta in (0.9, 2.0, 4.0):
        nu_ext = (
            eta
            * model["threshold_mV"]
            * u.mV
            / (
                model["JE_mV"]
                * u.mV
                * model["CE"]
                * model["tau_m_ms"]
                * u.ms
            )
        )
        mean_count = float(model["Cext"] * nu_ext * model["dt_ms"] * u.ms)
        assert np.isclose(mean_count, eta)


def test_fixed_fan_in_has_unique_sources_and_no_autapses():
    structures = lif_network.build_random_structures(
        11,
        ne=80,
        ni=20,
        ce=8,
        ci=2,
        sample_size=10,
        reset_mV=10.0,
        threshold_mV=20.0,
    )
    lif_network.validate_connectivity(
        structures["exc_indices"],
        structures["inh_indices"],
        ne=80,
        ni=20,
        ce=8,
        ci=2,
    )


def test_brainevent_fixed_fan_in_counts_and_preserves_voltage_unit():
    indices = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    connectivity = brainevent.FixedNumPerPost(
        (0.1 * u.mV, jnp.asarray(indices)), shape=(4, 3)
    )
    result = brainevent.BinaryArray(
        jnp.asarray([True, False, True, False])
    ) @ connectivity
    assert result.unit == u.mV
    np.testing.assert_allclose(result.to_decimal(u.mV), [0.1, 0.1, 0.1])


def test_brainstate_delay_delivers_an_impulse_after_exactly_15_steps():
    source_spikes = jnp.zeros((20, 1), dtype=bool).at[0, 0].set(True)
    previous_spikes = jnp.concatenate(
        [jnp.zeros((1, 1), dtype=bool), source_spikes[:-1]], axis=0
    )
    with brainstate.environ.context(dt=0.1 * u.ms):
        delay = brainstate.nn.Delay(
            jax.ShapeDtypeStruct((1,), jnp.bool_), 1.5 * u.ms
        )
        brainstate.nn.init_all_states(delay)

        def step(value):
            delay.update(value)
            return delay.retrieve_at_step(jnp.asarray(14, dtype=jnp.int32))

        output = brainstate.transform.for_loop(step, previous_spikes)
    expected = np.zeros((20, 1), dtype=bool)
    expected[15, 0] = True
    np.testing.assert_array_equal(np.asarray(output), expected)


def test_external_poisson_mean_and_variance_fixed_seed():
    sample_size = 250_000
    for offset, expected_mean in enumerate((0.9, 2.0, 4.0)):
        rng = brainstate.random.RandomState(7100 + offset)
        values = np.asarray(rng.poisson(lam=expected_mean, size=(sample_size,)))
        observed_mean = float(np.mean(values))
        observed_variance = float(np.var(values))
        mean_se = np.sqrt(expected_mean / sample_size)
        variance_se = np.sqrt(
            (expected_mean + 2.0 * expected_mean**2) / sample_size
        )
        assert abs(observed_mean - expected_mean) <= 5.0 * mean_se
        assert abs(observed_variance - expected_mean) <= 5.0 * variance_se


def test_signed_delta_sum_matches_hand_computed_transition():
    with brainstate.environ.context(dt=0.1 * u.ms, fit=False):
        neuron = brainpy.state.LIFRef(
            1,
            R=1.0 * u.ohm,
            tau=20.0 * u.ms,
            tau_ref=2.0 * u.ms,
            V_rest=0.0 * u.mV,
            V_th=20.0 * u.mV,
            V_reset=10.0 * u.mV,
            spk_reset="hard",
            V_initializer=braintools.init.Constant(10.0 * u.mV),
        )
        exc = brainpy.state.DeltaProj(
            comm=lif_network.ExternalJumpComm(0.1 * u.mV),
            post=neuron,
            label="test_exc",
        )
        inh = brainpy.state.DeltaProj(
            comm=lif_network.ExternalJumpComm(-0.3 * u.mV),
            post=neuron,
            label="test_inh",
        )
        brainstate.nn.init_all_states(neuron)
        with brainstate.environ.context(t=0.0 * u.ms, i=0):
            exc(jnp.asarray([2], dtype=jnp.int32))
            inh(jnp.asarray([1], dtype=jnp.int32))
            spike = neuron(jnp.zeros(1) * u.mA)
        expected_mV = 10.0 * np.exp(-0.1 / 20.0) + 0.2 - 0.3
        np.testing.assert_allclose(
            neuron.V.value.to_decimal(u.mV), [expected_mV], rtol=1e-6
        )
        np.testing.assert_array_equal(np.asarray(spike), [0.0])


def test_reference_delay_threshold_reset_and_refractory_transition():
    with brainstate.environ.context(dt=0.1 * u.ms, fit=False):
        neuron = brainpy.state.LIFRef(
            1,
            R=1.0 * u.ohm,
            tau=20.0 * u.ms,
            tau_ref=2.0 * u.ms,
            V_rest=0.0 * u.mV,
            V_th=20.0 * u.mV,
            V_reset=10.0 * u.mV,
            spk_reset="hard",
            V_initializer=braintools.init.Constant(10.0 * u.mV),
        )
        delayed_projection = brainpy.state.DeltaProj(
            comm=lif_network.ExternalJumpComm(11.0 * u.mV),
            post=neuron,
            label="delayed_test",
        )
        forcing_projection = brainpy.state.DeltaProj(
            comm=lif_network.ExternalJumpComm(20.0 * u.mV),
            post=neuron,
            label="refractory_test",
        )
        delay = brainstate.nn.Delay(
            jax.ShapeDtypeStruct((1,), jnp.bool_), 1.5 * u.ms
        )
        brainstate.nn.init_all_states(neuron)
        brainstate.nn.init_all_states(delay)
        steps = jnp.arange(40, dtype=jnp.int32)
        times = steps * (0.1 * u.ms)
        previous_source = jnp.zeros((40, 1), dtype=bool).at[1, 0].set(True)

        def step(t, index, source):
            with brainstate.environ.context(t=t, i=index):
                delay.update(source)
                delayed_projection(
                    delay.retrieve_at_step(jnp.asarray(14, dtype=jnp.int32))
                )
                forcing_projection(
                    jnp.asarray([index > 15], dtype=jnp.int32)
                )
                spike = neuron(jnp.zeros(1) * u.mA)
                return spike, neuron.V.value

        spikes, voltages = brainstate.transform.for_loop(
            step, times, steps, previous_source
        )
    spikes = np.asarray(spikes)[:, 0]
    voltages_mV = np.asarray(voltages.to_decimal(u.mV))[:, 0]
    assert spikes[15] == 1.0
    assert voltages_mV[15] >= 20.0
    assert voltages_mV[16] == 10.0
    assert not np.any(spikes[16:35])
    assert spikes[35] == 1.0


def test_lif_refractory_interval_is_at_least_two_ms():
    with brainstate.environ.context(dt=0.1 * u.ms, fit=False):
        neuron = brainpy.state.LIFRef(
            1,
            R=1.0 * u.ohm,
            tau=20.0 * u.ms,
            tau_ref=2.0 * u.ms,
            V_rest=0.0 * u.mV,
            V_th=20.0 * u.mV,
            V_reset=10.0 * u.mV,
            spk_reset="hard",
            V_initializer=braintools.init.Constant(10.0 * u.mV),
        )
        projection = brainpy.state.DeltaProj(
            comm=lif_network.ExternalJumpComm(20.0 * u.mV),
            post=neuron,
            label="forced",
        )
        brainstate.nn.init_all_states(neuron)
        steps = jnp.arange(80, dtype=jnp.int32)
        times = steps * (0.1 * u.ms)

        def step(t, index):
            with brainstate.environ.context(t=t, i=index):
                projection(jnp.ones(1, dtype=jnp.int32))
                return neuron(jnp.zeros(1) * u.mA)

        spikes = np.asarray(brainstate.transform.for_loop(step, times, steps))[:, 0]
    spike_steps = np.flatnonzero(spikes)
    assert spike_steps.size >= 3
    assert np.all(np.diff(spike_steps) >= 20)


def test_small_rollout_replays_exactly(tmp_path: Path):
    config = lif_network.load_config()
    model = {**config["model"], "NE": 80, "NI": 20, "CE": 8, "CI": 2}
    protocol = {
        **config["protocol"],
        "duration_ms": 40.0,
        "transient_ms": 10.0,
        "sample_size": 10,
    }
    small = {**config, "model": model, "protocol": protocol}
    structures = lif_network.build_random_structures(
        11,
        ne=80,
        ni=20,
        ce=8,
        ci=2,
        sample_size=10,
        reset_mV=10.0,
        threshold_mV=20.0,
    )
    with brainstate.environ.context(dt=0.1 * u.ms, fit=False):
        network = lif_network.CurrentLIFNetwork(small, structures)
        brainstate.nn.init_all_states(network)
        rollout = lif_network.create_rollout(network, small)
        network.set_condition(3.0, 2.0, small)
        network.reset()
        first = tuple(np.asarray(value) for value in rollout())
        network.reset()
        replay = tuple(np.asarray(value) for value in rollout())
    for observed, repeated in zip(first, replay, strict=True):
        np.testing.assert_array_equal(observed, repeated)
    assert first[0].shape == (400,)
    assert first[1].shape == (400,)
    assert first[2].shape == (400, 10)
    assert np.all(np.isfinite(np.asarray(network.neurons.V.value.to_decimal(u.mV))))


def test_config_is_locked_to_approved_contract():
    config = json.loads((Path(__file__).parent / "config.json").read_text())
    assert config["protocol"]["seeds"] == [11, 29, 47]
    assert config["protocol"]["duration_ms"] == 5000.0
    assert config["protocol"]["transient_ms"] == 1000.0
    assert config["model"]["dt_ms"] == 0.1
