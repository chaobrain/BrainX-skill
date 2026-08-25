from __future__ import annotations

import json
import numpy as np

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

import sparse_ei_network as model


def small_config():
    return model.NetworkConfig(
        n_exc=16,
        n_inh=4,
        exc_indegree=4,
        inh_indegree=2,
        external_indegree=4,
        burn_ms=2.0,
        analysis_ms=8.0,
    )


def test_canonical_units_and_external_rate_conversion():
    assert np.isclose(model.NU_THRESHOLD.to_decimal(u.Hz), 10.0)
    assert int(round(model.DELAY / model.DT)) == 15
    for eta in (0.9, 2.0, 4.0):
        aggregate_lambda = (
            1_000 * model.NU_THRESHOLD * eta * model.DT
        )
        assert np.isclose(aggregate_lambda, eta)


def test_fixed_indegree_unique_in_range_and_no_autapses():
    cfg = small_config()
    _, _, exc, inh = model.make_connectivity(cfg, repeat_seed=17)
    assert exc.shape == (cfg.num_neurons, cfg.exc_indegree)
    assert inh.shape == (cfg.num_neurons, cfg.inh_indegree)
    assert np.all((0 <= exc) & (exc < cfg.n_exc))
    assert np.all((0 <= inh) & (inh < cfg.n_inh))
    assert all(np.unique(row).size == cfg.exc_indegree for row in exc)
    assert all(np.unique(row).size == cfg.inh_indegree for row in inh)
    assert all(target not in exc[target] for target in range(cfg.n_exc))
    assert all(
        target - cfg.n_exc not in inh[target]
        for target in range(cfg.n_exc, cfg.num_neurons)
    )


def test_delay_impulse_arrives_at_step_15_only():
    with brainstate.environ.context(dt=model.DT):
        delay = brainstate.nn.Delay(
            jax.ShapeDtypeStruct((1,), jnp.bool_), model.DELAY
        )
        brainstate.nn.init_all_states(delay)
        values = jnp.arange(20) == 0

        def step(value):
            delay.update(jnp.asarray([value]))
            delayed = delay.retrieve_at_step(jnp.asarray(15, dtype=jnp.int32))
            return delayed[0]

        observed = np.asarray(brainstate.transform.for_loop(step, values))
    expected = np.zeros(20, dtype=bool)
    expected[15] = True
    assert np.array_equal(observed, expected)


def test_lif_threshold_reset_and_refractory_jump_insensitivity():
    with brainstate.environ.context(dt=model.DT):
        neuron = model.BrunelLIF(1)
        brainstate.nn.init_all_states(neuron)

        def step(t, jump):
            with brainstate.environ.context(t=t):
                return neuron(jump), neuron.V.value

        times = u.math.arange(0.0 * u.ms, 2.3 * u.ms, model.DT)
        jumps = jnp.zeros(times.size).at[0].set(10.1).at[1:22].set(100.0) * u.mV
        spikes, voltages = brainstate.transform.for_loop(step, times, jumps)
    spikes = np.asarray(spikes[:, 0])
    voltages_mv = np.asarray(voltages[:, 0].to_decimal(u.mV))
    assert spikes[0]
    assert not spikes[1:21].any()
    assert np.allclose(voltages_mv[0:21], 10.0)


def test_compiled_small_rollout_replays_exactly():
    cfg = small_config()
    with brainstate.environ.context(dt=model.DT):
        exc_conn, inh_conn, _, _ = model.make_connectivity(cfg, repeat_seed=7)
        net = model.SparseEINetwork(
            cfg, external_seed=9, exc_conn=exc_conn, inh_conn=inh_conn
        )
        brainstate.nn.init_all_states(net)
        runner = model.build_runner(net)
        model.reset_run(net, initial_seed=8, external_seed=9)
        first = np.asarray(runner(jnp.asarray(5.0), jnp.asarray(2.0)))
        model.reset_run(net, initial_seed=8, external_seed=9)
        replay = np.asarray(runner(jnp.asarray(5.0), jnp.asarray(2.0)))
    assert first.shape == (cfg.num_steps, cfg.num_neurons)
    assert np.array_equal(first, replay)


def test_small_rollout_eager_and_compiled_parity():
    cfg = small_config()
    with brainstate.environ.context(dt=model.DT):
        eager_exc, eager_inh, eager_exc_indices, eager_inh_indices = (
            model.make_connectivity(cfg, repeat_seed=7)
        )
        compiled_exc, compiled_inh, compiled_exc_indices, compiled_inh_indices = (
            model.make_connectivity(cfg, repeat_seed=7)
        )
        assert np.array_equal(eager_exc_indices, compiled_exc_indices)
        assert np.array_equal(eager_inh_indices, compiled_inh_indices)
        times = u.math.arange(0.0 * u.ms, cfg.num_steps * model.DT, model.DT)
        indices = jnp.arange(cfg.num_steps, dtype=jnp.int32)

        eager_net = model.SparseEINetwork(
            cfg, external_seed=9, exc_conn=eager_exc, inh_conn=eager_inh
        )
        brainstate.nn.init_all_states(eager_net)
        model.reset_run(eager_net, initial_seed=8, external_seed=9)
        eager = np.asarray(
            brainstate.transform.for_loop(
                lambda t, i: eager_net.update(t, i, 5.0, 2.0),
                times,
                indices,
            )
        )

        compiled_net = model.SparseEINetwork(
            cfg, external_seed=9, exc_conn=compiled_exc, inh_conn=compiled_inh
        )
        brainstate.nn.init_all_states(compiled_net)
        runner = model.build_runner(compiled_net)
        model.reset_run(compiled_net, initial_seed=8, external_seed=9)
        compiled = np.asarray(
            runner(jnp.asarray(5.0), jnp.asarray(2.0))
        )
    assert np.array_equal(eager, compiled)


def test_no_recurrence_external_drive_is_finite_and_active():
    cfg = small_config()
    with brainstate.environ.context(dt=model.DT):
        exc_conn, inh_conn, _, _ = model.make_connectivity(cfg, repeat_seed=11)
        exc_conn.data = jnp.asarray(0.0)
        inh_conn.data = jnp.asarray(0.0)
        net = model.SparseEINetwork(
            cfg, external_seed=12, exc_conn=exc_conn, inh_conn=inh_conn
        )
        brainstate.nn.init_all_states(net)
        runner = model.build_runner(net)
        model.reset_run(net, initial_seed=10, external_seed=12)
        spikes = np.asarray(
            runner(jnp.asarray(5.0), jnp.asarray(2.0))
        )
    assert np.isfinite(np.asarray(net.neurons.V.value.to_decimal(u.mV))).all()
    assert spikes.any()


def test_metric_classifier_contract():
    base = {
        "overall_firing_rate_hz": 5.0,
        "population_rate_cv_1ms": 0.3,
        "spectral_peak_to_background": 10.0,
        "dominant_frequency_hz": 22.0,
        "isi_cv_all_mean": 0.9,
    }
    assert model.classify_metrics(base) == "slow_synchronous_irregular"
    assert model.classify_metrics({**base, "dominant_frequency_hz": 180.0}) == (
        "fast_synchronous_irregular"
    )
    assert model.classify_metrics({**base, "spectral_peak_to_background": 2.0}) == (
        "asynchronous_irregular"
    )
    assert model.classify_metrics({**base, "isi_cv_all_mean": 0.2}) == (
        "synchronous_regular"
    )


def test_locked_run_contract_round_trip(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(model.config_payload(model.NetworkConfig(), model.REPEAT_SEEDS)),
        encoding="ascii",
    )
    config, seeds = model.load_run_contract(path)
    assert config == model.NetworkConfig()
    assert seeds == model.REPEAT_SEEDS
