from __future__ import annotations

import brainevent
import brainpy
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import brunel_fig8 as model


def small_config() -> model.NetworkConfig:
    return model.NetworkConfig(
        n_exc=16,
        n_inh=4,
        exc_indegree=4,
        inh_indegree=2,
        external_indegree=4,
        burn_ms=2.0,
        analysis_ms=8.0,
    )


def test_brainpy_owner_units_and_external_conversion():
    config = model.NetworkConfig()
    assert np.isclose(config.nu_threshold.to_decimal(u.Hz), 10.0)
    assert np.isclose(config.external_lambda(0.9), 0.9)
    assert np.isclose(config.external_lambda(2.0), 2.0)
    assert np.isclose(config.external_lambda(4.0), 4.0)
    assert int(round(model.DELAY / model.DT)) == 15
    assert issubclass(model.BrunelLIF, brainpy.state.Neuron)


def test_fixed_indegree_is_unique_in_range_and_has_no_autapses():
    config = small_config()
    _, _, exc, inh = model.make_connectivity(config, graph_seed=11)
    assert exc.shape == (config.num_neurons, config.exc_indegree)
    assert inh.shape == (config.num_neurons, config.inh_indegree)
    assert np.all((0 <= exc) & (exc < config.n_exc))
    assert np.all((0 <= inh) & (inh < config.n_inh))
    assert all(np.unique(row).size == config.exc_indegree for row in exc)
    assert all(np.unique(row).size == config.inh_indegree for row in inh)
    assert all(target not in exc[target] for target in range(config.n_exc))
    assert all(
        target - config.n_exc not in inh[target]
        for target in range(config.n_exc, config.num_neurons)
    )


def test_fixed_num_per_post_orientation():
    indices = jnp.asarray([[0, 2], [1, 3]], dtype=jnp.int32)
    conn = brainevent.FixedNumPerPost(1.0, indices, shape=(4, 2))
    output = np.asarray(brainevent.BinaryArray([True, False, True, False]) @ conn)
    assert np.array_equal(output, np.asarray([2.0, 0.0]))


def test_delay_impulse_has_1p5_ms_physical_latency():
    with brainstate.environ.context(dt=model.DT):
        delay = brainstate.nn.Delay(
            jax.ShapeDtypeStruct((1,), jnp.bool_), model.DELAY
        )
        brainstate.nn.init_all_states(delay)
        inputs = jnp.arange(20) == 0

        def step(value):
            delayed = delay.retrieve_at_step(jnp.asarray(15, dtype=jnp.int32))[0]
            delay.update(jnp.asarray([value]))
            return delayed

        observed = np.asarray(brainstate.transform.for_loop(step, inputs))
    expected = np.zeros(20, dtype=bool)
    expected[16] = True
    assert np.array_equal(observed, expected)
    elapsed_ms = ((np.flatnonzero(observed)[0] - 1) * model.DT).to_decimal(u.ms)
    assert np.isclose(elapsed_ms, 1.5)


def test_lif_threshold_reset_and_refractory_jump_insensitivity():
    with brainstate.environ.context(dt=model.DT):
        neuron = model.BrunelLIF(1)
        brainstate.nn.init_all_states(neuron)
        times = u.math.arange(0.0 * u.ms, 2.3 * u.ms, model.DT)
        jumps = jnp.zeros(times.size).at[0].set(10.1).at[1:22].set(100.0) * u.mV

        def step(t, jump):
            with brainstate.environ.context(t=t):
                return neuron(jump), neuron.V.value

        spikes, voltage = brainstate.transform.for_loop(step, times, jumps)
    spikes = np.asarray(spikes[:, 0])
    voltage_mv = np.asarray(voltage[:, 0].to_decimal(u.mV))
    assert spikes[0]
    assert not spikes[1:21].any()
    assert np.allclose(voltage_mv[:21], 10.0)


def test_compiled_rollout_replays_after_complete_reset():
    config = small_config()
    with brainstate.environ.context(dt=model.DT):
        exc_conn, inh_conn, _, _ = model.make_connectivity(config, graph_seed=7)
        net = model.SparseEINetwork(config, exc_conn, inh_conn, external_seed=9)
        brainstate.nn.init_all_states(net)
        runner = model.build_runner(net)
        model.reset_run(net, initial_seed=8, external_seed=9)
        first = np.asarray(runner(jnp.asarray(5.0), jnp.asarray(2.0)))
        first_v = np.asarray(net.neurons.V.value.to_decimal(u.mV))
        model.reset_run(net, initial_seed=8, external_seed=9)
        replay = np.asarray(runner(jnp.asarray(5.0), jnp.asarray(2.0)))
        replay_v = np.asarray(net.neurons.V.value.to_decimal(u.mV))
    assert first.shape == (config.num_steps, config.num_neurons)
    assert np.array_equal(first, replay)
    assert np.array_equal(first_v, replay_v)


def test_eager_and_compiled_rollouts_match():
    config = small_config()
    with brainstate.environ.context(dt=model.DT):
        exc_a, inh_a, _, _ = model.make_connectivity(config, graph_seed=7)
        eager_net = model.SparseEINetwork(config, exc_a, inh_a, external_seed=9)
        brainstate.nn.init_all_states(eager_net)
        model.reset_run(eager_net, initial_seed=8, external_seed=9)
        times = u.math.arange(0.0 * u.ms, config.num_steps * model.DT, model.DT)
        indices = jnp.arange(config.num_steps, dtype=jnp.int32)
        eager = np.asarray(
            brainstate.transform.for_loop(
                lambda t, i: eager_net.update(t, i, 5.0, 2.0), times, indices
            )
        )
        eager_v = np.asarray(eager_net.neurons.V.value.to_decimal(u.mV))

        exc_b, inh_b, _, _ = model.make_connectivity(config, graph_seed=7)
        compiled_net = model.SparseEINetwork(config, exc_b, inh_b, external_seed=9)
        brainstate.nn.init_all_states(compiled_net)
        model.reset_run(compiled_net, initial_seed=8, external_seed=9)
        compiled = np.asarray(
            model.build_runner(compiled_net)(jnp.asarray(5.0), jnp.asarray(2.0))
        )
        compiled_v = np.asarray(compiled_net.neurons.V.value.to_decimal(u.mV))
    assert np.array_equal(eager, compiled)
    assert np.array_equal(eager_v, compiled_v)


def test_no_recurrence_external_drive_is_finite_and_active():
    config = small_config()
    with brainstate.environ.context(dt=model.DT):
        exc_conn, inh_conn, _, _ = model.make_connectivity(config, graph_seed=13)
        exc_conn.data = jnp.asarray(0.0)
        inh_conn.data = jnp.asarray(0.0)
        net = model.SparseEINetwork(config, exc_conn, inh_conn, external_seed=15)
        brainstate.nn.init_all_states(net)
        model.reset_run(net, initial_seed=14, external_seed=15)
        spikes = np.asarray(
            model.build_runner(net)(jnp.asarray(5.0), jnp.asarray(4.0))
        )
        voltage = np.asarray(net.neurons.V.value.to_decimal(u.mV))
    assert np.isfinite(voltage).all()
    assert spikes.any()


def test_acceptance_uses_only_declared_predicates():
    ai = {
        "overall_firing_rate_hz": 37.7,
        "isi_cv_all_mean": 0.9,
        "population_rate_cv_1ms": 10.0,
        "dominant_frequency_hz": 500.0,
    }
    assert model.assess_condition("asynchronous_irregular", ai) == (
        "reproduced",
        [],
    )
    sr = {
        **ai,
        "isi_cv_all_mean": 0.1,
        "population_rate_cv_1ms": 0.0,
    }
    assert model.assess_condition("synchronous_regular", sr) == (
        "reproduced",
        [],
    )


def test_probe_selection_is_fixed_and_stratified():
    config = model.NetworkConfig()
    first = model.choose_probes(config, seed=303)
    replay = model.choose_probes(config, seed=303)
    assert np.array_equal(first, replay)
    assert first.shape == (50,)
    assert (first[:40] < config.n_exc).all()
    assert (first[40:] >= config.n_exc).all()
