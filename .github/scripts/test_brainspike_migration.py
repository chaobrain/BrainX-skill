"""Pinned Linux regression: block BrainPy for native/NEST paths; exempt BrainCell import."""
import importlib.abc
import importlib.metadata
import json
import re
from pathlib import Path
import sys
import unittest

class BlockBrainPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("brainpy", "brainpy_state"):
            raise ImportError("BrainPy is forbidden in the migration tests")

BLOCKER = BlockBrainPy()
sys.meta_path.insert(0, BLOCKER)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "skills/package-skills/brainspike/scripts"))
from native_workflows import LIF, RecurrentSNN, ExponentialConductance
import brainstate
import braintrace
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
brainstate.environ.set(precision=64, dt=0.1 * u.ms)

class MigrationTests(unittest.TestCase):
    def setUp(self):
        brainstate.random.seed(42)

    def test_custom_lif_reset_and_analytic_subthreshold(self):
        neuron = LIF(2)
        neuron.init_state()
        voltage = brainstate.transform.for_loop(
            lambda _: (neuron(0.02 * u.nA), neuron.V.value)[1], jnp.arange(20)
        )
        expected = -65 + 2 * (1 - 0.99 ** 20)
        np.testing.assert_allclose(voltage[-1].to_decimal(u.mV), expected, atol=1e-10)
        neuron.reset_state()
        np.testing.assert_array_equal(neuron.V.value.to_decimal(u.mV), [-65, -65])
        spikes = brainstate.transform.for_loop(lambda _: neuron(1 * u.nA), jnp.arange(100))
        self.assertGreater(float(spikes.sum()), 0)
        self.assertTrue(np.all((spikes == 0) | (spikes == 1)))

    def test_conductance_filter_and_event_communication(self):
        import brainevent
        synapse = ExponentialConductance(2)
        synapse.init_state()
        weights = jnp.asarray([[1., 2.], [3., 4.]]) * u.nS
        increment = brainevent.BinaryArray(jnp.asarray([True, False])) @ weights
        first = synapse(increment)
        second = synapse(jnp.zeros(2) * u.nS)
        np.testing.assert_allclose(first.to_decimal(u.nS), [1., 2.])
        np.testing.assert_allclose(second.to_decimal(u.nS), np.asarray([1., 2.]) * np.exp(-0.1/5))
        point = LIF(2)
        point.init_state()
        current = second * (0 * u.mV - point.V.value)
        self.assertTrue(u.math.all(current > 0 * u.nA))
        point(current)
        synapse.reset_state()
        np.testing.assert_array_equal(synapse.g.value.to_decimal(u.nS), [0., 0.])
        state = synapse.g
        synapse.reset_state(batch_size=2)
        synapse(jnp.ones((2, 2))*u.nS)
        synapse.reset_state(batch_size=3)
        self.assertIs(synapse.g, state)
        self.assertEqual(synapse.g.value.shape, (3, 2))
        np.testing.assert_array_equal(synapse.g.value.to_decimal(u.nS), np.zeros((3, 2)))
        self.assertEqual(synapse(jnp.ones((3, 2))*u.nS).shape, (3, 2))
        synapse.reset_state()
        self.assertEqual(synapse.g.value.shape, (2,))


    def test_three_scale_coupling_and_zero_limit(self):
        import brainmass
        # BrainCell 0.2.0 has a transitive BrainPy runtime dependency.
        sys.meta_path.remove(BLOCKER)
        try:
            from multiscale_workflow import CoupledScales
        finally:
            sys.meta_path.insert(0, BLOCKER)
        print("Multiscale exception: BrainCell 0.2.0 imports BrainPy transitively")
        dt = 0.1 * u.ms
        monitors = {"point": lambda m: m.point.V.value,
                    "spikes": lambda m: m.point.get_spike(),
                    "cell": lambda m: m.cell.V.value,
                    "mass": lambda m: m.mass.x.value}
        brainstate.random.seed(42)
        zero = brainmass.Simulator(CoupledScales(gain=0), dt=dt).run(10*u.ms, monitors=monitors)
        brainstate.random.seed(42)
        independent = CoupledScales(gain=0)
        brainstate.nn.init_all_states(independent)
        def separate(i):
            with brainstate.environ.context(t=i*dt):
                independent.point(0.5*u.nA)
                independent.cell.update(0*u.uA/u.cm**2)
                independent.mass.update(x_inp=0., y_inp=0.)
                return independent.point.V.value, independent.cell.V.value, independent.mass.x.value
        traces = brainstate.transform.for_loop(separate, jnp.arange(100))
        for name, trace in zip(["point", "cell", "mass"], traces):
            actual = zero[name].to_decimal(u.mV) if name != "mass" else zero[name]
            expected = trace.to_decimal(u.mV) if name != "mass" else trace
            np.testing.assert_allclose(actual, expected, atol=1e-10)
        brainstate.random.seed(42)
        coupled = brainmass.Simulator(CoupledScales(gain=1), dt=dt).run(10*u.ms, monitors=monitors)
        self.assertTrue(np.isfinite(coupled["cell"].to_decimal(u.mV)).all())
        self.assertGreater(float(jnp.max(jnp.abs(coupled["mass"] - zero["mass"]))), 0)
        first_spike = np.flatnonzero(np.asarray(coupled["spikes"])[:, 0])[0]
        np.testing.assert_array_equal(np.asarray(coupled["mass"])[:first_spike+1], 0)
        self.assertGreater(abs(float(coupled["mass"][first_spike+1, 0])), 0)

    def test_bptt_trainable_spikes_and_readout(self):
        model = RecurrentSNN()
        brainstate.nn.init_all_states(model)
        params = model.states(brainstate.ParamState)
        inputs = jnp.ones((80, 1))
        def loss():
            brainstate.nn.reset_all_states(model)
            output = brainstate.transform.for_loop(model, inputs)
            return jnp.mean((output - 0.25) ** 2)
        differentiate = brainstate.transform.grad(loss, params, return_value=True)
        gradients, initial = differentiate()
        self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves(gradients)))
        self.assertGreater(float(sum(jnp.sum(v*v) for v in jax.tree.leaves(gradients))), 0)
        print("Gradient norms:", {str(k): float(sum(jnp.sum(x*x) for x in jax.tree.leaves(v))) for k, v in gradients.items()})
        drive_before = jax.tree.map(lambda x: x.copy(), model.drive.weight.value)
        readout_before = jax.tree.map(lambda x: x.copy(), model.readout.weight.value)
        optimizer = braintools.optim.Adam(lr=0.01)
        optimizer.register_trainable_weights(params)
        @brainstate.transform.jit
        def train():
            grads, value = differentiate()
            optimizer.update(grads)
            return value
        for _ in range(40):
            train()
        final = loss()
        print("BPTT loss:", float(initial), "->", float(final))
        self.assertLess(float(final), float(initial))
        self.assertFalse(all(np.array_equal(x, y) for x, y in zip(jax.tree.leaves(model.drive.weight.value), jax.tree.leaves(drive_before))))
        self.assertFalse(all(np.array_equal(x, y) for x, y in zip(jax.tree.leaves(model.readout.weight.value), jax.tree.leaves(readout_before))))

    def test_heterogeneous_connection_delays(self):
        import brainspike
        sim = brainspike.Simulator(dt=0.1*u.ms)
        source = sim.create(brainspike.spike_generator, spike_times=jnp.asarray([1.])*u.ms)
        targets = sim.create(brainspike.iaf_psc_delta, 2, params={
            "E_L": 0*u.mV, "V_reset": 0*u.mV, "V_th": 20*u.mV,
            "V_initializer": braintools.init.Constant(0*u.mV),
        })
        recorder = sim.create(brainspike.spike_recorder)
        sim.connect(source, targets[:1], weight=30*u.mV, delay=0.5*u.ms)
        sim.connect(source, targets[1:2], weight=30*u.mV, delay=1.5*u.ms)
        sim.connect(targets, recorder)
        result = sim.simulate(5*u.ms)
        spikes = np.asarray(result.spikes(recorder))
        arrivals = [np.flatnonzero(spikes[:, i]) for i in range(2)]
        self.assertEqual([len(x) for x in arrivals], [1, 1])
        self.assertEqual(int(arrivals[1][0]-arrivals[0][0]), 10)
        print("Heterogeneous delay arrival indices:", [int(x[0]) for x in arrivals])

    def test_native_symmetric_gap_currents(self):
        left, right = LIF(1), LIF(1)
        left.init_state(); right.init_state()
        left.V.value = jnp.asarray([-60.])*u.mV
        right.V.value = jnp.asarray([-70.])*u.mV
        initial = abs(float((left.V.value-right.V.value).to_decimal(u.mV)[0]))
        def step(_):
            current = 2*u.nS*(right.V.value-left.V.value)
            left(current)
            right(-current)
            return left.V.value, right.V.value
        vl, vr = brainstate.transform.for_loop(step, jnp.arange(20))
        np.testing.assert_allclose(((vl+vr)/2).to_decimal(u.mV), -65., atol=1e-10)
        self.assertLess(abs(float((vl[-1]-vr[-1]).to_decimal(u.mV)[0])), initial)

    def test_nest_simulator_canonical(self):
        scope = {}
        content = (ROOT / "skills/package-skills/brainspike/SKILL.md").read_text()
        for block in re.findall(r"```python\n(.*?)```", content, re.S):
            exec(compile(block, "brainspike/SKILL.md", "exec"), scope)
        self.assertEqual(scope["spikes"].shape, (1000, 100))
        self.assertTrue(np.isfinite(scope["voltage"].to_decimal(u.mV)).all())

    def test_plot_scripts_write_to_current_directory(self):
        import os
        import runpy
        import tempfile
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        base = ROOT / "skills/package-skills/brainspike/references/nest-compatible/scripts"
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                for name, output in [("synapsecollection", "synapsecollection.png"),
                                     ("brunel_alpha", "brunel_alpha_raster.png"),
                                     ("spatial_gaussex", "spatial_gaussex.png"),
                                     ("brunel_delta", "brunel_delta_raster.png"),
                                     ("clopath_synapse_spike_pairing", "clopath_synapse_spike_pairing.png"),
                                     ("evaluate_tsodyks2_synapse", "evaluate_tsodyks2_synapse.png")]:
                    namespace = runpy.run_path(str(base / f"{name}.py"))
                    globals_ = namespace["main"].__globals__
                    if name in ("brunel_alpha", "brunel_delta"):
                        build = globals_["build"]
                        globals_["build"] = lambda **kwargs: build(order=20, simtime=5.)
                    elif name == "spatial_gaussex":
                        run = globals_["run"]
                        globals_["run"] = lambda: run(shape=(6, 6))
                    namespace["main"]()
                    self.assertTrue(Path(output).is_file(), name)
                    self.assertEqual(Path(output).read_bytes()[:8], b"\x89PNG\r\n\x1a\n", name)
                    plt.close("all")
            finally:
                os.chdir(previous_cwd)

    def test_pp_prop_compile_and_sequence_gradient(self):
        model = RecurrentSNN()
        inputs = jnp.ones((80, 1, 1))
        learner = braintrace.compile(model, braintrace.pp_prop, inputs[0], batch_size=1, decay_or_rank=0.9)
        self.assertEqual(learner.report.counts["errors"], 0)
        self.assertTrue(learner.report.etrace_weights)
        brainstate.nn.reset_all_states(model, batch_size=1)
        learner.reset_state(batch_size=1)
        def step_loss(x):
            return jnp.mean((learner(x) - 0.25) ** 2)
        gradients, losses = learner.etrace_grad(
            inputs, step_fn=step_loss, reduction="sum", return_value=True
        )
        self.assertTrue(np.isfinite(losses).all())
        self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves(gradients)))
        self.assertGreater(float(sum(jnp.sum(v*v) for v in jax.tree.leaves(gradients))), 0)
        for path, group in gradients.items():
            self.assertGreater(float(sum(jnp.sum(v*v) for v in jax.tree.leaves(group))), 0, str(path))
        def bptt_loss():
            brainstate.nn.reset_all_states(model, batch_size=1)
            output = brainstate.transform.for_loop(model, inputs)
            return jnp.sum(jnp.mean((output - 0.25)**2, axis=(1, 2)))
        oracle = brainstate.transform.grad(bptt_loss, model.states(brainstate.ParamState))()
        a = jnp.concatenate([v.ravel() for v in jax.tree.leaves(gradients)])
        b = jnp.concatenate([v.ravel() for v in jax.tree.leaves(oracle)])
        cosine = jnp.vdot(a,b) / (jnp.linalg.norm(a)*jnp.linalg.norm(b))
        print("pp-prop/BPTT gradient cosine:", float(cosine))
        self.assertGreater(float(cosine), 0)
        optimizer = braintools.optim.Adam(lr=0.01)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        initial = float(losses.mean())
        for _ in range(30):
            brainstate.nn.reset_all_states(model, batch_size=1)
            learner.reset_state(batch_size=1)
            grads, losses = learner.etrace_grad(inputs, step_fn=step_loss, reduction="sum", return_value=True)
            optimizer.update(grads)
        brainstate.nn.reset_all_states(model, batch_size=1)
        learner.reset_state(batch_size=1)
        outputs = learner.etrace_evolve(inputs, return_outputs=True)
        final = float(jnp.mean((outputs-0.25)**2))
        print("pp-prop loss:", initial, "->", final)
        self.assertLess(final, initial)

if __name__ == "__main__":
    print(json.dumps({p: importlib.metadata.version(p) for p in
        ["brainspikes", "brainstate", "braintrace", "brainmass", "braincell", "jax"]}))
    unittest.main(verbosity=2)
