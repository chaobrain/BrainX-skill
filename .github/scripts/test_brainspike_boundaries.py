"""Positive replacement patterns plus release-scoped counterexamples."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"skills/package-skills/brainspike/scripts"))
import brainspike
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from advanced_native_patterns import SaturatingSynapse,run_multirate
from nest_boundary_workflows import delay_arrivals,neuron_delay_matrix,voltage_trace,window_voltage,AutonomousLIF,fit_hard_count
from train_nest_simulator import rollout
from native_workflows import LIF


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(precision=64,dt=.1*u.ms)
        brainstate.random.seed(7)

    def test_nonlinear_pre_sharing_matches_per_edge_reference(self):
        pre=SaturatingSynapse((2,))
        edge=SaturatingSynapse((2,3))
        pre.init_state();edge.init_state()
        events=jnp.asarray([[1.,0.],[1.,1.],[0.,1.],[0.,0.]]*8)
        weights=jnp.asarray([[1.,2.,3.],[.5,.8,1.2]])*u.nS
        def step(x):
            a=pre(x)
            b=edge(jnp.broadcast_to(x[:,None],(2,3)))
            return (a[:,None]*weights).sum(axis=0),(b*weights).sum(axis=0)
        shared,reference=brainstate.transform.for_loop(step,events)
        np.testing.assert_allclose(shared.to_decimal(u.nS),reference.to_decimal(u.nS),atol=1e-12)
        print("Nonlinear homogeneous sharing max error",float(u.math.max(u.math.abs(shared-reference)).to_decimal(u.nS)))

    def test_nonlinear_target_merge_and_heterogeneous_reuse_are_not_equivalent(self):
        pre=SaturatingSynapse((2,))
        edge=SaturatingSynapse((2,2),alpha=jnp.asarray([[1.,2.],[1.,3.]])/u.ms)
        target=SaturatingSynapse((1,))
        pre.init_state();edge.init_state();target.init_state()
        source=pre(jnp.ones(2))
        heterogeneous=edge(jnp.ones((2,2)))
        merged=target(jnp.asarray([2.]))
        self.assertGreater(abs(float(source.sum()-merged[0])),1e-3)
        self.assertGreater(abs(float(heterogeneous[:,1].sum()-source.sum())),1e-3)
        self.assertTrue(np.all((np.asarray(heterogeneous)>=0)&(np.asarray(heterogeneous)<=1)))

    def test_multirate_conservation_causality_reset_and_zero_limit(self):
        model,result=run_multirate(ratio=5)
        events=np.asarray(result["events"])
        counts=events.sum(axis=1)
        np.testing.assert_allclose(result["rate"].to_decimal(u.Hz)*float(model.coarse_dt.to_decimal(u.second)),counts)
        np.testing.assert_array_equal(result["held"][0],0.)
        np.testing.assert_array_equal(result["held"][1:],result["mass"][:-1])
        first=np.flatnonzero(counts[:,0])[0]
        np.testing.assert_array_equal(result["mass"][:first],0.)
        np.testing.assert_array_equal(result["held"][:first+1],0.)
        self.assertGreater(float(np.abs(result["mass"][first,0])),0.)
        _,replay=run_multirate(ratio=5)
        np.testing.assert_array_equal(replay["mass"],result["mass"])
        _,zero=run_multirate(ratio=5,gain=0.)
        brainstate.environ.set(dt=.1*u.ms)
        independent=LIF(1);independent.init_state()
        spikes=brainstate.transform.for_loop(lambda _:independent(.5*u.nA),jnp.arange(200))
        np.testing.assert_array_equal(np.asarray(zero["events"]).reshape(200,1),spikes)
        np.testing.assert_array_equal(zero["mass"],0.)
        print("Multirate final mass",np.asarray(result["mass"][-1]).tolist())

    def test_multirate_exchange_refinement_is_measured(self):
        final=[]
        for ratio in [5,2,1]:
            _,result=run_multirate(ratio=ratio)
            final.append(float(result["mass"][-1,0]))
        self.assertTrue(np.isfinite(final).all())
        print("Exchange .5/.2/.1 ms final mass",final)

    def test_direct_delay_array_and_spec_rejections(self):
        delay=jnp.asarray([.2,.5])*u.ms
        self.assertEqual(delay_arrivals(delay),[[13],[16]])
        with self.assertRaisesRegex(TypeError,"scalar"):
            delay_arrivals(delay,"spec")
        with self.assertRaisesRegex(TypeError,"scalar"):
            delay_arrivals(delay,"continuous")
        with self.assertRaisesRegex(ValueError,"not settable"):
            delay_arrivals(delay,"set")

    def test_full_edge_matrix_requires_grouped_connections(self):
        self.assertEqual(neuron_delay_matrix("scalar"),[[15],[15]])
        self.assertEqual(neuron_delay_matrix("grouped"),[[14,16],[15,17]])
        with self.assertRaisesRegex(TypeError,"equal types"):
            neuron_delay_matrix("matrix")

    def test_fractional_default_and_continuous_trace_interpolation(self):
        low=voltage_trace(.2)
        high=voltage_trace(.3)
        expected=.6*low+.4*high
        for path in ["direct","continuous"]:
            actual=voltage_trace(.24,path)
            np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
        print("Fractional PSC interpolation max error",np.max(np.abs(actual-expected)))

    def test_explicit_spec_delay_point_has_amplitude_anomaly(self):
        expected=voltage_trace(.3,"direct")
        for path in ["spec","continuous"]:
            actual=voltage_trace(.3,path)
            difference=float(np.max(np.abs(actual-expected)))
            self.assertGreater(difference,.5)
            print("Observed explicit-spec delay .3 ms discrepancy",path,difference)

    def test_custom_zero_argument_model_and_unadapted_rejection(self):
        sim=brainspike.Simulator(dt=.1*u.ms)
        pop=sim.create(AutonomousLIF,1)
        rec=sim.create(brainspike.spike_recorder);sim.connect(pop,rec)
        self.assertGreater(float(sim.simulate(10*u.ms).spikes(rec).sum()),0)
        bad=brainspike.Simulator(dt=.1*u.ms)
        bad.create(LIF,1)
        with self.assertRaisesRegex(TypeError,"current"):
            bad.simulate(1*u.ms)

    def test_continuation_preserves_local_objective_gradient(self):
        full,full_gradient=jax.value_and_grad(window_voltage)(jnp.asarray(.5))
        windowed,windowed_gradient=jax.value_and_grad(lambda w:window_voltage(w,True))(jnp.asarray(.5))
        np.testing.assert_allclose(full,windowed,atol=1e-12)
        np.testing.assert_allclose(full_gradient,windowed_gradient,atol=1e-12)
        self.assertGreater(float(full_gradient),0.)
        print("Continuous/windowed objective and gradient",float(full),float(full_gradient))

    def test_hard_count_zero_gradient_and_derivative_free_alternative(self):
        count=lambda current:jnp.sum(rollout(jnp.asarray(.5),current=current,duration=20)[1].astype(jnp.float64))
        value,gradient=jax.value_and_grad(count)(jnp.asarray(10.))
        self.assertGreater(float(value),0.)
        self.assertEqual(float(gradient),0.)
        fit=fit_hard_count()
        self.assertEqual(fit["count"],5.)
        self.assertEqual(fit["loss"],0.)
        print("Derivative-free hard-count fit",fit)


if __name__=="__main__":
    unittest.main(verbosity=2)
