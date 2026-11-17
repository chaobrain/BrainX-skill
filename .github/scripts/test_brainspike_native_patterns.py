"""Regression tests for the replacement native modeling and temporal patterns."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"skills/package-skills/brainspike/scripts"))
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from native_ei_network import EINetwork,simulate
from delay_workflow import DelayBank,DelayedProjection
from delayed_cue_training import DelayedCueSNN,trials,masked_loss,sequence_loss,train


class NativePatternTests(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(precision=64,dt=0.1*u.ms)
        brainstate.random.seed(7)

    def test_ei_signs_state_shape_and_activity(self):
        net,outputs = simulate()
        self.assertEqual(outputs["spikes"].shape,(300,20))
        self.assertEqual(net.syn_exc.g.value.shape,(20,))
        self.assertEqual(net.syn_inh.g.value.shape,(20,))
        for name in ["g_exc","g_inh"]:
            self.assertTrue(np.all(outputs[name].to_decimal(u.nS)>=0))
            self.assertGreater(float(u.math.max(outputs[name]).to_decimal(u.nS)),0)
        self.assertTrue(np.all(outputs["i_exc"].to_decimal(u.pA)>=0))
        self.assertTrue(np.all(outputs["i_inh"].to_decimal(u.pA)<=0))
        self.assertTrue(np.isfinite(outputs["voltage"].to_decimal(u.mV)).all())
        self.assertGreater(float(outputs["spikes"].sum()),0)

    def test_projection_order_and_independent_reset(self):
        net = EINetwork(n_exc=2,n_inh=1)
        net.w_exc = jnp.ones((2,3))*0.3*u.nS
        net.w_inh = jnp.ones((1,3))*0.8*u.nS
        brainstate.nn.init_all_states(net)
        net.exc.spike.value = jnp.asarray([1.,0.])
        net.inh.spike.value = jnp.asarray([1.])
        result = net(0*u.nA)
        np.testing.assert_allclose(result["g_exc"].to_decimal(u.nS),0.3)
        np.testing.assert_allclose(result["g_inh"].to_decimal(u.nS),0.8)
        np.testing.assert_allclose(result["voltage"].to_decimal(u.mV),-65+0.0075,atol=1e-10)
        second = net(0*u.nA)
        np.testing.assert_allclose(second["g_exc"].to_decimal(u.nS),0.3*np.exp(-.1/5))
        brainstate.nn.reset_all_states(net)
        np.testing.assert_array_equal(net.syn_exc.g.value.to_decimal(u.nS),np.zeros(3))
        np.testing.assert_array_equal(net.exc.get_spike(),np.zeros(2))

    def test_heterogeneous_delay_impulses_and_reset(self):
        bank = DelayBank()
        brainstate.nn.init_all_states(bank)
        impulses = jnp.zeros((6,3),dtype=jnp.bool_).at[0].set(True)
        expected = np.zeros((6,3),dtype=bool)
        expected[[0,2,3],[0,1,2]] = True
        actual = brainstate.transform.for_loop(bank,impulses)
        np.testing.assert_array_equal(actual,expected)
        brainstate.nn.init_all_states(bank)
        empty = brainstate.transform.for_loop(bank,jnp.zeros_like(impulses))
        self.assertFalse(bool(empty.any()))
        brainstate.nn.init_all_states(bank)
        replay = brainstate.transform.for_loop(bank,impulses)
        np.testing.assert_array_equal(replay,expected)
        projection = DelayedProjection()
        brainstate.nn.init_all_states(projection)
        source = jnp.zeros((6,3),dtype=jnp.bool_).at[0,1].set(True)
        result = brainstate.transform.for_loop(projection,source)
        conductance = np.asarray(result["conductance"].to_decimal(u.nS))[:,0]
        voltage = np.asarray(result["voltage"].to_decimal(u.mV))[:,0]
        np.testing.assert_array_equal(conductance[:2],0.)
        self.assertAlmostEqual(conductance[2],.7)
        np.testing.assert_array_equal(voltage[:2],-65.)
        self.assertGreater(voltage[2],-65.)
        brainstate.nn.init_all_states(projection)
        quiet = brainstate.transform.for_loop(projection,jnp.zeros_like(source))
        np.testing.assert_array_equal(quiet["conductance"].to_decimal(u.nS),0.)


    def test_report_mask_and_checkpointed_gradient_parity(self):
        brainstate.environ.set(dt=1*u.ms)
        inputs,targets,mask = trials(jnp.asarray([.6,.8,1.2,1.4]))
        self.assertTrue(np.all(np.asarray(inputs[5:])==0))
        outputs = jnp.zeros((20,4,1))
        altered = outputs.at[:15].set(100.)
        np.testing.assert_equal(masked_loss(outputs,targets,mask),masked_loss(altered,targets,mask))
        model = DelayedCueSNN()
        brainstate.nn.init_all_states(model,batch_size=4)
        params = model.states(brainstate.ParamState)
        def differentiate(checkpointed):
            f = lambda: sequence_loss(model,inputs,targets,mask,checkpointed=checkpointed)
            return brainstate.transform.grad(f,params,return_value=True)()
        ordinary,ordinary_loss = differentiate(False)
        checkpointed,checkpointed_loss = differentiate(True)
        np.testing.assert_allclose(ordinary_loss,checkpointed_loss,rtol=1e-10,atol=1e-12)
        for a,b in zip(jax.tree.leaves(ordinary),jax.tree.leaves(checkpointed)):
            np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10)
        self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves(checkpointed)))
        for key,group in checkpointed.items():
            self.assertGreater(float(sum(jnp.sum(v*v) for v in jax.tree.leaves(group))),0,str(key))

    def test_delayed_cue_training_and_ablation(self):
        _,metrics = train()
        print("Delayed-cue metrics:",metrics)
        self.assertLess(metrics["final_loss"],metrics["initial_loss"]*.2)
        self.assertEqual(metrics["held_accuracy"],1.)
        self.assertEqual(metrics["cue_ablated_accuracy"],.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
