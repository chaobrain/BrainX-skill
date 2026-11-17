"""Grid-aligned native delay taps, independent of retired BrainPy projections."""
import brainevent
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
from native_workflows import LIF, ExponentialConductance


class DelayBank(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        dt = brainstate.environ.get_dt()
        self.history = brainstate.nn.Delay(jax.ShapeDtypeStruct((3,),jnp.bool_), 3*dt)
        self.history.register_entry("axonal",jnp.asarray([0,2,3])*dt,jnp.arange(3))

    def update(self, spikes):
        self.history.update(spikes)
        return self.history.at("axonal")


class DelayedProjection(brainstate.nn.Module):
    """External binary events -> named taps -> conductance -> native point neuron."""
    def __init__(self):
        super().__init__()
        self.bank = DelayBank()
        self.synapse = ExponentialConductance(1)
        self.post = LIF(1)
        self.weights = jnp.asarray([[.5],[.7],[.9]])*u.nS

    def update(self,spikes):
        delayed = self.bank(spikes)
        increment = brainevent.BinaryArray(delayed) @ self.weights
        conductance = self.synapse(increment)
        self.post(conductance*(0*u.mV-self.post.V.value))
        return {"events":delayed,"conductance":conductance,"voltage":self.post.V.value}


if __name__ == "__main__":
    brainstate.environ.set(dt=0.1*u.ms)
    bank = DelayBank()
    brainstate.nn.init_all_states(bank)
    impulses = jnp.zeros((6,3),dtype=jnp.bool_).at[0].set(True)
    print(brainstate.transform.for_loop(bank,impulses))
