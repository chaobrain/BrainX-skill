"""LIFNeuron state-management reference.

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/01_state_and_pytrees.html
"""

import jax.numpy as jnp
import brainstate


class LIFNeuron(brainstate.nn.Module):
    """Leaky Integrate-and-Fire neuron model."""

    def __init__(self, n_neurons, tau=10.0, V_th=1.0, V_reset=0.0):
        super().__init__()
        self.tau = tau
        self.V_th = V_th
        self.V_reset = V_reset

        # Hidden state: membrane potential (evolves continuously)
        self.V = brainstate.HiddenState(jnp.full(n_neurons, V_reset))

        # Short-term state: refractory period counter
        self.t_last_spike = brainstate.ShortTermState(jnp.full(n_neurons, -1e7))

        # Parameters: input weights
        self.w_in = brainstate.ParamState(brainstate.random.randn(n_neurons, n_neurons) * 0.1)

    def __call__(self, I_ext, t):
        # Membrane potential dynamics
        dV = (-self.V.value + I_ext) / self.tau
        self.V.value = self.V.value + dV

        # Spike generation
        spike = self.V.value >= self.V_th

        # Reset
        self.V.value = jnp.where(spike, self.V_reset, self.V.value)
        self.t_last_spike.value = jnp.where(spike, t, self.t_last_spike.value)

        return spike

# Create and test the neuron
neuron = LIFNeuron(n_neurons=5)
print("Initial state:")
print(f"V: {neuron.V.value}")

# Simulate
for t in range(20):
    I_ext = jnp.ones(5) * 0.2  # External current
    spikes = neuron(I_ext, t)
    if jnp.any(spikes):
        print(f"t={t}: Spikes at neurons {jnp.where(spikes)[0]}")
