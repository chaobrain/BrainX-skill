"""Direct gradients and weight fitting through NEST Simulator.simulate()."""
import argparse
import json
import brainspike
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp

brainstate.environ.set(precision=64)


def rollout(weight, comm="dense", current=0.0, duration=5.0):
    # All state is local: each objective evaluation starts a fresh independent run.
    sim = brainspike.Simulator(dt=0.1*u.ms)
    source = sim.create(brainspike.spike_generator, spike_times=jnp.asarray([1., 2., 3.])*u.ms)
    target = sim.create(brainspike.iaf_psc_alpha, 1, params={
        "I_e": current*100*u.pA,
        "V_initializer": braintools.init.Constant(-65*u.mV),
    })
    sim.connect(source, target, weight=weight*100*u.pA, delay=0.1*u.ms, comm=comm)
    meter = sim.create(brainspike.multimeter, record_from=["V_m"], interval=0.1*u.ms)
    recorder = sim.create(brainspike.spike_recorder)
    sim.connect(meter, target)
    sim.connect(target, recorder)
    result = sim.simulate(duration*u.ms)
    # Explicit unit boundary for a scalar, dimensionless optimization objective.
    return result.trace(meter, "V_m").to_decimal(u.mV)[:, 0], result.spikes(recorder)[:, 0]


def experiment(comm="dense", epochs=100):
    target_weight = 1.7
    observed, _ = rollout(jnp.asarray(target_weight), comm)
    parameter = brainstate.ParamState(jnp.asarray(0.4))
    params = {"weight": parameter}
    optimizer = braintools.optim.Adam(lr=0.05)
    optimizer.register_trainable_weights(params)

    def objective(weight):
        voltage, _ = rollout(weight, comm)
        return jnp.mean((voltage-observed)**2)

    # The local graph is a pure input-output function; no persistent State crosses JAX.
    value_and_grad = jax.value_and_grad(objective)
    initial, gradient = value_and_grad(parameter.value)
    epsilon = 1e-4
    finite_difference = (objective(parameter.value+epsilon)-objective(parameter.value-epsilon))/(2*epsilon)

    # Keep host-side graph construction outside an outer JIT.
    def train_step():
        value, grad = value_and_grad(parameter.value)
        optimizer.update({"weight": grad})
        return value

    for _ in range(epochs):
        train_step()
    final = objective(parameter.value)
    return {"comm": comm, "initial_weight": 0.4, "target_weight": target_weight,
            "learned_weight": float(parameter.value), "initial_loss": float(initial),
            "final_loss": float(final), "autodiff_gradient": float(gradient),
            "finite_difference_gradient": float(finite_difference), "epochs": epochs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--comm", choices=["dense", "sparse"], default="dense")
    args = parser.parse_args()
    print(json.dumps(experiment(args.comm)))
