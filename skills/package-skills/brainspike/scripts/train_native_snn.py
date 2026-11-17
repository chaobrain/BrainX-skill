"""Run: python scripts/train_native_snn.py --estimator bptt (or pp-prop)."""
import argparse
import json
import brainstate
import braintrace
import braintools
import brainunit as u
from native_workflows import RecurrentSNN


def train(estimator="bptt", epochs=40):
    brainstate.environ.set(precision=64, dt=0.1 * u.ms)
    brainstate.random.seed(42)
    model = RecurrentSNN()
    batched = estimator == "pp-prop"
    batch = 1 if batched else None
    inputs = u.math.ones((80, 1, 1) if batched else (80, 1))
    if batched:
        learner = braintrace.compile(model, braintrace.pp_prop, inputs[0], batch_size=1, decay_or_rank=0.9)
        assert learner.report.counts["errors"] == 0 and learner.report.etrace_weights
    else:
        brainstate.nn.init_all_states(model)
    params = model.states(brainstate.ParamState)
    optimizer = braintools.optim.Adam(lr=0.01)
    optimizer.register_trainable_weights(params)

    def reset():
        brainstate.nn.reset_all_states(model, batch_size=batch)
        if batched:
            learner.reset_state(batch_size=1)

    def loss():
        reset()
        output = brainstate.transform.for_loop(model, inputs)
        return u.math.mean((output - 0.25)**2)

    differentiate = brainstate.transform.grad(loss, params, return_value=True)

    @brainstate.transform.jit
    def train_step():
        if batched:
            reset()
            grads, values = learner.etrace_grad(
                inputs, step_fn=lambda x: u.math.mean((learner(x)-0.25)**2),
                reduction="mean", return_value=True,
            )
            value = values.mean()
        else:
            grads, value = differentiate()
        optimizer.update(grads)
        return value

    initial = float(loss())
    for _ in range(epochs):
        train_step()
    final = float(loss())
    assert final < initial
    return {"estimator": estimator, "initial_loss": initial, "final_loss": final, "epochs": epochs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", choices=["bptt", "pp-prop"], default="bptt")
    args = parser.parse_args()
    print(json.dumps(train(args.estimator)))
