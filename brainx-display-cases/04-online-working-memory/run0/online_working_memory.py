"""Online match/non-match working memory in a recurrent spiking network.

The first cue is followed by an input-free delay. A second cue then arrives,
and a two-class readout reports non-match (0) or match (1). BrainTrace pp-prop
updates eligibility traces online, so training memory does not grow with the
sequence length.
"""

import numpy as np
import jax.numpy as jnp

import brainevent
import brainpy
import brainstate
import braintrace
import braintools
import brainunit as u


DT = 1.0 * u.ms
CUE_DURATION = 5.0 * u.ms
DELAY_DURATION = 40.0 * u.ms
REPORT_DURATION = 10.0 * u.ms

N_CUE = 2
N_REC = 48
N_CLASS = 2
OUT_DEGREE = 8
BATCH_SIZE = 16
N_UPDATES = 320


def fixed_degree_csr(n_neuron: int, out_degree: int, seed: int):
    """Create a reproducible sparse recurrent topology without self-edges."""
    rng = np.random.default_rng(seed)
    candidates = np.arange(n_neuron)
    targets = np.stack(
        [
            rng.choice(candidates[candidates != source], out_degree, replace=False)
            for source in range(n_neuron)
        ]
    ).astype(np.int32)
    indices = targets.reshape(-1)
    indptr = np.arange(n_neuron + 1, dtype=np.int32) * out_degree
    topology = brainevent.CSR(
        (jnp.ones(indices.size), indices, indptr),
        shape=(n_neuron, n_neuron),
    )
    return topology, indices.size


class WorkingMemorySNN(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent_topology, n_synapse = fixed_degree_csr(
            N_REC, OUT_DEGREE, seed=11
        )
        self.recurrent_weight = brainstate.ParamState(
            brainstate.random.normal(size=n_synapse) * 3.0 * u.mA
        )
        self.cue_projection = braintrace.nn.Linear(
            N_CUE,
            N_REC,
            w_init=braintools.init.KaimingNormal(scale=24.0, unit=u.mA),
            b_init=braintools.init.ZeroInit(unit=u.mA),
        )
        self.neurons = brainpy.state.LIF(
            N_REC,
            tau=20.0 * u.ms,
            R=1.0 * u.ohm,
            V_rest=0.0 * u.mV,
            V_reset=0.0 * u.mV,
            V_th=1.0 * u.mV,
            spk_fun=braintools.surrogate.ReluGrad(),
            spk_reset="soft",
        )
        self.readout = braintrace.nn.LeakyRateReadout(
            N_REC,
            N_CLASS,
            tau=20.0 * u.ms,
            w_init=braintools.init.KaimingNormal(),
        )

    def update(self, cue):
        previous_spikes = (self.neurons.get_spike() != 0.0).astype(jnp.float32)
        recurrent_current = braintrace.sparse_matmul(
            previous_spikes,
            self.recurrent_weight.value,
            sparse_mat=self.recurrent_topology,
        )
        spikes = self.neurons(self.cue_projection(cue) + recurrent_current)
        return self.readout(spikes)


def make_trials(batch_size: int):
    total_duration = (
        2.0 * CUE_DURATION + DELAY_DURATION + REPORT_DURATION
    )
    times = u.math.arange(0.0 * u.ms, total_duration, DT)

    # Repeating all four cue pairs keeps match/non-match and both cue identities
    # balanced in every online update.
    pair = jnp.arange(batch_size) % 4
    first = pair // 2
    second = pair % 2
    labels = (first == second).astype(jnp.int32)

    first_window = times < CUE_DURATION
    second_start = CUE_DURATION + DELAY_DURATION
    second_window = (times >= second_start) & (
        times < second_start + CUE_DURATION
    )
    report_window = times >= second_start + CUE_DURATION

    first_cue = jnp.eye(N_CUE)[first]
    second_cue = jnp.eye(N_CUE)[second]
    inputs = (
        first_window[:, None, None] * first_cue[None, :, :]
        + second_window[:, None, None] * second_cue[None, :, :]
    ).astype(jnp.float32)
    report_mask = report_window.astype(jnp.float32)

    delay_window = (times >= CUE_DURATION) & (times < second_start)
    assert not bool(jnp.any(inputs[delay_window]))
    assert inputs.shape == (times.shape[0], batch_size, N_CUE)
    return times, inputs, labels, report_mask


def report_logits(outputs, report_mask):
    weights = report_mask[:, None, None]
    return jnp.sum(outputs * weights, axis=0) / jnp.sum(weights)


def main():
    brainstate.random.seed(7)

    with brainstate.environ.context(dt=DT):
        times, inputs, labels, report_mask = make_trials(BATCH_SIZE)

        model = WorkingMemorySNN()
        # Map is BrainState's stateful vmap wrapper: parameters are shared while
        # membrane, spike, readout, and trace State are independent per example.
        mapped_model = brainstate.nn.Map(model, init_map_size=BATCH_SIZE)
        mapped_model.init_all_states()

        learner = braintrace.pp_prop(mapped_model, decay_or_rank=0.9)
        learner.compile_graph(inputs[0])
        learner.report.show(1)

        optimizer = braintools.optim.Adam(lr=2e-3)
        optimizer.register_trainable_weights(learner.param_states)
        current_logits = brainstate.ShortTermState(
            jnp.zeros((BATCH_SIZE, N_CLASS), dtype=jnp.float32)
        )
        mapped_hidden = mapped_model.states(brainstate.HiddenState)

        def reset_sequence():
            for state in mapped_hidden.values():
                state.value = u.math.zeros_like(state.value)
            learner.reset_state()

        def evaluate(sequence):
            reset_sequence()

            def step(cue):
                def probe_loss():
                    logits = learner(cue)
                    current_logits.value = logits
                    return jnp.mean(logits**2)

                brainstate.transform.grad(
                    probe_loss,
                    learner.param_states,
                )()
                return current_logits.value

            outputs = brainstate.transform.for_loop(step, sequence)
            logits = report_logits(outputs, report_mask)
            loss = braintools.metric.softmax_cross_entropy_with_integer_labels(
                logits, labels
            ).mean()
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
            return loss, accuracy

        def online_update(_):
            reset_sequence()

            def step(cue, mask):
                def masked_loss():
                    logits = learner(cue)
                    current_logits.value = logits
                    loss = braintools.metric.softmax_cross_entropy_with_integer_labels(
                        logits, labels
                    ).mean()
                    return loss * mask

                grads, loss = brainstate.transform.grad(
                    masked_loss,
                    learner.param_states,
                    return_value=True,
                )()
                grads = brainstate.nn.clip_grad_norm(grads, 1.0)
                brainstate.transform.cond(
                    mask > 0.0,
                    optimizer.update,
                    lambda _: None,
                    grads,
                )
                return loss, current_logits.value

            step_losses, outputs = brainstate.transform.for_loop(
                step, inputs, report_mask
            )
            logits = report_logits(outputs, report_mask)
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
            return jnp.sum(step_losses) / jnp.sum(report_mask), accuracy

        def train_online():
            return brainstate.transform.for_loop(
                online_update, jnp.arange(N_UPDATES)
            )

        initial_loss, initial_accuracy = evaluate(inputs)
        losses, accuracies = train_online()
        final_loss, final_accuracy = evaluate(inputs)

    print(f"sequence: {times.shape[0]} steps at {DT}")
    print(f"input-free delay: {DELAY_DURATION}")
    print(
        f"before training: loss={float(initial_loss):.4f}, "
        f"accuracy={float(initial_accuracy):.3f}"
    )
    print(
        f"after {N_UPDATES} online updates: loss={float(final_loss):.4f}, "
        f"accuracy={float(final_accuracy):.3f}"
    )
    print(
        f"last 10 updates: loss={float(losses[-10:].mean()):.4f}, "
        f"accuracy={float(accuracies[-10:].mean()):.3f}"
    )

    assert jnp.isfinite(losses).all()
    assert final_loss < initial_loss
    assert final_accuracy >= 0.75


if __name__ == "__main__":
    main()
