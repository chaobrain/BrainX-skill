"""Online delayed-match-to-sample training with a recurrent spiking network.

The first one-hot cue is followed by a zero-input delay and a second cue.  The
network reports nonmatch (0) or match (1) only after the second cue has ended.
BrainTrace accumulates eligibility-trace gradients online, so the temporal
training memory does not grow with the sequence length as BPTT memory does.
"""

from __future__ import annotations

import argparse

import brainevent
import brainpy
import brainstate
import braintools
import braintrace
import brainunit as u
import jax
import jax.numpy as jnp


DT = 1.0 * u.ms
CUE_DURATION = 8.0 * u.ms
DELAY_DURATION = 30.0 * u.ms
REPORT_DURATION = 10.0 * u.ms
N_CUES = 2


def _num_steps(duration: u.Quantity["time"]) -> int:
    return int(round(float(duration / DT)))


CUE_STEPS = _num_steps(CUE_DURATION)
DELAY_STEPS = _num_steps(DELAY_DURATION)
REPORT_STEPS = _num_steps(REPORT_DURATION)
TEST_START = CUE_STEPS + DELAY_STEPS
REPORT_START = TEST_START + CUE_STEPS
NUM_STEPS = REPORT_START + REPORT_STEPS
LOSS_MASK = (jnp.arange(NUM_STEPS) >= REPORT_START).astype(jnp.float32)


def make_trial_stream(key, num_batches: int, batch_size: int):
    """Return inputs [batch stream, time, lane, cue] and match labels."""
    sample_key, match_key = jax.random.split(key)
    sample = jax.random.randint(sample_key, (num_batches, batch_size), 0, N_CUES)
    is_match = jax.random.bernoulli(match_key, 0.5, (num_batches, batch_size))
    test = jnp.where(is_match, sample, 1 - sample)

    sample_cue = jax.nn.one_hot(sample, N_CUES, dtype=jnp.float32)
    test_cue = jax.nn.one_hot(test, N_CUES, dtype=jnp.float32)
    inputs = jnp.zeros(
        (num_batches, NUM_STEPS, batch_size, N_CUES), dtype=jnp.float32
    )
    inputs = inputs.at[:, :CUE_STEPS].set(sample_cue[:, None])
    inputs = inputs.at[:, TEST_START:REPORT_START].set(test_cue[:, None])
    return inputs, is_match.astype(jnp.int32)


def make_balanced_trials(batch_size: int):
    """Create a fixed evaluation batch containing all four cue pairs."""
    lane = jnp.arange(batch_size)
    sample = lane % N_CUES
    is_match = (lane // N_CUES) % 2
    test = jnp.where(is_match.astype(bool), sample, 1 - sample)
    inputs = jnp.zeros((NUM_STEPS, batch_size, N_CUES), dtype=jnp.float32)
    inputs = inputs.at[:CUE_STEPS].set(jax.nn.one_hot(sample, N_CUES))
    inputs = inputs.at[TEST_START:REPORT_START].set(
        jax.nn.one_hot(test, N_CUES)
    )
    return inputs, is_match.astype(jnp.int32)


class SparseRecurrentCommunication(brainstate.nn.Module):
    """Trainable currents on a fixed BrainEvent CSR recurrent graph."""

    def __init__(self, size: int, fanout: int):
        super().__init__()
        targets = (
            jnp.arange(size)[:, None] + jnp.arange(1, fanout + 1)[None, :]
        ) % size
        indices = targets.reshape(-1).astype(jnp.int32)
        indptr = jnp.arange(0, size * fanout + 1, fanout, dtype=jnp.int32)
        self.topology = brainevent.CSR(
            (jnp.ones(indices.size), indices, indptr), shape=(size, size)
        )
        self.weight = brainstate.ParamState(
            brainstate.random.randn(indices.size) * (0.25 * u.mA)
        )

    def update(self, spikes):
        return braintrace.sparse_matmul(
            spikes,
            self.weight.value,
            sparse_mat=self.topology,
        )


class WorkingMemorySNN(brainstate.nn.Module):
    def __init__(self, size: int = 48, fanout: int = 8):
        super().__init__()
        self.neuron = brainpy.state.ALIF(
            size,
            R=1.0 * u.ohm,
            tau=20.0 * u.ms,
            tau_a=100.0 * u.ms,
            V_th=1.0 * u.mV,
            beta=1.5 * u.mV,
            V_reset=0.0 * u.mV,
            V_rest=0.0 * u.mV,
            V_initializer=braintools.init.ZeroInit(unit=u.mV),
        )
        self.input = braintrace.nn.Linear(
            N_CUES,
            size,
            w_init=brainstate.random.randn(N_CUES, size) * (4.0 * u.mA),
            b_init=braintools.init.ZeroInit(unit=u.mA),
        )
        self.recurrent = brainpy.state.AlignPostProj(
            comm=SparseRecurrentCommunication(size, fanout),
            syn=brainpy.state.Expon(
                size,
                tau=8.0 * u.ms,
                g_initializer=braintools.init.ZeroInit(unit=u.mA),
            ),
            out=brainpy.state.CUBA(scale=1.0),
            post=self.neuron,
        )
        self.readout = braintrace.nn.LeakyRateReadout(
            size,
            2,
            tau=10.0 * u.ms,
            w_init=braintools.init.KaimingNormal(),
        )

    def update(self, cue):
        previous_spikes = self.neuron.get_spike()
        self.recurrent(previous_spikes)
        spikes = self.neuron(self.input(cue))
        return self.readout(spikes)


def report_logits(outputs):
    return jnp.sum(outputs * LOSS_MASK[:, None, None], axis=0) / jnp.sum(LOSS_MASK)


def accuracy(outputs, labels):
    return jnp.mean(jnp.argmax(report_logits(outputs), axis=-1) == labels)


def run(num_updates: int, batch_size: int, stream_size: int, seed: int):
    if batch_size < 4 or batch_size % 4:
        raise ValueError("batch_size must be a positive multiple of four")
    brainstate.random.seed(seed)
    with brainstate.environ.context(dt=DT):
        model = WorkingMemorySNN()
        example = jnp.zeros((batch_size, N_CUES), dtype=jnp.float32)
        learner = braintrace.compile(
            model,
            braintrace.pp_prop,
            example,
            batch_size=batch_size,
            vmap=True,
            decay_or_rank=0.97,
            vjp_method="single-step",
        )
        algorithm = learner.module
        if algorithm.report.counts["errors"]:
            algorithm.report.show(2)
            raise RuntimeError("BrainTrace compilation reported errors")
        if not algorithm.report.etrace_weights:
            raise RuntimeError("no recurrent parameter entered the eligibility graph")

        weights = model.states(brainstate.ParamState)
        optimizer = braintools.optim.Adam(lr=2e-3)
        optimizer.register_trainable_weights(weights)
        mapped_states = learner.states("new")

        @brainstate.transform.vmap(in_states=mapped_states)
        def reset_sequence():
            brainstate.nn.reset_all_states(learner)

        def step_loss(cue, labels, mask):
            output = learner(cue)
            loss = braintools.metric.softmax_cross_entropy_with_integer_labels(
                output, labels
            ).mean()
            return loss * mask, output

        step_grad = brainstate.transform.grad(
            step_loss,
            weights,
            has_aux=True,
            return_value=True,
        )

        def train_batch(inputs, labels):
            reset_sequence()
            initial_grads = jax.tree.map(
                jnp.zeros_like,
                {key: state.value for key, state in weights.items()},
            )

            def accumulate(grads, sample):
                current, loss, _ = step_grad(sample[0], labels, sample[1])
                grads = jax.tree.map(lambda total, value: total + value, grads, current)
                return grads, loss

            grads, losses = brainstate.transform.scan(
                accumulate,
                initial_grads,
                (inputs, LOSS_MASK),
            )
            optimizer.update(brainstate.nn.clip_grad_norm(grads, 1.0))
            return losses.sum() / LOSS_MASK.sum()

        @brainstate.transform.jit
        def train_stream(input_stream, label_stream):
            return brainstate.transform.for_loop(
                train_batch, input_stream, label_stream
            )

        @brainstate.transform.jit
        def evaluate(inputs):
            reset_sequence()
            return brainstate.transform.for_loop(learner, inputs)

        eval_inputs, eval_labels = make_balanced_trials(batch_size)
        initial_accuracy = accuracy(evaluate(eval_inputs), eval_labels)

        losses = []
        key = jax.random.PRNGKey(seed + 1)
        completed = 0
        while completed < num_updates:
            count = min(stream_size, num_updates - completed)
            key, data_key = jax.random.split(key)
            stream_inputs, stream_labels = make_trial_stream(
                data_key, count, batch_size
            )
            losses.extend(train_stream(stream_inputs, stream_labels).tolist())
            completed += count

        intact_accuracy = accuracy(evaluate(eval_inputs), eval_labels)
        ablated_inputs = eval_inputs.at[:CUE_STEPS].set(0.0)
        ablated_accuracy = accuracy(evaluate(ablated_inputs), eval_labels)

    assert bool(jnp.all(eval_inputs[CUE_STEPS:TEST_START] == 0.0))
    assert all(jnp.isfinite(jnp.asarray(losses)))
    return {
        "initial_accuracy": float(initial_accuracy),
        "final_loss": float(losses[-1]),
        "intact_accuracy": float(intact_accuracy),
        "ablated_accuracy": float(ablated_accuracy),
        "etrace_parameters": len(algorithm.report.etrace_weights),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stream-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.updates < 1 or args.stream_size < 1 or args.batch_size < 4 or args.batch_size % 4:
        parser.error(
            "updates and stream-size must be positive; "
            "batch-size must be a positive multiple of four"
        )

    result = run(args.updates, args.batch_size, args.stream_size, args.seed)
    print(
        f"delay={DELAY_DURATION}, updates={args.updates}, "
        f"initial_accuracy={result['initial_accuracy']:.3f}, "
        f"final_loss={result['final_loss']:.4f}"
    )
    print(
        f"intact_accuracy={result['intact_accuracy']:.3f}, "
        f"first_cue_ablated_accuracy={result['ablated_accuracy']:.3f}, "
        f"etrace_parameters={result['etrace_parameters']}"
    )


if __name__ == "__main__":
    main()
