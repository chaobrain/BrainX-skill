"""BrainState spiking neural-network training reference.

Purpose: full representative SNN training workflow; use when the task
crosses from simulation into optimization.

Source mirrored:
https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/05_training_an_snn.html
"""

import brainunit as u
import brainpy
import brainstate
import braintools
import matplotlib.pyplot as plt


brainstate.random.seed(0)
brainstate.environ.set(dt=1.0 * u.ms)

num_inputs, num_hidden, num_outputs = 100, 4, 2
num_steps, batch_size = 100, 128


class SNN(brainstate.nn.Module):
    """Small surrogate-gradient spiking classifier."""

    def __init__(self, n_in, n_rec, n_out):
        super().__init__()
        decay = 1.0 - u.math.exp(-brainstate.environ.get_dt() / (1.0 * u.ms))
        self.i2r = brainstate.nn.Sequential(
            brainstate.nn.Linear(
                n_in,
                n_rec,
                w_init=braintools.init.KaimingNormal(scale=7.0 * decay, unit=u.mA),
                b_init=braintools.init.ZeroInit(unit=u.mA),
            ),
            brainpy.state.Expon(
                n_rec,
                tau=10.0 * u.ms,
                g_initializer=braintools.init.Constant(0.0 * u.mA),
            ),
        )
        self.r = brainpy.state.LIF(
            n_rec,
            tau=20.0 * u.ms,
            V_rest=0.0 * u.mV,
            V_reset=0.0 * u.mV,
            V_th=1.0 * u.mV,
            spk_fun=braintools.surrogate.ReluGrad(),
        )
        self.r2o = brainstate.nn.Linear(
            n_rec, n_out, w_init=braintools.init.KaimingNormal()
        )
        self.o = brainpy.state.Expon(
            n_out, tau=10.0 * u.ms, g_initializer=braintools.init.Constant(0.0)
        )

    def update(self, spike):
        return self.o(self.r2o(self.r(self.i2r(spike))))


net = SNN(num_inputs, num_hidden, num_outputs)

firing_rate = 5.0 * u.Hz
x_data = (
    brainstate.random.rand(num_steps, batch_size, num_inputs)
    < firing_rate * brainstate.environ.get_dt()
)
y_data = u.math.asarray(brainstate.random.rand(batch_size) < 0.5, dtype=int)

optimizer = braintools.optim.Adam(lr=3e-3)
optimizer.register_trainable_weights(net.states(brainstate.ParamState))


def accuracy():
    brainstate.nn.init_all_states(net, batch_size=batch_size)
    outs = brainstate.transform.for_loop(net.update, x_data)
    pred = u.math.argmax(u.math.max(outs, axis=0), axis=1)
    return float(u.math.mean(pred == y_data))


def loss_fn():
    outs = brainstate.transform.for_loop(net.update, x_data)
    outs = u.math.mean(outs, axis=0)
    return braintools.metric.softmax_cross_entropy_with_integer_labels(
        outs, y_data
    ).mean()


@brainstate.transform.jit
def train_step():
    brainstate.nn.init_all_states(net, batch_size=batch_size)
    grads, loss = brainstate.transform.grad(
        loss_fn, net.states(brainstate.ParamState), return_value=True
    )()
    optimizer.update(grads)
    return loss


def main():
    print(f"accuracy before training: {accuracy():.3f}")
    losses = []
    for epoch in range(1, 801):
        losses.append(float(train_step()))
        if epoch % 100 == 0:
            print(f"epoch {epoch:4d} | loss {losses[-1]:.4f}")

    print(f"accuracy after training: {accuracy():.3f}")
    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.title("Surrogate-gradient SNN training")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
