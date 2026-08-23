# Prebuilt BrainTrace layers

Use this reference when selecting and composing the default ETP-aware
`braintrace.nn` layers. Prefer these layers before built-in ETP operators or a
custom primitive; import activation, normalization, and pooling layers from
their owning package instead.

## Select by parameterized operation

`braintrace.nn` mirrors the BrainState layer pattern but routes each trainable
forward operation through an ETP primitive, so parameters participate without
manual trace wiring.

| Family | Use when | ETP behavior |
|---|---|---|
| Linear maps | The layer applies dense, grouped, signed, sparse, or low-rank weights. | Uses the matching ETP matrix operation. |
| Embeddings | Integer indices select trainable rows. | Uses ETP-aware indexed weight access. |
| Convolutions | Data has one, two, or three spatial dimensions. | Uses `braintrace.conv()`. |
| Recurrent cells | Hidden State must update once per input step. | Uses ETP dense and element-wise operations inside the recurrence. |
| Readouts | Recurrent activity needs leaky continuous output dynamics. | Uses an ETP-aware projection. |

The class namespace is convenient, but the operation in the traced graph is
what the compiler recognizes.

## Choose a linear map

Use the representation that matches the weight structure; do not emulate a
specialized structure with an unrelated dense layer.

| API | Description |
|---|---|
| `braintrace.nn.Linear` | Use for a standard dense linear transformation; its weight passes through `braintrace.matmul()`. |
| `braintrace.nn.GroupedLinear` | Use for a block-diagonal grouped transformation backed by `braintrace.grouped_matmul()`. |
| `braintrace.nn.SignedWLinear` | Use for a linear layer with signed absolute weights. |
| `braintrace.nn.SparseLinear` | Use for a sparse weight matrix. |
| `braintrace.nn.LoRA` | Use for a standalone low-rank adaptation layer. |

## Choose an embedding or convolution

Use embeddings for trainable indexed lookup and choose convolution dimensionality
from the number of spatial axes.

| API | Description |
|---|---|
| `braintrace.nn.Embedding` | Use for a fixed-size trainable embedding table. |
| `braintrace.nn.Conv1d` | Use for one-dimensional convolution. |
| `braintrace.nn.Conv2d` | Use for two-dimensional convolution. |
| `braintrace.nn.Conv3d` | Use for three-dimensional convolution. |

## Choose a recurrent cell

Recurrent layers consume one time step, update hidden State in place, and
return the new hidden State; `LRUCell` instead returns its projected output.

| API | Description |
|---|---|
| `braintrace.nn.ValinaRNNCell` | Use for the package's vanilla RNN cell. Preserve the exported identifier spelling. |
| `braintrace.nn.GRUCell` | Use for a standard gated recurrent unit. |
| `braintrace.nn.MGUCell` | Use for a minimal gated recurrent unit. |
| `braintrace.nn.LSTMCell` | Use for a long short-term memory core. |
| `braintrace.nn.URLSTMCell` | Use for an update-reset LSTM core. |
| `braintrace.nn.MinimalRNNCell` | Use for the minimal RNN cell. |
| `braintrace.nn.MiniGRU` | Use for the package's minimal GRU cell. |
| `braintrace.nn.MiniLSTM` | Use for the package's minimal LSTM cell. |
| `braintrace.nn.LRUCell` | Use for a linear recurrent unit; it returns the projected output rather than the hidden State itself. |

## Choose a readout

Use the BrainTrace readout when an online recurrent model needs leaky output
dynamics.

| API | Description |
|---|---|
| `braintrace.nn.LeakyRateReadout` | Use for leaky readout dynamics in Real-Time Recurrent Learning. |

## Import supporting layers from their owner

Activation, normalization, and pooling layers are intentionally not
re-implemented in `braintrace.nn`.

| Need | Import from |
|---|---|
| Activation | `brainstate.nn` |
| Normalization | `brainstate.nn` |
| Pooling | `brainstate.nn` |

Do not import these through `braintrace.nn`. Compatibility forwarding, such as
`braintrace.nn.LayerNorm`, emits `DeprecationWarning` and forwards to the owning
package.

## Compose and compile a recurrent model

Put the recurrent cell and observable readout in one Module, compile from a
representative step, then call the learner from a BrainState loop for repeated
execution.

```python
import jax.numpy as jnp
import brainstate
import braintrace


class TinySequenceModel(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = braintrace.nn.MiniGRU(in_size=1, out_size=4)
        self.readout = braintrace.nn.Linear(4, 1)

    def update(self, x):
        return self.readout(self.rnn(x))


model = TinySequenceModel()
learner = braintrace.compile(
    model,
    braintrace.D_RTRL,
    jnp.ones(1),
)

sequence = jnp.linspace(-1.0, 1.0, 6).reshape(6, 1)
brainstate.nn.reset_all_states(model)
learner.reset_state()
outputs = brainstate.transform.for_loop(learner, sequence)

assert outputs.shape == (6, 1)
```

The recurrent weights influence hidden State and therefore need eligibility
traces. The final `Linear` weight remains trainable, but because its output does
not feed hidden State, the compiler classifies it as non-temporal and computes
its instantaneous gradient directly.

**Relation invariant:** A `weight -> weight -> hidden` path crosses two
trainable ETP operations. The compiler stops at the downstream operation and
does not record the upstream weight as an independent ETP relation, which
prevents double counting.

Open `ETP operators.md` when no prebuilt layer matches the required custom
block. Escalate to `custom ETP primitives.md` only when built-in ETP operations
cannot express it.

## Sources

- [Neural-Network Layers](https://brainx.chaobrain.com/braintrace/apis/nn.html)
- [Neural Network Layers for Online Learning](https://brainx.chaobrain.com/braintrace/tutorials/neural_network_layers.html)

The layer catalogue, ownership boundary, composition, and compiler relations
above are trimmed and reorganized from the official pages without changing
their API names.
