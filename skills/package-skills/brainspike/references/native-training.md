# Train a native BrainSpike network

Use this path for a native graph built from public `brainspike.Neuron` and BrainState Modules. For direct NEST Simulator continuous-voltage fitting, open `references/simulator-training.md`. That path is separately verified and does not pass a Simulator graph to this learner.

Open `references/braintools/data-preprocessing.md` for encoded input axes and `references/braintools/parameter-initializer.md` for reproducible initial weights.

## Choose the temporal gradient

| API | Use |
|---|---|
| `brainstate.transform.grad(loss, params, return_value=True)` | Use for BPTT through the complete native rollout; select only `ParamState` and reset hidden State inside the loss. |
| `braintrace.nn.Linear` | Use for trainable projections and readouts that must participate in eligibility-trace compilation. |
| `braintrace.compile(model, braintrace.pp_prop, example_step, batch_size=..., decay_or_rank=0.9)` | Use for factorized online traces; compile one representative step and inspect the report before training. |
| `learner.etrace_grad(..., step_fn=..., reduction="mean", return_value=True)` | Advance every sequence step and accumulate the declared gradient reduction; update parameters after the sequence. |
| `learner.reset_state(batch_size=...)` | Reset eligibility State alongside the native graph at every independent sequence boundary. |
| `braintools.optim.Adam(...).register_trainable_weights(params)` | Register the same parameter collection used for differentiation, then update it with matching gradients. |

Open `scripts/train_native_snn.py` and run `python scripts/train_native_snn.py --estimator bptt` or `--estimator pp-prop` from this skill root. Both modes use the same recurrent spike model and trained readout. BPTT compiles the reset-gradient-update step; pp-prop resets model and eligibility State, drives every step, and applies one update per sequence. Keep time-major inputs and preserve the batch axis in compiled learner resets.

A readout that never feeds hidden State can be trainable without receiving a temporal trace. The expected compiler warning for that path does not mean the recurrent projection failed; inspect `report.etrace_weights` and require zero report errors.

Open `references/temporal-task-training.md` for the complete batched report-window task, cue ablation, and ordinary/checkpointed BPTT parity. Open `references/model-and-readout-selection.md` when choosing an instantaneous or leaky readout.

## Verify temporal learning

Require finite nonzero gradients for both the recurrent projection and readout, actual parameter changes, and decreasing fixed-fixture loss. Compare pp-prop with BPTT on a reduced matched model, but do not assume estimator equality. Surrogate gradients are deliberately not finite-difference derivatives of hard threshold spikes.

The Docker regression fixture separately checks the recurrent and readout updates, compiler participation, online gradients, and a matched BPTT gradient direction. Its small synthetic target validates mechanics; it does not demonstrate held-out cognitive performance or long-memory accuracy.

Open `braintrace` → `references/pp_pprop workflow.md` for estimator and memory constraints. Open `references/braintools/optimizer.md`, `references/braintools/metric.md`, or `references/braintools/surrogate.md` when choosing the corresponding infrastructure.

## Official contracts

- [BrainSpike Neuron surrogate interface](https://brainx.chaobrain.com/brainspike/apis/generated/brainspike.Neuron.html)
- [BrainTrace pp-prop](https://brainx.chaobrain.com/braintrace/tutorials/pp_prop.html)
- [BrainTrace online SNN workflow](https://brainx.chaobrain.com/braintrace/tutorials/snn_online_learning.html)
