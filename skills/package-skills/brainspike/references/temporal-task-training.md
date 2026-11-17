# Train a masked temporal report

A temporal task advances hidden State through cue, silent delay, and report phases; a report mask removes loss outside the scoring window without freezing the model. A dynamic readout is part of the model's retained memory, not merely a final plotting operation.

## Keep trials and time distinct

| API | Use |
|---|---|
| `DelayedCueSNN` in `scripts/delayed_cue_training.py` | Compose an ETP-aware input projection, native BrainSpike LIF population, and leaky trainable readout. |
| `braintrace.nn.LeakyRateReadout(..., tau=...)` | Use a readout with explicit temporal State when accumulated activity is part of the task. |
| `brainstate.nn.reset_all_states(model, batch_size=B)` | Reset independent trial State while preserving parameters; the example trains with four lanes and evaluates with two. |
| `brainstate.transform.for_loop(model, inputs)` | Advance every time step of `(time,batch,feature)` inputs. |
| `brainstate.transform.checkpointed_for_loop(model, inputs, base=4)` | Trade recomputation for activation memory inside the differentiated native loss; verify against the ordinary loop first. |

Run the complete canonical task from this skill root:

```bash
python scripts/delayed_cue_training.py
```

The fixture supplies cue amplitude for 5 steps, then zero external input for 15 steps at `dt=1 ms`. Only the final 5 steps contribute loss. Average per-output error first, multiply by the `(time,)` report mask broadcast over trials, and divide by the number of scored time-trial pairs. Do not average masked zeros over all time unless that scaling is intentional. Validate that the mask has positive support before using another protocol.

Initialize once, reset runtime State inside every loss, differentiate only `ParamState`, register optimizer State once, and apply one update after the sequence. The checkpointed loop belongs inside that differentiated loss. It does not eliminate membrane or readout State, nor does it turn BPTT into an online estimator.

## Verify the temporal claim

The Docker regression checks exact silent input, mask invariance outside the report window, finite nonzero input-projection/readout gradients, and ordinary/checkpointed loss and gradient parity on the same graph. Training lowers loss from approximately `0.424` to `0.0201`. Two fixed held-out cue amplitudes classify correctly; cue ablation reduces accuracy to 50 percent.

These tiny deterministic fixtures validate task mechanics and retained cue information in the complete graph. They do not prove population-only memory, scientific generalization, robustness over many trials, or a memory/performance benchmark. The leaky readout itself can retain the cue. Freeze evaluation trials and add matched state interventions before making a mechanism-specific claim.

Open `references/model-and-readout-selection.md` for instantaneous, dynamic, and windowed readouts. Open `brainstate` → `references/brainstate/brainstate-control-flow-patterns.md` for other loop/carry/checkpoint choices. Open `references/native-training.md` for the separately verified pp-prop workflow; this checkpointed task uses BPTT and does not validate an arbitrary pp-prop task integration.

## Sources

- [BrainTrace readout layers](https://brainx.chaobrain.com/braintrace/apis/nn.html)
- [BrainState control flow](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html)
