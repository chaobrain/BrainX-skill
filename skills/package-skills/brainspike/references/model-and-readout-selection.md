# Select point-neuron components and readouts

Choose the scientific dynamics and observable before the execution wrapper. A common lifecycle or similarly named class does not establish identical equations, reset timing, refractory behavior, or units.

## Choose the represented dynamics

| Component | Use |
|---|---|
| Custom `LIF` in `scripts/native_workflows.py` | Use as an explicit Euler/surrogate mechanics fixture; it has soft reset and no absolute refractory or adaptation State. |
| NEST `iaf_psc_delta`, `iaf_psc_alpha`, `iaf_psc_exp` | Choose the requested current response and refractory contract; preserve the model's own buffering and reset semantics. |
| NEST `iaf_cond_*` | Use when reversal-dependent conductance is part of the model; select receptor ports and positive conductance magnitudes deliberately. |
| NEST `aeif_*`, `gif_*`, `glif_*` | Use the documented adaptive or generalized family when its additional State is scientifically required. |
| BrainCell models | Use when ions, channels, compartments, or morphology are explicitly modeled or modified; selecting a fixed point-HH NEST model alone does not require rewriting it as a detailed cell. |

Open `references/nest-compatible/model-library.md` for exact public models and `references/nest-compatible/integration-categories.md` for numerical behavior. Verify startup voltage, current/voltage/conductance units, firing response, adaptation, and stability at the intended timestep before accepting a replacement.

## Choose synaptic meaning before storage

Use a current-valued filter for CUBA, a nonnegative conductance with `g*(E-V)` for COBA, and a documented voltage-jump contract for delta input. Keep excitatory/inhibitory reversal channels separate. Open `references/native-projections.md` for the complete native E/I path and the conditions under which shared linear filters are exact.

Read a model's spike-access contract before recurrent wiring: a method may evaluate spikes from current voltage rather than return cached events from the last update. Preserve the appropriate update output or documented cache. The bundled custom LIF explicitly stores its completed spike array; do not generalize that implementation to every NEST model.

## Choose the temporal readout

| API or operation | Use |
|---|---|
| `braintrace.nn.Linear` | Use an instantaneous trainable projection when each step's output is the task observable. |
| `braintrace.nn.LeakyRateReadout(..., tau=...)` | Use when a trainable temporally filtered readout is part of the model; reset its hidden State with independent trials. |
| Masked window mean or sum | Use when the declared report window determines the statistic; retain time and trial axes until the final reduction. |
| `SimulationResult.trace` | Use a recorded continuous NEST observable for the verified direct fitting path. |

The minimal native training fixture uses an instantaneous readout. The cue-delay-report fixture uses a leaky readout plus explicit report-window reduction. Run `scripts/delayed_cue_training.py` for that canonical dynamic-readout example, then open `references/temporal-task-training.md` for the scoring and validation contract.

A leaky readout adds memory, a window sum changes with window duration, and a window mean changes the weighting of variable-length reports. Select deliberately. Hard recorded spike counts have zero gradients in the tested Simulator configuration; open `references/simulator-training.md` before using them as training targets.

For readout-layer variants, open `braintrace` → `references/pre-built-braintrace-layer.md`. Use BrainTrace's learner only when its temporal estimator is needed; a BrainTrace layer can also participate in ordinary BPTT, as the native examples demonstrate.

## Sources

- [BrainSpike model directory](https://brainx.chaobrain.com/brainspike/nest-style/models.html)
- [BrainTrace readout layers](https://brainx.chaobrain.com/braintrace/apis/nn.html)
