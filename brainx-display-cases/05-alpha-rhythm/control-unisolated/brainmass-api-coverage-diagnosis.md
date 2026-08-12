# BrainX diagnosis: alpha rhythm under weakened inhibition

## Evidence studied

- Generated artifacts: `alpha_rhythm.py`, `README.md`, `alpha_rhythm.png`,
  `agent-final.md`, the complete JSONL event stream, stderr, and harness metadata.
- Independent execution with
  `/Users/nijiachen/Downloads/Brainx testing/.venv-brainx/bin/python`; the script
  exited successfully and reproduced all six reported rows and the PNG.
- Owning skills and references: `brainx-general-guard`, `brainmass`,
  `brainstate`, `brainunit`, `brainmass/references/modellibrary.md`,
  `batch-transform-acceleration.md`, `visualization-analysis-api.md`,
  `parameter-sweeps-and-regime-analysis.md`, BrainState control-flow and vmap
  references, and BrainUnit's root conversion workflow.
- Closest executable examples: `jansen-rit-eeg-proxy.py`,
  `wilson-cowan-ei-dynamics.py`, `eeg-fitting-with-gradients.py`, and the
  parameter-sweep examples.
- Authoritative APIs: `brainmass.WilsonCowanStep`,
  `brainmass.JansenRitStep`, `brainmass.Simulator`,
  `brainmass.viz.plot_power_spectrum`, `brainstate.transform.vmap`,
  `brainstate.transform.for_loop`, and `brainunit.Quantity.to_decimal`.
- Focused reference experiment: a Jansen-Rit column driven at `220 Hz` retained
  an alpha peak while reducing `Ai` from `22 mV` to `15 mV` reduced EEG-proxy
  RMS from about `1.26 mV` to `0.00009 mV` under the fixed protocol.

## Executive diagnosis

The artifact is executable, uses the requested package set, maps complete
independent rollouts, validates all initial-condition lanes, and honestly
reports that its phenomenological intervention collapses the oscillation. Its
main scientific defect is a fabricated physical scale: the official
Wilson-Cowan contract defines `rE`, `rI`, and their external inputs as
dimensionless activity, but the script multiplies them by an arbitrary
`100 Hz` constant and labels the proxy, RMS, and spectral power as physical
frequency quantities. The artifact also bypasses the package-routed spectral
metric and plotting helper and labels an unnormalised FFT sum as power in
`Hz squared`.

The skill gap is narrower than the artifact gap. BrainMass routes alpha/EEG
tasks to Jansen-Rit and E/I dynamics to Wilson-Cowan, but it does not expose the
parameter-level decision that `JansenRitStep.Ai` is the unit-aware inhibitory
gain while `WilsonCowanStep.wEI` is dimensionless I-to-E coupling. The agent
therefore guessed rejected Jansen-Rit parameter names, fell back to
Wilson-Cowan, and invented a rate scale. Existing guidance already says to use
BrainMass/BrainTools spectral APIs and to convert quantities only at explicit
raw-array boundaries, so those failures do not justify broader edits.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `alpha_rhythm.py:18-28`, `46-55`, `81`, `112-113`, `131-136` | `rE`, `rI`, and their inputs are dimensionless in the official Wilson-Cowan contract, but the script applies an arbitrary `100 Hz` scale and reports physical Hz and `Hz squared`. | The numerical regime is real, but every amplitude and power unit is fabricated and the figure visually presents a nonphysical firing-rate calibration. | Keep Wilson-Cowan activity and its proxy dimensionless, or use Jansen-Rit `Ai` and native `eeg()` when a unit-bearing EEG proxy is required. |
| P2 | `alpha_rhythm.py:60-64`, `91-101`, `133-137` | The custom FFT quantity is not a documented PSD normalization, yet the figure calls it power with physical squared units. | Absolute values cannot be interpreted or compared as a calibrated PSD. | Use `braintools.metric.power_spectral_density` for numeric analysis and `brainmass.viz.plot_power_spectrum` for display, preserving the documented sampling-step convention. |
| P2 | `README.md:22-23` | The README says the script prints alpha-power fraction, while the final code prints an absolute FFT-bin sum. | The prose misstates the delivered metric. | Name the exact metric and its units/normalization, or remove the redundant claim. |
| P3 | `alpha_rhythm.py:44-47` | Initial State is initialized and then overwritten manually although the generated model contract exposes `rE_init` and `rI_init`. | The code is more coupled to model State names than necessary. | Prefer documented initializer arguments when they express the mapped initial-condition design cleanly; direct State assignment remains valid only when the initializer cannot accept the mapped value. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Select aggregate cortical model | `WilsonCowanStep` | BrainMass model library | Valid phenomenological E/I model, but not the package's native alpha/EEG path. | Prefer `JansenRitStep` plus `eeg()` when EEG meaning is central; retain Wilson-Cowan only with dimensionless proxy language. |
| Weaken inhibition | Change `wEI` from 12 to 8 | `WilsonCowanStep.wEI` | Correct: the official API defines `wEI` as dimensionless I-to-E coupling. | Keep only the selective parameter change and label its semantics exactly. |
| Physical timing | BrainUnit `dt`, duration, transient, and time constants | BrainUnit quantities | Correct. | Retain. |
| Model inputs and activity | Divide and multiply by `RATE_SCALE = 100 Hz` | Wilson-Cowan dimensionless API boundary | Incorrect fabricated scale. | Pass documented dimensionless values directly or change to a model with unit-bearing States and inputs. |
| Initialize independent lanes | `init_all_states()` then assign `rE.value`/`rI.value` | `rE_init`, `rI_init`, BrainState lifecycle | Executable but lower-level than necessary. | Use documented initializers when possible and keep each lane independent. |
| Evolve time | `brainstate.transform.for_loop` over time and index | BrainState `for_loop` | Correct and explicitly required by the prompt. | Retain one State-driven time loop. |
| Compare parameter and initial-condition lanes | `brainstate.transform.vmap(simulate_one)` | BrainState `vmap` | Correct: the mapped function constructs independent model State and returns no mutable State. | Retain complete-rollout mapping. |
| EEG/population observable | `rE - rI` | Host-defined Wilson-Cowan proxy or `JansenRitStep.eeg()` | Legitimate only as a dimensionless local activity proxy, not calibrated EEG. | Prefer native `eeg()` for Jansen-Rit; otherwise label the host-defined proxy dimensionless. |
| Convert to analysis arrays | `to_decimal(u.Hz)` then NumPy | BrainUnit explicit boundary | API use is syntactically correct but rests on an invented input unit. | Convert only genuine quantities to a stated unit; plain dimensionless arrays need no unit conversion. |
| Spectral summary | `numpy.fft` | `braintools.metric.power_spectral_density` | Bypasses an owning API already routed by the skill. | Use the BrainTools metric and recorded sampling interval. |
| Spectral plot | `Axes.plot` of custom FFT | `brainmass.viz.plot_power_spectrum` | Works but duplicates a high-level BrainMass helper. | Use the helper unless a required overlay cannot be expressed by it. |
| Figure composition | One `plt.subplots` call and basic axes methods | High-level Matplotlib host boundary | Correct. | Retain. |
| Per-lane scientific validation | Host NumPy assertions | Host boundary | Correct and important; it prevents averaging from hiding a failed lane. | Retain and add an explicit paired alpha-power comparison when using a calibrated metric. |
| Serialization/reporting | PNG and CSV-like stdout | Host boundary | Correct. | Retain only requested outputs. |

## Missing, bypassed, or misused BrainX APIs

- `brainmass.JansenRitStep(Ai=...)` and `JansenRitStep.eeg()` should be the
  first decision for a native alpha/EEG intervention. `Ai` is inhibitory gain
  in millivolts; lowering it selectively weakens the inhibitory postsynaptic
  gain. This is not a drop-in replacement for `wEI`: it changes both the model
  family and observable semantics.
- `brainmass.WilsonCowanStep(wEI=...)` is used correctly for I-to-E coupling,
  but its `rE`, `rI`, `rE_inp`, and `rI_inp` are dimensionless. BrainUnit must
  not be used to manufacture physical units for them.
- `WilsonCowanStep(rE_init=..., rI_init=...)` is omitted. These constructor
  hooks are preferable when documented initializers can express the intended
  initial State.
- `braintools.metric.power_spectral_density(signal, dt_ms)` should replace the
  custom FFT when a numerical PSD is reported.
- `brainmass.viz.plot_power_spectrum(signal, dt, ax=..., loglog=...)` should
  replace the custom spectral plotting path when its standard PSD display is
  sufficient.

## Performance and code simplicity

The performance structure is sound: one `vmap` owns the six independent
condition/initial-State lanes and one `for_loop` owns all timesteps. No Python
timestep loop or raw JAX State transform is present. Model construction inside
the mapped function gives each lane independent State. The host list
comprehensions run only over six settled traces after device execution, so
they are not a material bottleneck.

The arbitrary rate-scale layer adds conversions, misleading labels, and
assertion complexity without scientific value. Removing it is the largest
simplicity win. Replacing the custom FFT/plot pair with routed BrainTools and
BrainMass APIs would further reduce numerical and presentation code while
making the PSD contract explicit.

## Skill improvements

- Add one compact decision section to
  `skills/brainmass/references/modellibrary.md` that distinguishes native
  Jansen-Rit alpha/EEG interventions from Wilson-Cowan E/I interventions.
- Ground the section in the generated APIs: `Ai` is Jansen-Rit inhibitory gain
  in mV and `eeg()` returns `E - I`; `wEI` is Wilson-Cowan dimensionless I-to-E
  coupling and `rE`/`rI` plus their inputs remain dimensionless.
- Route mapped multi-condition comparisons back to
  `parameter-sweeps-and-regime-analysis.md` and
  `batch-transform-acceleration.md` rather than duplicating transform detail.
- Do not edit `brainx-general-guard`, BrainState, BrainUnit, or visualization
  references: their current rules already prohibit the artifact's remaining
  failures.

## Checks for the next run

- The chosen model and inhibition parameter follow one documented semantic
  path: Jansen-Rit `Ai` plus `eeg()`, or Wilson-Cowan `wEI` plus a dimensionless
  activity proxy.
- No arbitrary scale turns dimensionless model State or inputs into Hz, mV, or
  another physical quantity.
- Every intervention and initial-condition lane runs through the same complete
  `vmap` path, and time evolves through `for_loop` as requested.
- Baseline alpha evidence is checked per lane with a defined peak and nonzero
  amplitude/power; a flat intervention trace has no claimed peak.
- Numeric spectral results use the routed BrainTools metric or state an exact,
  correct normalization; plots use the BrainMass helper when sufficient.
- The figure is opened and shows both the trace and spectral intervention
  change with accurate axis units and readable labels.
