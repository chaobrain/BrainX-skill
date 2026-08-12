# BrainX diagnosis: alpha rhythm

## Evidence studied

The task represents one aggregate cortical column, so BrainMass owns the
scientific model and observable. BrainState owns initialization, environment
time, State-aware `for_loop`, and `vmap`; BrainUnit owns physical quantities;
BrainTools owns initialization and standard spectral analysis.

Generated evidence:

- `prompt.md`, `alpha_rhythm.py`, `agent-final.md`, and the complete CLI event
  and stderr logs in this run
- `alpha_rhythm_inhibition.png`, inspected at its rendered size
- An independent execution with the required BrainX virtualenv, which
  reproduced all six rows and regenerated the figure without warnings
- `py_compile` and the run's deterministic replay reported by the generating
  agent

Skill and example evidence:

- `skills/brainx-general-guard/SKILL.md`
- `skills/brainmass/SKILL.md`
- `skills/brainmass/references/modellibrary.md`
- `skills/brainmass/references/scripts/jansen-rit-eeg-proxy.py`
- `skills/brainmass/references/simulator-input-monitor-api.md`
- `skills/brainmass/references/batch-transform-acceleration.md`
- `skills/brainmass/references/parameter-sweeps-and-regime-analysis.md`
- `skills/brainmass/references/visualization-analysis-api.md`
- The routed BrainState initialization, `for_loop`, and `vmap` guidance
- The routed BrainUnit conversion guidance

Authoritative API Reference pages:

- `brainmass.JansenRitStep`: model equations, `Ai` as inhibitory gain in mV,
  input units, six-State lifecycle, and `eeg() = E - I`
- `brainmass.WilsonCowanStep`: `wEI` as dimensionless I-to-E coupling and its
  subtractive role in the excitatory equation
- `brainmass.Simulator`: initialization, callable inputs and monitors,
  transformed execution, transient removal, batching, and returned time axis
- `braintools.metric.power_spectral_density`: one- and two-dimensional input,
  Welch estimation, `dt` in seconds or as a time `Quantity`, Hz output, and PSD
  shape
- BrainState generated references for `brainstate.transform.for_loop` and
  `brainstate.transform.vmap`

## Executive diagnosis

The generated program is executable, compact, and predominantly BrainX-native.
It selects the correct `JansenRitStep` alpha/EEG model, changes only its
inhibitory postsynaptic gain, records the model's derived `eeg()` observable,
keeps units through simulation, maps the complete independent simulation over
two gains and three initial conditions, and uses transformed State-aware time
iteration as the prompt explicitly requires. The baseline produces a stable
11 Hz alpha peak and 1.318-1.346 mV RMS across all initial conditions. The
intervention reduces RMS to 0.014-0.026 mV and alpha power to approximately
`1e-6`-`3e-6 mV^2`, so the saved matched-scale traces and spectra support
suppression of the oscillation.

The most consequential scientific error is reporting a 10 Hz "peak" for the
nearly flat intervention traces. `argmax` always chooses a spectral bin even
when in-band power is too small to support a frequency estimate. The analysis
must predefine a meaningful power or amplitude floor and report the peak as
undefined below it.

The intervention value `Ai=15.4 mV` is also outcome-calibrated without a cited
source or nearby sensitivity, and it is below the API page's documented
17.6-110 mV estimation range. It is a legitimate exploratory perturbation, but
must be labeled phenomenological rather than presented as a generally
validated weakening protocol.

The main API omission is the hand-written Hann periodogram.
`braintools.metric.power_spectral_density` owns this operation, accepts a time
`Quantity`, supports multiple channels, and returns Hz plus a one-sided Welch
PSD. The current BrainMass analysis reference obscures that API by naming its
argument `dt_ms` and passing a millisecond-valued float even though the current
contract interprets every float as seconds.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `alpha_rhythm.py:84-86`, printed rows, and `agent-final.md` | A 10 Hz peak is selected for intervention spectra whose integrated alpha power is only approximately `1e-6`-`3e-6 mV^2`. No signal-presence floor precedes `argmax`. | Numerical residue is presented as an oscillatory frequency, contradicting the visible near-flat trace and overstating what the simulation identifies. | Freeze a power or amplitude floor before comparing conditions; return and print an undefined peak below it. Keep RMS and band power as the evidence for suppression. |
| P2 | `alpha_rhythm.py:24`, module docstring, plot title, and final response | `Ai` is reduced from the documented 22 mV default to 15.4 mV without a source, calibration split, or neighboring values. The value is below the API page's documented estimation range. | The qualitative intervention is valid, but its magnitude and apparent near-complete suppression are not externally validated or shown robust to the chosen value. | Label 15.4 mV as a phenomenological perturbation. Prefer a frozen small sweep around it when making a robust regime claim; do not imply the value is physiological or canonical. |
| P3 | `alpha_rhythm.py:155-157` | The assertion checks RMS suppression but not the claimed absence of an identifiable spectral peak. | A future trace with tiny amplitude still passes while producing a misleading frequency label. | Assert the frozen signal-presence rule and preserve band-power/RMS checks separately. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Choose an aggregate alpha/EEG model | `brainmass.JansenRitStep` | `brainmass.JansenRitStep` | Correct. This is the documented six-State cortical-column model for alpha-like EEG. | Preserve. |
| Represent inhibitory weakening | Vary `Ai` from 22 to 15.4 mV | `JansenRitStep(Ai=...)` | Correct parameter family and unit; the chosen magnitude is unsourced and outside the documented estimation range. | Teach `Ai` versus Wilson-Cowan `wEI` semantics and require phenomenological labeling/sensitivity for calibrated values. |
| Represent time, potentials, rates, and drive | BrainUnit quantities | BrainUnit | Correct. Units remain attached through model construction and rollout. | Preserve. |
| Initialize three matched State conditions | `braintools.init.Constant` plus `brainstate.nn.init_all_states` | BrainTools initialization and BrainState lifecycle | Correct for independent custom rollouts. | Preserve; state that these are nuisance initial-condition checks, not stochastic trials. |
| Set integration time and current time | Nested `brainstate.environ.context` | BrainState environment | Correct for the prompt-required explicit time loop. | Preserve. |
| Advance the cortical model | `JansenRitStep.update(0 mV, 220 Hz, 0 mV)` | `JansenRitStep.update` | Correct input order and units according to the generated API contract. | Preserve. |
| Observe simulated EEG | `column.eeg()[0]` | `JansenRitStep.eeg()` | Correct. It returns the documented `E - I` postsynaptic-potential difference. | Preserve. |
| Iterate through simulation time | `brainstate.transform.for_loop` | BrainMass `Simulator` by default; BrainState `for_loop` for required custom control flow | Correct exception. The user explicitly requires `for_loop`, so replacing it with `Simulator` would fail the task contract. | Preserve the complete transformed loop; do not add a Python timestep loop. |
| Compare gains and initial conditions | Construct a model inside `simulate_one`, then `brainstate.transform.vmap` the complete rollout | BrainState `vmap`; BrainMass parameter-sweep pattern | Correct. Each mapped lane owns independent model State and returns the same shape. | Preserve. A small sensitivity grid could use the same mapped path. |
| Construct time coordinates and discard transient | BrainUnit `arange`, then a host Boolean mask | BrainMass `Simulator` normally; BrainUnit plus host analysis boundary here | Correct for the custom-loop requirement. | Preserve or slice by a precomputed step count; no new API is needed. |
| Convert EEG and time for analysis/plotting | `Quantity.to_decimal` at the host boundary | BrainUnit `to_decimal` | Correct explicit conversion. | Preserve. |
| Estimate one-sided PSD | Custom NumPy Hann window, `rfft`, normalization, and frequency construction | `braintools.metric.power_spectral_density` | Bypassed named BrainTools metric. The custom calculation is coherent but duplicates supported Welch machinery. | Use the BrainTools metric with the recorded time `Quantity`; set `nperseg` deliberately for frequency resolution. |
| Compute dominant frequency | Host mask plus `argmax` | Host scientific-analysis boundary | A host boundary is legitimate; the decision rule is incomplete. | Apply a frozen signal-presence floor before `argmax`; return undefined below it. |
| Integrate alpha power and compute RMS/mean | NumPy reductions and `trapezoid` | Host scientific-statistics boundary | Correct. No more specific BrainX API owns these simple summaries. | Preserve, with units stated after the explicit conversion. |
| Validate matched effects | Baseline alpha-band assertion and paired RMS-ratio assertion | Host scientific-validation boundary | Mostly correct and performed for every initial condition. | Add the signal-presence invariant and label the intervention calibration. |
| Plot matched traces and ensemble spectra | One `plt.subplots`, high-level Matplotlib calls | `brainmass.viz` for simple panels; Matplotlib host boundary for this custom 2-by-2 ensemble comparison | Legitimate boundary. The shaded min/max spectra and shared condition scales exceed the thin single-signal helper's canonical figure. | Preserve the composed plot. Mark undefined peaks in text rather than annotating spectral residue. |
| Save and report artifacts | `pathlib`, Matplotlib save, formatted text | Host presentation and serialization boundary | Correct. BrainX need not wrap file naming or terminal tables. | Preserve. |

## Missing, bypassed, or misused BrainX APIs

### `braintools.metric.power_spectral_density`

Use it to replace `power_spectrum()`. It performs a one-sided Hann-windowed
Welch PSD, accepts `(n_time,)` or `(n_time, n_channels)`, accepts `dt` either as
a float in seconds or as a time `Quantity`, and returns frequency in Hz. For
this deterministic two-second analysis window, set `nperseg` explicitly if the
default `n_time // 8` does not provide enough alpha-band resolution. Do not
pass a millisecond-valued float: the current API interprets every float as
seconds.

The API does not decide whether a spectrum contains an identifiable peak and
does not integrate an arbitrary scientific band. The signal-presence floor,
band mask, `argmax` decision, and alpha-power integration remain explicit host
analysis.

### `brainmass.Simulator`

`Simulator.run` normally replaces manual initialization, time iteration,
monitoring, and transient removal. It should not replace this run's explicit
loop because the prompt requires `brainstate.transform.for_loop` and `vmap`.
The generated code follows the documented lower-level exception: it constructs
each independently mapped model inside the mapped function, initializes State,
sets environment time, and transforms the entire rollout.

### `brainmass.viz.plot_power_spectrum`

This helper is suitable for one simple spectrum panel, as shown in the routed
Jansen-Rit example. It is not a direct replacement for the required comparison
of ensemble mean and min/max envelopes across two conditions. Continue to use
high-level Matplotlib after computing PSD with BrainTools.

No BrainX API should replace the RMS, mean, band integral, threshold decision,
formatted report, `Path`, or custom multi-panel presentation logic.

## Performance and code simplicity

The program has one compiled State-aware time loop per mapped lane and one
stateful `vmap` over six independent runs. It contains no Python timestep loop,
no raw JAX transform over State, and no full-trajectory duplication beyond the
six requested traces. Model construction inside the mapped function is the
correct parameter-batching pattern. The small Python loop over six completed
host trajectories is not a simulation bottleneck, but the spectral calculation
can be both shorter and more BrainX-native by passing a time-major multi-channel
array to `power_spectral_density` once.

`Simulator` would be the normal shorter rollout, but using it here would remove
an explicitly requested API and reduce requirement coverage. The current
custom loop is therefore justified, not accidental infrastructure duplication.

The figure uses exactly one `plt.subplots` call and basic plotting operations.
Its matched trace scales and matched log-power scales make the intervention
effect visible. The two-by-two layout, repeated label setup, and min/max band
are proportionate to the scientific comparison; no custom visualization class
or HTML is warranted.

## Skill improvements

Make only these BrainMass reference edits:

1. Add a compact inhibition-intervention decision in `modellibrary.md`:
   identify Jansen-Rit `Ai` as inhibitory gain in mV, identify Wilson-Cowan
   `wEI` as dimensionless I-to-E coupling, state that lowering each weakens its
   specific inhibitory mechanism, and require unsourced or outside-range
   outcome-tuned values to be labeled phenomenological with nearby sensitivity.
2. Correct `visualization-analysis-api.md` to the current
   `power_spectral_density(signal, dt, ...)` contract: a float is seconds, a
   time `Quantity` is converted to seconds, one- and two-dimensional signals
   are accepted, and returned frequency is Hz. Update its example to pass the
   recorded `Quantity` time difference directly.
3. Add one spectral-identifiability rule beside that workflow: predefine an
   amplitude or power floor before peak search and report the peak as undefined
   below it because `argmax` otherwise always returns a bin.

Do not edit `brainx-general-guard`: it already requires observable-based claim
validation, frozen thresholds, phenomenological labeling, high-level BrainX
analysis, and custom host statistics only at verified boundaries. Do not grow
the BrainMass root skill: it already routes model choice to `modellibrary.md`
and spectral analysis to `visualization-analysis-api.md`.

## Checks for the next run

- The script runs with the required BrainX virtualenv and produces a readable
  alpha comparison figure without warnings or non-finite values.
- The resting condition shows a visible, nontrivial 8-13 Hz oscillation for
  every matched initial condition; frequency, RMS, and alpha power are retained
  per condition.
- A dominant frequency is reported only when a predefined amplitude or power
  floor is met. A near-flat intervention is reported as having no identifiable
  peak rather than an arbitrary FFT/Welch bin.
- The weakening mechanism is model-specific and explicit: Jansen-Rit uses
  `Ai` in mV or Wilson-Cowan uses dimensionless `wEI`; the two are not
  conflated.
- Any unsourced, outcome-tuned, or outside-documented-range intervention value
  is labeled phenomenological and preferably accompanied by nearby sensitivity.
- Standard PSD computation uses
  `braintools.metric.power_spectral_density` with a time `Quantity` or a float
  in seconds, and the selected `nperseg` resolves the claimed band.
- The complete independent simulation remains inside
  `brainstate.transform.vmap`, and time remains inside
  `brainstate.transform.for_loop` as the prompt requires.
- Units remain attached through simulation and cross into NumPy/Matplotlib only
  at an explicit analysis or presentation boundary.
- Matched nuisance initial conditions are preserved, the baseline and
  intervention are independently validated, and the saved figure uses aligned
  time and spectral scales.
