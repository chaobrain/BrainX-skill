# BrainX diagnosis: alpha rhythm after refinement

## Evidence studied

Generated evidence:

- `prompt.md`, `alpha_rhythm.py`, `agent-final.md`, the complete event log,
  stderr, and harness metadata for this run
- `alpha_rhythm_comparison.png`, inspected at its native 1980 by 1296 pixels
- An independent execution with the required BrainX virtualenv, which
  reproduced all six rows and regenerated the figure without warnings
- An independent `py_compile` check
- Run 0's diagnosis checks and generated scientific outputs for comparison

BrainX review standard:

- `skills/brainx-general-guard/SKILL.md`
- `skills/brainmass/SKILL.md`
- `skills/brainmass/references/modellibrary.md`
- `skills/brainmass/references/scripts/jansen-rit-eeg-proxy.py`
- `skills/brainmass/references/batch-transform-acceleration.md`
- `skills/brainmass/references/parameter-sweeps-and-regime-analysis.md`
- `skills/brainmass/references/visualization-analysis-api.md`
- The routed BrainState initialization, environment, `for_loop`, and `vmap`
  guidance
- The routed BrainUnit array and conversion guidance
- Official API Reference pages for `brainmass.JansenRitStep`,
  `brainmass.Simulator`, `braintools.metric.power_spectral_density`,
  `brainstate.transform.for_loop`, and `brainstate.transform.vmap`

The run used the same 577-byte prompt with SHA-256
`c6b42ffbed8f7f18ed64b1184e8c68ffb9491e63d8967f0234264075964a3f41`,
`gpt-5.6-sol`, `xhigh` reasoning, Codex CLI `0.147.0-alpha.1.2`, the same
virtualenv, and macOS Seatbelt host-read isolation as Run 0.

## Executive diagnosis

Run 1 resolves every material Run 0 finding. It retains the correct
Jansen-Rit model, `eeg() = E - I` observable, unit-aware dynamics, complete
stateful `vmap`, and prompt-required `for_loop`. It now uses the documented
20% reduction from 22.0 to 17.6 mV, the lower endpoint of the API page's
reported estimation range, instead of an unexplained 15.4 mV value. It replaces
the custom FFT periodogram with the BrainTools Welch PSD, passes the recorded
sampling interval as a BrainUnit `Quantity`, and explicitly declares a 0.001 mV
RMS floor before assigning a dominant frequency.

The scientific outcome is stable across all three initial excitatory
potentials. Baseline runs produce an 11.0 Hz peak, approximately 1.249 mV RMS,
and 1.553 mV squared alpha power. Every weakened-inhibition run falls below the
reporting floor at 0.000176-0.000180 mV RMS and approximately
`1.65e-9`-`1.87e-9 mV^2` alpha power, and is correctly printed as `n/a` rather
than assigned a residual spectral bin. The figure shows the trace, spectrum,
per-initial-condition alpha power, and per-initial-condition RMS on readable
matched comparisons.

The residual issues are low severity. The 0.001 mV floor is reproducible and
lies far between the two observed regimes, but no external noise floor or
pre-analysis calibration justifies that exact threshold. The event log also
shows exploratory runs before the final threshold was written, so
"predeclared" means declared in code before `analyze()` rather than frozen
before any intervention outcome was observed. The unused
`relative_alpha_power` calculation is minor avoidable analysis. Existing guard
and BrainMass analysis guidance already prohibit post-outcome thresholding and
unneeded calculations, so neither finding justifies another skill edit.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P3 | `alpha_rhythm.py:35`, `alpha_rhythm.py:113-119`, and the event log | The 0.001 mV RMS floor has no empirical, numerical-noise, or held-out calibration source, and exploratory condition results preceded its final selection. | The frequency suppression conclusion is not sensitive to this exact value because the conditions differ by roughly four orders of magnitude, but the word "predeclared" overstates the experimental freezing procedure. | Describe the floor as a fixed reporting threshold for this demonstration. For an empirical claim, calibrate it from measurement noise or a separate run before viewing intervention outcomes. |
| P3 | `alpha_rhythm.py:111`, returned metrics | Relative alpha power is computed but never reported, plotted, asserted, or used in a decision. | It adds one undefined-risk division and one concept without contributing evidence. | Remove it unless a relative-band-power claim is required. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Select the cortical alpha model | `brainmass.JansenRitStep` | `JansenRitStep` | Correct and unchanged from Run 0. | Preserve. |
| Define inhibitory weakening | `Ai` from 22.0 to 17.6 mV | `JansenRitStep(Ai=...)` | Correct model-specific parameter and unit. The magnitude is an explicit 20% perturbation at the documented lower estimation endpoint. | Preserve; avoid generalizing one perturbation into a full monotonic regime claim. |
| Define synaptic timing | Unit-bearing time constants inverted into `be` and `bi` | BrainUnit plus `JansenRitStep` parameters | Correct and physically legible. Values reproduce model defaults. | Preserve or omit the explicit defaults if further simplicity is needed. |
| Define nuisance initial conditions | Three `E_init` values through `braintools.init.Constant` | BrainTools initialization | Correct matched sensitivity check. | Preserve. |
| Construct parameter coordinates | Unit-aware `u.math.meshgrid(..., indexing="ij")`, flatten, map, reshape | BrainUnit structural operations plus BrainState `vmap` | Correct and clearer than Run 0's manual repeat/tile grid. | Preserve. |
| Initialize independent State | Construct the model inside the mapped function and call `init_all_states()` | BrainState lifecycle | Correct. Each mapped call owns six-State Jansen-Rit dynamics. | Preserve. |
| Advance time | `brainstate.transform.for_loop` over unit-aware times and indices | BrainState `for_loop`; BrainMass `Simulator` normally | Correct prompt-required lower-level path. No Python timestep loop is present. | Preserve. |
| Set time context | `brainstate.environ.context(dt=...)` and per-step `t`/`i` | BrainState environment | Correct for explicit transformed execution. | Preserve. |
| Drive the model | `column.update(E_inp=220 * u.Hz)` | `JansenRitStep.update` | Correct named input and unit. | Preserve. |
| Observe EEG | `column.eeg()` | `JansenRitStep.eeg()` | Correct derived `E - I` observable. | Preserve. |
| Remove transient and downsample | Unit-aware trajectory slicing after rollout | Host array boundary; `Simulator` normally owns these controls | Correct for the prompt-required custom loop. `RECORDED_DT` tracks the changed sample interval. | Preserve. |
| Cross into analysis | `eeg.to_decimal(u.mV)` | BrainUnit conversion | Correct explicit boundary. Units are not stripped during dynamics. | Preserve. |
| Estimate spectra | One multichannel call to `braintools.metric.power_spectral_density(..., RECORDED_DT, nperseg=2000)` | BrainTools metric | Correct. This resolves Run 0's custom FFT and stale-`dt` problems. It yields 0.5 Hz resolution. | Preserve. |
| Estimate alpha power and RMS | NumPy reductions and trapezoidal integration | Host scientific-analysis boundary | Correct; no more specific BrainX API owns these summaries. | Preserve. |
| Gate dominant-frequency reporting | Fixed RMS floor, then host `argmax` | Host scientific decision boundary | Correct structure and resolves Run 0's false 10 Hz intervention peak. Exact threshold provenance is weak. | Calibrate externally for empirical use. |
| Validate outputs | Finite/unit assertions and alpha-band checks | Host scientific-validation boundary | Correct baseline checks; printed intervention evidence is clear. | An explicit assertion that all intervention runs are below the frozen floor would make the intended comparison executable. |
| Plot scientific evidence | One `plt.subplots(2, 2)` figure with high-level Matplotlib | Matplotlib host presentation boundary | Correct. The ensemble summaries and matched conditions justify custom composition beyond a one-signal `brainmass.viz` helper. | Preserve; close the figure after saving in reusable code. |
| Save and report | `Path`, PNG, formatted stdout | Host boundary | Correct. | Preserve. |

## Missing, bypassed, or misused BrainX APIs

No material BrainX API is missing, bypassed, or misused.

`brainmass.Simulator` remains intentionally unused because the exact prompt
requires the simulation to evolve through `brainstate.transform.for_loop` and
requires `vmap` across inhibition strengths and initial conditions. The custom
path initializes State, sets environment time, maps complete independent
simulations, and contains no Python timestep loop.

`braintools.metric.power_spectral_density` now owns the complete standard PSD
calculation. Passing `RECORDED_DT = 1 * u.ms` follows the current contract and
returns frequencies in Hz. The host mask, band integral, RMS threshold, and
frequency decision remain legitimate scientific-analysis boundaries.

`brainmass.viz.plot_power_spectrum` would remove the ensemble spectrum and the
coordinated four-panel comparison. It is not a semantically complete
replacement for this output.

## Performance and code simplicity

The simulation uses one complete `vmap` over six independent full rollouts and
one transformed 60,000-step time loop per lane. Downsampling occurs before the
host analysis boundary, and all six channels enter one multichannel Welch call.
This is BrainX-native and avoids both Python simulation loops and six repeated
spectral calls.

Compared with Run 0, the simulation is longer, from 2.5 to 6 seconds, and the
stored analysis data is downsampled from 0.1 to 1 ms after the transient. The
longer settled window produces a stable comparison while downsampling limits
host memory. The 1 ms recorded interval preserves far more bandwidth than the
40 Hz figure needs.

The program is larger than Run 0 because it adds explicit grids, a fixed
identifiability rule, a four-panel evidence figure, and multichannel analysis.
Those additions directly address the diagnosis. The unused relative-alpha
calculation, duplicate `braintools.init` import, and returned unit-bearing
trajectory used only for one assertion are minor simplification opportunities,
not reasons for another refinement checkpoint.

## Skill improvements

No further skill edit is justified by Run 1.

- `skills/brainmass/references/modellibrary.md` now supplied the correct `Ai`
  semantics and documented-range context used by the agent.
- `skills/brainmass/references/visualization-analysis-api.md` supplied the
  correct unit-aware PSD contract and the peak-identifiability rule used by the
  agent.
- `skills/brainx-general-guard/SKILL.md` already requires thresholds to be
  frozen before outcomes, calibrated values to be marked phenomenological,
  matched controls to be retained, and unnecessary logic to be removed.

Do not duplicate these rules or expand the root BrainMass skill. The residual
threshold-provenance and unused-calculation findings are implementation-level
departures from existing guidance.

## Checks for the next run

No Run 2 is needed. If this case is evaluated again after an unrelated future
change, retain these regression checks:

- Use the exact 577-byte prompt and frozen execution conditions.
- Produce an alpha-band baseline for every matched initial condition.
- Change only the declared model-specific inhibitory parameter.
- Use a BrainTools spectral metric with a time `Quantity` or seconds-valued
  float and enough frequency resolution for the claimed band.
- Report no dominant frequency below a threshold frozen independently of the
  compared outcomes.
- Preserve per-condition RMS and band power rather than reporting only an
  aggregate.
- Keep complete independent simulations in `vmap` and time evolution in
  `for_loop`.
- Preserve units through simulation and cross into host arrays explicitly.
- Save a readable figure with aligned condition comparisons.

## Comparison with Run 0

| Check | Run 0 | Run 1 | Result |
|---|---|---|---|
| Model and observable | `JansenRitStep`, native `eeg()` | Same | Preserved. |
| Stateful execution | Complete `vmap` plus `for_loop` | Same, with a unit-aware coordinate grid | Preserved and clearer. |
| Inhibitory intervention | 22.0 to 15.4 mV, unsourced and below the documented estimation range | 22.0 to 17.6 mV, explicit 20% reduction at the documented lower endpoint | Improved. |
| Spectral API | Hand-written NumPy Hann periodogram | Multichannel BrainTools Welch PSD | Improved. |
| Sampling-interval contract | Manually derived sample rate | Recorded `1 ms` BrainUnit `Quantity` passed directly | Improved. |
| Near-flat peak | Misleading 10 Hz printed for all intervention runs | `n/a` below a fixed 0.001 mV RMS floor | Resolved. |
| Baseline evidence | 11 Hz, 1.318-1.346 mV RMS, 1.645-1.672 mV squared alpha power | 11 Hz, approximately 1.249 mV RMS, 1.553 mV squared alpha power | Preserved; differences follow the longer transient/window and initialization definition. |
| Intervention evidence | 0.014-0.026 mV RMS and approximately `1e-6`-`3e-6 mV^2` alpha power | 0.000176-0.000180 mV RMS and approximately `1.65e-9`-`1.87e-9 mV^2` alpha power | Stronger settled suppression at a less extreme gain; no residual frequency claim. |
| Initial-condition sensitivity | Three matched offsets distributed across `M`, `E`, and `I` | Three matched `E_init` values | Both settle consistently; Run 1 makes the varied State explicit. |
| Figure | Matched trace and spectrum rows | Trace, spectrum, per-run alpha power, and per-run RMS | Improved evidence without reducing readability. |

Run 1 is good enough to close the alpha-rhythm refinement.
