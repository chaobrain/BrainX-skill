# NeuroSpecification

- Status: locked
- Researcher approval: The 2026-08-24 request specifies the six channels, source data, current-to-trace mapping, and one-trace/three-trace split. Trace #9 is fixed as training data; Traces #6-#8 are held out.

## Researcher request
- Brain-modeling question or behavior: Can a six-current Hodgkin-Huxley-type C. elegans body-wall muscle model fitted to one current-clamp recording predict responses to other current amplitudes?
- Requested model, experiment, or comparison: Model SHK-1, EGL-19, SLO-2, Kr, Na, and leak currents; fit parameters from one of Traces #6-#9; compare predictions with the other three traces.
- Execution mode: parameter-fitting
- Required outputs: BrainX model code, fitted parameter estimates and uncertainty samples, held-out voltage predictions, protocol-level and waveform metrics, and a conclusion about agreement with experiment.
- Constraints: Use `/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/16-C.elegans-muscle-cell/Fig4A-D.txt`; keep raw data read-only; use only Trace #9 (30 pA) during fitting and reserve Traces #6 (15 pA), #7 (20 pA), and #8 (25 pA) for final testing.

## Inspected data contract
- Data sources and inspected contents: Axon Text File `Fig4A-D.txt`; columns are time plus ten voltage traces. Only Traces #6-#9 are in scope. The authors' publication and source repository define the corresponding 15, 20, 25, and 30 pA protocols.
- Shapes, axes, sampling/time base, and physical units: 10,000 raw samples over 0.5 s at 0.05 ms/sample; voltage in mV. Deterministic stride-2 downsampling produces 5,000 samples at 0.1 ms. The current step is 57.8-257.8 ms and the model input is total current in pA.
- Required preprocessing and the subset used to fit each transform: Downsample all four traces identically without filtering. Compute fitting summaries from Trace #9 only. Do not inspect Traces #6-#8 for parameter selection, proposal adaptation, or candidate selection.
- Mapping from data to model inputs, targets, and observables: Trace #9 receives a 30 pA step and supplies the fitting summaries. Held-out traces receive 15, 20, and 25 pA. Observables include stimulus spike count, mean interspike interval, first-spike latency, resting mean voltage, stimulus mean/std/peak voltage, post-stimulus spike count, whole-trace RMSE, and correlation.
- Known data limitations or unresolved mismatches: One recording per current gives no trial variability. Trace-to-trace differences may include cell variability, not only current dependence. The requested six-current model omits the small M-type current present in one author implementation. The paper equation includes a voltage factor for SLO-2 while one fitting helper omits it; this case follows the paper equation. This contract was reconstructed while resuming an existing implementation/run and is not a prospective preregistration.

## Acceptance boundary
- Evidence required for success, failure, or an inconclusive result: Report every held-out metric. Qualitative current-response consistency requires correct held-out stimulus spike counts and the experimental direction of firing-rate/ISI change with current. Waveform consistency requires low residuals and aligned spike timing; it must be assessed separately and may fail even when protocol-level behavior passes.
- Required baselines and controls: No-current baseline must remain finite and quiet before stimulation; all six named currents must be present; candidate batches must be independent; nominal and boundary candidates must be finite or explicitly rejected; the selected solver must agree with a smaller-step reference closely enough to preserve protocol spike counts and voltage summaries; synthetic recovery must be reported before interpreting fitted values mechanistically.
- Invalid-result conditions: Fit/test leakage, missing or non-finite predictions, incorrect units or trace-current mapping, runtime State carried across independent candidates, unreported boundary solutions, or changing summaries/bounds after selecting a favorable observed-data result.
- Allowed claims and explicit non-claims: Allowed claims concern predictive behavior under the four tested current-clamp protocols. Parameter values are calibration estimates with uncertainty, not uniquely identified biological conductances unless exact-pipeline recovery supports that parameter. ABC kernel weights are not an exact posterior density, and the lowest-discrepancy sample is not a mathematical MAP estimate. Agreement in spike count alone is not full waveform agreement.
