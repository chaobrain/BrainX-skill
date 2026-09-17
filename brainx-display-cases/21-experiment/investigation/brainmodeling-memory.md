# Brain modeling memory

## Checkpoint
- Iteration: 1
- Step: 0

### Artifacts
- `NeuroSpecification.md`: locked comparison implementing the user's explicit authorization.

### Important milestones
- Entry case: fresh-new. Existing `../prompt.md` is preserved.
- Literature gate triggered by competing mechanisms, unknown assay mapping and recent evidence.
- Boggess et al. (2026), DOI 10.1038/s41467-026-75882-0, reports the matching phenotype. Its assay is calcium integration, not a spike count. The closest culture is glutamatergic, so inhibition is a conditional alternative.
- No additional scientific approval is required for the already-requested comparison. Unmeasured parameters remain explicit assumptions.

## Literature evidence: KCNT2 sign reversal - 2026-09-07
- Research question: Which mechanisms can explain mosaic versus population-wide KCNT2 effects?
- Review artifact: `literature-review.md` (to be completed).
- Essential papers: Boggess et al. 2026, DOI 10.1038/s41467-026-75882-0 (published full text); Tomasello et al. 2017, PMID 28943756, PMC5602212 (published full text); Engel et al. 2024, PMID 39744124, PMC11688182 (published full text); Wu et al. 2024, PMID 38457342, PMC11013952 (accepted manuscript); Mao et al. 2020, PMID 32038177, PMC6992647 (published full text).
- Modeling consequences: Include Na-channel availability and sustained depolarization; preserve calcium/spike distinction; model explicit recurrent excitatory communication; test synaptic weakening independently; do not infer KCNT2 knockdown from KCNT1 gain-of-function or dominant-negative variants.
- Limitations: The exact network-collapse explanation remains a hypothesis in the matching study; synaptic effects of KCNT2 were not directly established there. No calibrated channel kinetics or network parameters are supplied.

## Checkpoint
- Iteration: 1
- Step: 1

### Artifacts
- `brainx-study-record.md`: selected scales, APIs, lifecycle and verification plan.

### Important milestones
- BrainCell owns conductance-based membrane/channel dynamics; BrainPy-State owns explicit spike-driven recurrent synapses. No aggregate population dynamics or training are required.
- Read guard, modeling-loop, bio-neuro-lit and tool contracts, BrainCell, BrainPy-State, BrainState, BrainUnit; channel/ion/custom channel, solver, input, projection and adaptation references; HH basics, ablation and conductance-network scripts.
- Existing `/home/yixinliu/anaconda3/envs/braincell-released/bin/python` contains BrainX and required components. No installation or environment mutation is planned.

## Checkpoint
- Iteration: 1
- Step: 2

### Artifacts
- `models.py`, `experiment.py`, `test_models.py`, `rescue.py`: conductance cells, recurrent projections, immutable runner, checks and interventions.
- `model-methods.md`: equations, assumptions, optical mapping and limits.
- `validation.json`: all focused checks passed.
- `literature-review.md`: source-level evidence, reading depth and competing mechanisms.

### Important milestones
- No training/fitting coverage. Added BrainEvent skill and communication example to the completed study; pure array vmap is restricted to event matrix communication, not mutable cell State.
- An implementation smoke exposed a synaptic conductance unit mismatch; fixed Expon's initializer to conductance density. Failed launch remains in `runs/network_exploration` and is excluded.
- Exploratory conventional HH/gKNa settings did not uniformly generate reversal or earlier stable block. Reduced delayed-rectifier reserve and faster local sodium sensing expose the relevant cellular regime; this selection is documented, not treated as fitted evidence.
- Spike-amplitude failure can precede stable block. Retain both events and voltage/gates; do not rename all event loss as block.

## Checkpoint
- Iteration: 1
- Step: 3

### Artifacts
- `validation.json`: cold/warm timing, exact repeat/reset and uncoupled parity, dt refinement.
- `models.py`: BrainState compiled rollout, native condition/neuron dimensions, event-driven projections.

### Important milestones
- Acceleration decision: unchanged compiled implementation. No approximation or solver shortcut introduced.
- Cell dt halving changed firing rates by 0 Hz, late mean voltage by at most 0.0041 mV and no block flags in the decisive low/high-drive comparison.
- Test output and State are bit-identical on repeated reset runs. Proceed to immutable seed/drive validation; simulation seed spread is not biological confidence.

## Checkpoint
- Iteration: 1
- Step: 4

### Artifacts
- `configs/` and `runs/`: completed fixed experiments, raw trajectories, metrics, source snapshots, configurations and logs. Failed initial network launch is explicitly labeled.
- `artifact-manifest.json`, `environment-provenance.json`: result identity and existing environment.
- `assessment.json`, `fraction-summary.csv`, `numerical-assessment.json`: deterministic optical sensitivity, fraction summaries and convergence evidence.
- `report.md`: scoped scientific assessment and claim-evidence matrix.
- `follow-up-experiments.md`: new predictions and discriminating protocols.
- `acceleration-audit.md`, `brain-tools-api-gap.md`, `metric-boundary-validation.json`: execution and external-boundary evidence.

### Important milestones
- Literature review is complete, superseding the earlier pending artifact label.
- Three independent seeds at 11 targeting fractions completed for strong/low-drive channel loss and two transmission-loss regimes; neighboring high drives and matched rescue/uncoupled/residual/dt controls are retained.
- Optical sign reversal occurs in the high-drive block model over a broad unfitted mixture interval; voltage-only calcium does not reproduce the population decrease. Low drive does not reverse.
- A strong transmission-loss model generates non-monotonic electrical output without block; restoring 10% transmission removes its below-control endpoint in the tested seed. Moderate transmission loss produces an optical, not electrical, reversal.
- Two solvers agree on decisive cell results. Network dt halving preserves optical signs, with <=1 Hz compared rate difference and <=0.036 block-fraction difference.
- Raw NPZ data are retained locally but ignored by Git to prevent accidental multi-GB commits. All result paths and hashes remain in the manifest.
- Proceed to required independent MCP review; no biological mechanism or critical fraction is identified.

## Checkpoint
- Iteration: 1
- Step: 5

### Artifacts
- `reviews/iteration-1.md`: verbatim independent report, saved before classification.

### Important milestones
- Reviewer threadId: `01a07a05-5d54-7ce0-9c06-580b962dee70`.
- Outcome: REFUSE; scientific outcome: SUPPORTED. The reviewer verified 44 assessment conditions, 560 summary rows and 176 manifest hashes.
- BX-001 (minor): replace time-predicate current construction in the rescue with precomputed BrainTools Constant protocols. Keep the channel State intervention and verify waveform/trajectory parity.
- Fresh reviewer execution was blocked by its Matplotlib cache permissions; offline artifact validation succeeded. Supply explicit writable cache settings on the next review.
- Increment iteration to 2 and return to step 1 before correction.

## Checkpoint
- Iteration: 2
- Step: 1

### Artifacts
- `brainx-study-record-iteration-2.md`: BX-001 ownership, restudied instructions and correction design.

### Important milestones
- Reopened the guard first, then BrainCell, BrainState, BrainUnit, the complete input-current reference, the HH canonical script and BrainUnit array mechanics.
- Restudy complete: precompute unit-bearing currents with Constant under dt; join condition protocols with unit-preserving stack; pass time and current to the existing BrainState loop. Preserve membrane equations, channel intervention, parameter values and timing.
- No fitting/training or network changes. Run a new immutable rescue and compare it with the retained original.

## Checkpoint
- Iteration: 2
- Step: 2

### Artifacts
- `rescue.py`: native Constant current construction under dt; time-major currents passed to for_loop.
- `test_rescue_parity.py`, `rescue-parity.json`: exact waveform, full recorded-State and summary parity.
- `brain-tools-api-gap.md`: removes the invalid stimulus-gap claim.

### Important milestones
- BX-001 corrected without changing model equations, parameters, duration or channel State intervention. Current inputs are now saved explicitly.
- Native-input rollout and focused parity check completed successfully. Original rescue artifacts remain untouched.

## Checkpoint
- Iteration: 2
- Step: 3

### Artifacts
- `rescue-parity.json`: bit-identical t, V, spikes, h, Na and summaries; exact 32000-by-3 input waveform match.
- `acceleration-audit.md`: unchanged compiled execution decision from iteration 1.

### Important milestones
- No new approximation or acceleration claim. Move current construction outside the timestep path while preserving the same BrainState loop and solver; exact parity permits reuse of all unchanged scientific evidence.

## Checkpoint
- Iteration: 2
- Step: 4

### Artifacts
- `runs/acute_rescue_native/raw.npz`, `summary.json`, `rescue.py`, `models.py`: corrected immutable run and source snapshots.
- `artifact-manifest-iteration-2.json`: original manifest preserved, new rescue artifacts appended.
- `report.md`: canonical rescue pointers updated; scientific values unchanged.

### Important milestones
- Completed native-input rescue with the same existing CPU environment and explicit writable cache settings.
- No network reruns required because network/model sources, configurations and results are unchanged; previous independent scientific verification remains applicable.
- Ready for a fresh iteration-2 MCP review of the corrected input boundary and preserved conclusions.

## Checkpoint
- Iteration: 2
- Step: 5

### Artifacts
- `reviews/iteration-2.md`: verbatim independent report, saved before classification.

### Important milestones
- Reviewer threadId: `01a07a17-067a-7c01-b872-306a90a06501`.
- Outcome: PASS; scientific outcome: SUPPORTED; no findings. BX-001 resolved.
- Independent offline review verified all 180 manifest hashes, all 44 assessment conditions, all 560 summary rows, exact rescue waveform/trajectory/summary parity and unchanged model/network sources.
- Accepted scope: bounded numerical mechanisms, controls, assumptions and predictions. Biological parameter relevance and literature synthesis are not independently established by this numerical review. Fresh reviewer execution remained blocked by its cache permissions, while the main-agent executions and offline verification succeeded.

## Checkpoint
- Iteration: 2
- Step: 6

### Artifacts
- `report.md`, `fraction-summary.csv`, `assessment.json`: accepted numerical report and full fraction/observation summaries.
- `runs/`, `artifact-manifest-iteration-2.json`: preserved raw trajectories and identity.
- `follow-up-experiments.md`: experimentally discriminating predictions and protocol.

### Important milestones
- Visualization stage blocked: `/home/yixinliu/.agents/skills/brainx-modeling-loop/references/visualization-workflow.md` does not exist (verified again after PASS).
- Follow the skill's explicit instruction not to invent that workflow. No figures are generated; accepted numerical tables and raw traces are retained.
- The user's requested literature investigation, mechanism comparison, 0%-100% simulations, predictions and follow-up experiments are complete. The unavailable optional figure stage does not justify withholding those results. No mechanism is claimed to be identified in the user's culture.
