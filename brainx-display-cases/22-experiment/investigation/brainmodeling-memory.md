# Brain modeling memory

## Checkpoint
- Iteration: 1
- Step: 0

### Artifacts
- `NeuroSpecification.md`: locked exploratory comparison, implementing the user's authorization.

### Important milestones
- Entry case: fresh-new. Existing `../prompt.md` and unrelated case 21 are preserved.
- Literature gate triggered: exact mechanism, regional difference, and recency are unresolved.
- Matching rat/DCZ paper adds hDlx-positive S1 and S2; it explicitly leaves mechanism unresolved. User's mouse observation is not relabeled as rat data.
- Pending optional clarification on ligand, dose, timing, and EEG does not block an explicitly uncalibrated comparison.

## Literature evidence: S2 inhibitory DREADD paradox - 2026-09-07
- Research question: Which mechanisms explain promoter and regional specificity?
- Essential papers: Yoshinaga and Sato 2026, PMID 41962964, DOI 10.1152/jn.00074.2026 (abstract and publisher-indexed discussion; resolver blocked); Lopez et al. 2016, PMC4804014 (published full text); Goldenberg et al. 2023, PMID 35788286 (abstract); Stachniak et al. 2014, PMC4306349 (accepted manuscript, mechanistic sections); Veres et al. 2023, PMC10088982 (published full text); Chang et al. 2022, PMC8741167 (published full text); Wu et al. 2025, DOI 10.1101/2025.10.02.676593 (preprint abstract, resolver PDF-only).
- Modeling consequences: Separate cell firing from transmitter output; separate local disinhibition from downstream feedforward inhibition; model regional circuit and targeting alternatives independently. CaMKIIalpha promoter is not a perfect specificity control.
- Limitations: No direct subsequent resolution found; original has zero Europe PMC indexed citations at search. Wu's preprint is a related counterexample, not an S2 mechanism resolution. No S2-specific synaptic/biophysical parameters established.

## Checkpoint
- Iteration: 1
- Step: 1

### Artifacts
- `brainx-study-record.md`: selected scale, exact model API, lifecycle, assumptions, and validation plan.

### Important milestones
- Aggregate populations only: BrainMass owns scientific dynamics. BrainState/BrainUnit support composition and quantities. No cell, point-neuron, or training skills required.
- Study completed before implementation. Existing BrainX environment is usable; no environment mutation.

## Literature evidence: updated retrieval depth - 2026-09-07
- Supersedes the earlier abstract-only status for Wu et al. 2025: explicit bioRxiv source selection retrieved the full submitted manuscript. It proposes HCN modulation, disfacilitation of interneurons, and homeostatic adaptation; it does not isolate these mechanisms. The version resolver reports revision 1 and no published DOI.
- Publisher-indexed Results/Discussion of Yoshinaga and Sato were also readable through web search. These are partial full-text sections, not complete resolver retrieval. They discuss local susceptibility, laminar targeting and remote circuits, with no mechanistic resolution.
- Date-limited Europe PMC search after 2026-04-10 found only the matching report; its indexed forward citation list is empty. Absence of indexed follow-up is not proof of absence of all unpublished work.

## Checkpoint
- Iteration: 1
- Step: 2

### Artifacts
- `models.py`, `experiment.py`, `make_configs.py`, `test_models.py`: native population dynamics, immutable runner, frozen comparison configuration generator and checks.
- `validation.json`: all focused checks passed with float64.
- `runs/exploration`, `runs/variants`, `runs/variants_corrected`: retained exploratory evidence.

### Important milestones
- Analytic derivative comparison uncovered reversed cross-weight names in the public implementation versus rendered documentation. Corrected constructor mapping, independently verified derivatives, and excluded initial synaptic exploration; somatic dynamics were preserved.
- No training/fitting coverage. Models test sufficiency and negative controls; settings are outcome-selected illustrations with separately frozen neighboring validation.

## Checkpoint
- Iteration: 1
- Step: 3

### Artifacts
- `validation.json`: exact reset, batched/single, JIT/non-JIT and stock-model parity; equation and time-axis checks.
- `configs/`: frozen central, smoke, timestep, solver and sensitivity configurations.

### Important milestones
- Acceleration unchanged: use Simulator's compiled loop and native independent condition axis. No speedup claim or external numerical integration.
- Proceed to CPU smoke and separate immutable numerical validation runs.

## Checkpoint
- Iteration: 1
- Step: 4

### Artifacts
- `runs/smoke`, `runs/central`, `runs/sensitivity`, `runs/half_dt`, `runs/solver`: completed immutable CPU runs with sources, configuration, environment, raw data, metrics, status and logs.
- `assessment.json`, `artifact-manifest.json`: 26 deterministic claim checks, neighboring results and numerical agreement.
- `report.md`, `model-methods.md`, `literature-review.md`, `follow-up-experiments.md`: scientific claim boundaries and proposed experiments.

### Important milestones
- Three local parameterizations reproduce the qualitative control pattern; pure downstream gate withdrawal fails the CaMKIIalpha control. No model generates motor convulsions or measured EEG.
- Neighboring matches: 13/15 somatic, 8/9 synaptic, 3/3 targeting; failed conditions retained. All 57 central labels preserved by dt refinement and second solver.
- Late inhibitory population activity can increase despite a directly inhibitory perturbation; per-spike efficacy and network-recruited activity must be distinguished.
- Ready for independent MCP review of numerical sufficiency and scoped conclusions.

## Checkpoint
- Iteration: 1
- Step: 5

### Artifacts
- `reviews/iteration-1.md`: verbatim independent MCP review.
- Reviewer thread: `01a07ae1-79f5-7871-88e9-ed8e28bade45`.

### Important milestones
- Review outcome PASS; scientific outcome SUPPORTED for the explicitly scoped computational claims. No findings; no revision requested.
- Reviewer independently reproduced raw-data metrics, all sensitivity counts, all 57 central classifications and all 83 artifact hashes.
- Fresh reviewer execution of derivative/parity tests was blocked by its read-only Matplotlib cache. The successful local tests remain preserved in `validation.json`; independent raw-data calculations were completed.
- Biological parameter calibration, literature validity and mechanism ranking are outside numerical review. Acceptance does not identify an experimental cause.

## Checkpoint
- Iteration: 1
- Step: 6

### Artifacts
- Accepted numerical artifacts, report and comparison tables preserved.

### Important milestones
- Optional visualization workflow blocked: the installed and repository `brainx-modeling-loop` skills do not contain their planned `references/visualization-workflow.md`. The skill explicitly requires preserving accepted artifacts and recording this condition until that reference exists.
- No visualization workflow was invented. The user's requested literature analysis, minimal simulations, model comparison and experimental predictions are complete in the Markdown report and raw artifacts.
