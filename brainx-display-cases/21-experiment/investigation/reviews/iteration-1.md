# BrainX iteration review

- **OUTCOME:** `REFUSE`
- **SCIENTIFIC_OUTCOME:** `SUPPORTED`
- **LOSS_CLOSURE:** `NOT_APPLICABLE`
- **OPTIMIZATION_ADEQUACY:** `NOT_APPLICABLE`
- **NEXT_ACTION:** `RETURN_TO_STUDY`

## Good-enough reason

The bounded mechanistic conclusions are supported. Inspected raw trajectories reproduce their metrics; all 44 assessment conditions, 560 fraction-summary rows and 176 manifest hashes check out. Controls preserve the optical/electrical distinction and unfavorable transmission-loss evidence. Acceptance is withheld for the narrow BrainTools-native implementation issue below, not a demonstrated scientific failure.

## Findings

### BX-001: Rescue stimulus bypasses the routed input API

- **Severity:** `minor`
- **Location:** `rescue.py:22`; `brain-tools-api-gap.md:11`
- **Problem:** The rescue reconstructs piecewise-constant current inside every timestep using time predicates. The selected input reference explicitly assigns this operation to `braintools.input.Constant`, generated once under the rollout `dt`. The gap artifact identifies no missing capability; describing these masks as an intervention does not establish an API gap.
- **Scientific consequence:** No incorrect rescue result was found: recorded pre-intervention trajectories match exactly, and recovery summaries reproduce. Nevertheless, the implementation fails the required native-input boundary, whose unjustified bypass requires refusal under the review contract.
- **Minimum fix:** Generate the three current protocols with `braintools.input.Constant` and pass their samples into `brainstate.transform.for_loop`. Retain the channel State intervention. Verify waveform and rescue-trajectory parity, then update the source snapshot and manifest. No model or parameter change is needed.

## Unverified assumptions

- Human-culture relevance of the local sodium sensor, imposed transmission impairment and optical mixture remains unmeasured, as explicitly acknowledged.
- Primary-literature claims could only be assessed through the supplied evidence summaries; primary texts were not supplied or independently retrieved.
- Fresh BrainX execution was blocked by Matplotlib requiring a writable cache during import. Offline artifact verification succeeded; saved execution and numerical-test evidence were inspected.
