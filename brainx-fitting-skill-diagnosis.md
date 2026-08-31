# BrainX fitting skill diagnosis

## Conclusion

The run is BrainX-native at the model and execution layers, but not at the
fitting, input-protocol, or regression-metric layers. The agent correctly used
BrainCell, BrainState transforms, BrainUnit, and BrainTools initializers in
`model.py`. It then used raw SciPy optimization, handwritten current generation,
and handwritten regression losses even though BrainTools owns those operations.

This is primarily a skill-routing failure, followed by an agent-compliance
failure. The skills strongly prefer BrainTools, but the BrainCell fitting route
does not deliver the BrainTools references or a complete dependency-aware
backend decision. The agent then treated missing Nevergrad as permission to use
raw SciPy without first testing `braintools.optim.ScipyOptimizer` or recording a
specific wrapper capability gap.

The modeling loop compounds this problem after review. A reviewer `REFUSE`
currently returns directly to implementation at step 2. The next iteration can
therefore patch symptoms while preserving the same incomplete package study and
API choices. A refusal must begin the new iteration at step 1 so the agent
reopens the relevant BrainX package skills, references, and canonical scripts
before revising code.

BrainState should not receive optimizer documentation. BrainState owns State,
environments, initialization, and State-aware transforms. BrainTools owns the
optimizer, metric, and input APIs required here.

## What the run did

| Area | Implementation | Assessment |
|---|---|---|
| Cell model | `braincell.SingleCompartment`, BrainCell ions/channels, BrainUnit quantities | BrainX-native |
| Stateful rollout | `brainstate.environ.context` and `brainstate.transform.for_loop` | BrainX-native |
| Parameter initialization | `braintools.init.param` and `braintools.init.Constant` | BrainX-native |
| Optimizer | `scipy.optimize.differential_evolution` in `inference.py` | External fallback without a demonstrated BrainTools gap |
| Waveform loss | Handwritten NumPy RMSE/MAE in `inference.py` | Duplicates `braintools.metric` |
| Spike extraction | `scipy.signal.find_peaks` on measured and predicted voltage | Defensible observation-boundary custom logic; no equivalent voltage-peak API is routed |
| ATF parsing | `csv`, NumPy, `Path`, and SHA-256 in `data.py` | Defensible host file-I/O boundary; no BrainTools ATF loader is documented |
| Current protocol | Handwritten time mask and `u.math.where` in `data.py` | Duplicates `braintools.input.Constant` |
| Initial voltage | `u.math.mean` over the pre-stimulus interval | Correct BrainUnit operation |

The problem is therefore not that every line of `data.py` must use BrainTools.
Custom parsing is appropriate for the supplied Axon Text File. The avoidable
part is `current_protocol()`, which reconstructs a standard
baseline-pulse-baseline signal that BrainTools already represents directly.

## Why raw SciPy was selected

### 1. The BrainCell skill routes fitting to a Nevergrad-only example

The BrainCell root skill routes fitting to
`references/scripts/fitting_hh_neuron.py`. That example uses
`braintools.optim.NevergradOptimizer`, but the root does not route a BrainTools
optimizer, metric, or input-current reference. An agent following only the
selected cellular scale sees one concrete derivative-free backend: Nevergrad.

### 2. The parameter-fitting reference mentions APIs it does not operationalize

The fitting reference names both `ScipyOptimizer` and `NevergradOptimizer` in
the backend decision table, but its candidate-evaluation API table includes
only `NevergradOptimizer`. Its only high-level canonical example is a BrainMass
gradient fit. It does not show a current BrainCell composition using:

```text
braintools.input -> BrainCell rollout -> braintools.metric ->
braintools.optim.ScipyOptimizer or NevergradOptimizer
```

The final route says to open the active package's BrainTools references, but the
BrainCell skill has no such references. This is a dead route for a single-cell
task.

### 3. Missing Nevergrad was mistaken for missing BrainTools optimization

The run's optimizer-boundary artifact tests only
`braintools.optim.NevergradOptimizer` and records that the optional `nevergrad`
dependency is absent. It does not test `braintools.optim.ScipyOptimizer`.

The installed BrainTools SciPy wrapper is usable in the same environment. A
documented-API smoke check successfully minimized a bounded objective with
`braintools.optim.ScipyOptimizer`. BrainTools input and metric checks also
produced the expected 5,000-sample pA protocol and scalar squared error.

Missing an optional Nevergrad dependency therefore explains why Nevergrad was
not used, but it does not justify bypassing BrainTools entirely.

### 4. The study record created an untested escape hatch

The study record says to use `braintools.optim.ScipyOptimizer` when its contract
is sufficient and raw SciPy otherwise. The implementation never records what
was insufficient. Raw `differential_evolution` does provide vectorized
population evaluation and a custom plateau callback, while the documented
BrainTools `ScipyOptimizer` contract is scalar-objective oriented. That could be
a real capability gap, but it must be demonstrated before choosing the external
API.

The correct conclusion is:

- derivative-free fitting was scientifically reasonable because spike-count and
  peak-timing terms are discontinuous;
- raw SciPy differential evolution may have useful capabilities;
- the run did not prove those capabilities were necessary or unavailable
  through BrainTools;
- the skill did not require that proof.

## Why BrainTools disappeared from `data.py`

The active BrainCell route contains no input-current reference. The only
shipped `braintools.input` guidance is under the BrainPy-State skill, which the
general guard correctly excludes because this is not a point-neuron network.
The agent therefore saw no routed cellular example for a generated stimulation
protocol and wrote the obvious `u.math.where` mask.

The skill should preserve these boundaries:

| Operation | Required owner |
|---|---|
| Parse a researcher-supplied ATF file | Host Python parser unless a documented BrainTools loader supports ATF |
| Attach and validate physical units | BrainUnit |
| Generate the inferred 0/25/0 pA protocol | `braintools.input.Constant` under the rollout `dt` |
| Compute pre-stimulus mean voltage | BrainUnit math |
| Encode analog voltage into spikes | Custom observation logic unless a routed BrainTools API matches the declared detector exactly |

This distinction prevents the opposite error: forcing custom scientific file
formats into an unrelated BrainTools encoder merely to increase BrainTools
usage.

## Required skill improvements

### P0: Return reviewer refusals to package study

Change the modeling-loop transition from:

```text
step 5 REFUSE -> new iteration step 2 implementation
```

to:

```text
step 5 REFUSE -> new iteration step 1 BrainX restudy -> step 2 implementation
```

The new iteration must not merely reread the previous study record. It must:

1. Map every reviewer finding to the owning BrainX package or workflow skill.
2. Reopen `brainx-general-guard`, every affected package skill, and each routed
   reference or canonical script that can change the correction.
3. Recheck high-level BrainX and BrainTools APIs before retaining custom or
   external infrastructure.
4. Write a new append-only study checkpoint that records what was newly learned,
   which prior API assumption changed or remained valid, and the resulting
   implementation decision.
5. Begin step 2 only after the restudy artifact is complete.

Require focused restudy of all skills implicated by the refusal, not a blind
reload of every ecosystem skill. For example, a refusal about a raw optimizer,
loss, or input protocol in a BrainCell fit must reopen BrainCell, the fitting
workflow, and the routed BrainTools optimizer, metric, and input references.

This transition is essential because review findings often expose an incorrect
abstraction choice, not only a local code defect. Returning directly to step 2
encourages the agent to add more custom code around the same mistake.

### P0: Give BrainCell a complete BrainTools fitting route

Add self-contained BrainTools references under the BrainCell skill for:

- optimizer selection, including `ScipyOptimizer` and `NevergradOptimizer`;
- regression and neuroscience metrics;
- unit-aware input-current generation.

Route them from the BrainCell root when a single-compartment task includes
fitting, injected-current protocols, or trace scoring. Do not route these
through BrainState.

The BrainCell fitting row should direct the agent to read the official HH script
and all three BrainTools references before implementing a fit.

### P0: Add a BrainCell-native backend ladder

Replace the ambiguous backend prose with an executable decision order:

1. Determine whether the complete declared objective is differentiable.
2. For valid gradients, use `brainstate.transform.grad` with a BrainTools
   gradient optimizer.
3. For a scalar bounded derivative-free objective, try
   `braintools.optim.ScipyOptimizer` first.
4. For population-batched derivative-free search, use
   `braintools.optim.NevergradOptimizer` when its optional dependency is already
   available or installation has been authorized.
5. Use raw SciPy only after recording the exact missing BrainTools capability,
   such as an unavailable differential-evolution strategy, vectorized candidate
   contract, callback, or stopping rule.

The backend must be locked only after checking required optional dependencies.
An absent optional backend should trigger the next BrainTools backend or a user
decision, not an implicit external fallback.

### P0: Require an external-API gap artifact

Any use of `scipy.optimize`, `scipy.signal`, raw JAX, or another generic library
for an operation assigned to BrainTools should produce a small artifact before
implementation:

```markdown
## BrainTools API gap

- Operation:
- BrainTools API checked:
- Required capability:
- Observed limitation or error:
- Why another BrainTools API does not satisfy the contract:
- Smallest external boundary:
- Unit, State, shape, and numerical parity check:
```

“The preferred optional dependency is missing” is insufficient when another
documented BrainTools API covers the operation.

### P1: Add one current-clamp fitting composition

The parameter-fitting reference needs one compact BrainCell example that shows
the complete ownership chain once:

```python
with brainstate.environ.context(dt=dt):
    current = braintools.input.Constant([
        (0.0 * u.pA, 50.0 * u.ms),
        (amplitude, 250.0 * u.ms),
        (0.0 * u.pA, 200.0 * u.ms),
    ])()

def waveform_loss(predicted, observed):
    mse = braintools.metric.squared_error(
        predicted.to_decimal(u.mV),
        observed.to_decimal(u.mV),
        reduction="mean",
    )
    return u.math.sqrt(mse)
```

The example should then show both standalone optimizer choices, with the
Nevergrad dependency boundary stated beside Nevergrad rather than discovered
after the implementation is complete.

### P1: Separate standard metrics from custom observation features

Require agents to decompose an objective into:

- BrainTools-owned standard losses, such as squared or absolute error;
- package-owned model observables;
- genuinely custom observation features, such as this experiment's fixed
  prominence-based voltage peak detector.

This run could use `braintools.metric.squared_error` for the waveform and
pre-stimulus terms while retaining a small custom spike detector. It should not
reimplement all metrics merely because one component is custom.

### P1: Add a step-2 BrainX ownership audit

Before coding, require a short table for every infrastructure operation:

| Operation | Intended owner | Selected API | External fallback evidence |
|---|---|---|---|

At minimum, list model construction, initialization, input protocol, rollout,
observation mapping, loss, optimizer, metrics, serialization, and plotting.
Step 2 should be incomplete while a BrainTools-owned operation names an
external API without a gap artifact.

### P2: Make the reviewer enforce the same boundary

The review should refuse code that imports raw `scipy.optimize` for fitting or
manually builds standard BrainTools inputs and metrics unless the submitted
iteration includes the API-gap artifact. Scientific correctness alone should
not turn an undocumented infrastructure fallback into a BrainX-native `PASS`.

### P2: Add a fitting-skill regression evaluation

Add an evaluation prompt with an inferred step-current protocol and a
single-cell voltage fit. The expected implementation should:

- preserve custom host parsing for the supplied recording format;
- use `braintools.input` for the generated protocol;
- use `braintools.metric` for standard regression terms;
- select a BrainTools optimizer through the dependency-aware ladder;
- use raw SciPy only with a concrete capability-gap artifact;
- keep BrainState limited to State and transform ownership.

This evaluates the decision boundary rather than checking for a superficial
`import braintools` string.

## Files that should change later

| File | Change |
|---|---|
| `skills/package-skills/braincell/SKILL.md` | Route current protocols, metrics, and optimizers for cellular fitting. |
| `skills/package-skills/braincell/references/braintools/input-current.md` | Add the shared input-current reference to the installed BrainCell skill. |
| `skills/package-skills/braincell/references/braintools/metric.md` | Add the shared metric reference to the installed BrainCell skill. |
| `skills/package-skills/braincell/references/braintools/optimizer.md` | Add the shared optimizer reference to the installed BrainCell skill. |
| `skills/brainx-modeling-loop/SKILL.md` | Change `REFUSE` to start the next iteration at step 1 and require a new restudy checkpoint before implementation. |
| `skills/brainx-modeling-loop/references/parameter-fitting-workflow.md` | Add the backend ladder, BrainCell composition, and external-gap gate. |
| `skills/brainx-general-guard/SKILL.md` | Require a recorded API gap before generic infrastructure replaces BrainTools. |
| `mcp-servers/codex/system-prompt.md` | Refuse undocumented external replacements for BrainTools-owned operations. |

## Acceptance criteria for the skill revision

The revision is successful when a fresh BrainCell fitting agent can answer all
of these questions from routed local Markdown without opening BrainPy-State:

1. Which package owns the model, State execution, units, current protocol,
   metric, and optimizer?
2. Which BrainTools optimizer works without Nevergrad?
3. When is Nevergrad appropriate, and what happens when its optional dependency
   is absent?
4. What evidence is required before raw SciPy is allowed?
5. Which parts of a custom scientific data loader should remain host Python?
6. How are standard waveform losses separated from custom spike features?
7. After reviewer `REFUSE`, which affected BrainX skills and references must be
   restudied before implementation resumes, and where is that new study recorded?

The desired result is not “use BrainTools everywhere.” It is “use each BrainX
owner wherever it has a documented operation, and make every external boundary
explicit, minimal, and evidenced.”
