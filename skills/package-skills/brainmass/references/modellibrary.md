# BrainMass model library

Use this reference when selecting a public BrainMass model, comparing dynamical cost, identifying its primary observable, or distinguishing variants with similar names. Return to `skills/package-skills/brainmass/SKILL.md` after selection for the canonical `Simulator` workflow.

## Inspect the public catalogue

| API | Description |
|---|---|
| `brainmass.list_models()` | Return one `ModelInfo` record per public model, including user-facing HORN and network models. |
| `brainmass.list_models.to_table()` | Format the current catalogue for quick REPL inspection without hard-coding the inventory. |
| `brainmass.ModelInfo(name, category, n_state_vars, use_case)` | Represent one immutable catalogue entry for filtering and routing. |

```python
import brainmass

models = brainmass.list_models()
cheap_physiological = [
    model
    for model in models
    if model.category == "physiological" and model.n_state_vars <= 3
]

assert any(model.name == "WilsonCowanStep" for model in cheap_physiological)
assert next(
    model for model in models if model.name == "JansenRitStep"
).n_state_vars == 6
```

Use `list_models()` as the authority for the installed release. The table below is the source-documented inventory used by this skill, not a substitute for runtime discovery.

## Model inventory

Start with the cheapest model that represents the required regime, timescale, and observable. Add physiological detail only when it changes the scientific comparison; an optimizer cannot repair a mismatched model or measurement path.

| Model | Category | State variables | Use and boundary |
|---|---|---:|---|
| `HopfStep` | Phenomenological | 2 | Use as the cheap default for oscillation onset and rhythms; `a` crosses the supercritical Hopf bifurcation. |
| `VanDerPolStep` | Phenomenological | 2 | Use for nonlinear relaxation oscillations rather than near-Hopf amplitude dynamics. |
| `StuartLandauStep` | Phenomenological | 2 | Use when explicit amplitude and phase behavior matter beyond the canonical Hopf workflow. |
| `FitzHughNagumoStep` | Phenomenological | 2 | Use for fast-slow excitable population events, not conductance-based point neurons. |
| `ThresholdLinearStep` | Phenomenological | 2 | Use for fast threshold-linear E/I responses without a detailed physiological mechanism. |
| `Generic2dOscillatorStep` | Phenomenological | 2 | Use for configurable planar TVB dynamics when the fixed oscillator forms are too restrictive. |
| `LorenzStep` | Phenomenological | 3 | Use for chaos studies or as a nonlinear coupling test fixture. |
| `LinearStep` | Phenomenological | 1 | Use as an analytically simple baseline before diagnosing integration or coupling in a complex model. |
| `WilsonCowanStep` | Physiological | 2 | Use as the canonical physiological two-population E/I firing-rate model. |
| `JansenRitStep` | Physiological | 6 | Use for EEG and alpha-rhythm cortical columns; compare the derived `eeg()` observable. |
| `WongWangStep` | Physiological | 2 | Use the reduced `S1`/`S2` competing-population model for perceptual decisions. |
| `WongWangExcInhStep` | Physiological | 2 | Use the `S_E`/`S_I` model for resting-state BOLD, FC, or local E/I balance; it is not `WongWangStep`. |
| `MontbrioPazoRoxinStep` | Physiological | 2 | Use when firing rate and mean membrane potential must follow the exact QIF mean-field reduction. |
| `CoombesByrneStep` | Physiological | 2 | Use when synaptic conductance belongs explicitly in a next-generation mean-field model. |
| `LarterBreakspearStep` | Physiological | 3 | Use for conductance-based cortical limit cycles and chaotic dynamics. |
| `EpileptorStep` | Physiological | 6 | Use for seizure onset and offset; `x0` controls epileptogenicity and `lfp()` provides the proxy. |
| `KuramotoNetwork` | Network | 1 | Use when phase is the modeled State and synchronization order is the main summary. |
| `HORNStep` | Network | 2 | Use for one explicit harmonic-oscillator recurrent update. |
| `HORNSeqLayer` | Network | 2 | Use when composing one sequential HORN layer inside a larger model. |
| `HORNSeqNetwork` | Network | 2 | Use for oscillatory sequence learning; open `horn-task-training.md` instead of `Fitter`. |

## Validate a candidate

Use the uniform `Simulator` call to compare candidates under the same duration, `dt`, transient, and number of regions. Compare the actual scientific observable, not arbitrary internal State.

```python
import brainunit as u

jansen_rit = brainmass.JansenRitStep(in_size=8)
result = brainmass.Simulator(jansen_rit, dt=0.1 * u.ms).run(
    400.0 * u.ms,
    monitors=lambda model: model.eeg(),
    transient=100.0 * u.ms,
)

assert result["output"].shape == (3000, 8)
```

Use `JansenRitTR` only when its internal substep-to-TR behavior is explicitly required. Instantiate concrete model classes rather than shared bases such as `XY_Oscillator` or `WilsonCowanThreePopBase`.

## Choose an inhibitory intervention

Match the intervention to the model's equations; similarly named inhibitory parameters do not represent the same mechanism or unit.

| API | Description |
|---|---|
| `JansenRitStep(Ai=...)` | Use for inhibitory postsynaptic gain in an alpha/EEG cortical column; `Ai` is measured in mV and scales the inhibitory-population filter, so lowering it weakens the inhibitory postsynaptic potential represented in `eeg() = E - I`. |
| `WilsonCowanStep(wEI=...)` | Use for inhibitory feedback onto the excitatory rate population; `wEI` is dimensionless I-to-E coupling and enters the excitatory equation subtractively, so lowering it weakens that coupling. |

Change only the selected inhibitory parameter in a matched causal comparison. Treat defaults and source-documented estimation ranges as context, not as evidence for a chosen intervention magnitude. Label an unsourced, outcome-tuned, or outside-range value as phenomenological and show nearby sensitivity before claiming a robust regime change.

## Choose an Epileptor focus and input path

Use `x0` to represent autonomous regional epileptogenicity; use `x1_inp` or `x2_inp` only when the scientific intervention or network current is an input, and enable the corresponding gain explicitly.

| API | Description |
|---|---|
| `EpileptorStep(x0=...)` | Use a region-wise value to distinguish an epileptogenic focus from healthier tissue; less-negative `x0` increases epileptogenicity, while more-negative values render a region healthy. |
| `EpileptorStep(Kvf=..., Ks=...)` | Use when `x1_inp` should affect the fast `x1` subsystem and/or slow `z` recruitment path; `Kvf` scales its effect on `x1`, `Ks` scales its effect on `z`, and both default to zero. |
| `EpileptorStep(Kf=...)` | Use when `x2_inp` should affect the ultra-slow `x2` subsystem; `Kf` scales that input and defaults to zero. |
| `EpileptorStep.update(x1_inp=None, x2_inp=None)` | Use for one explicit step with population-specific coupling or external inputs; an input has no deterministic effect through a gain left at its zero default. |

```python
import jax.numpy as jnp

epileptor = brainmass.EpileptorStep(
    in_size=3,
    x0=jnp.asarray([-1.6, -2.4, -2.4]),
    Kvf=0.5,
    Ks=0.1,
)
```

This construction represents one autonomous focus and enables a first-channel network current to affect both fast activity and the slow permittivity path. Set only the gains required by the declared mechanism. Do not conclude that Epileptor lacks driven inputs from a trial that leaves `Kvf`, `Kf`, and `Ks` at zero.

## Extend Wong-Wang population currents

Use stock `update()` for bounded sensory evidence; compose the public current, transfer, derivative, and integration APIs only when a mechanism adds a distinct population-specific current.

| API | Description |
|---|---|
| `WongWangStep.update(coherence=...)` | Use for one standard decision step; `coherence` is motion evidence in `[-1, 1]`, not an arbitrary current. |
| `WongWangStep.compute_inputs(coherence=..., noise_1_val=..., noise_2_val=...)` | Use when extending the current path; it combines valid coherence and actual stochastic currents and returns total population currents `I1` and `I2`. |
| `WongWangStep.phi(I)` | Use after all current-valued mechanisms are applied; it maps one nA input current to a firing rate in Hz. |
| `WongWangStep.dS1_dt(S1, r1)` | Use with the population-1 rate to preserve the model's public `S1` gating equation. |
| `WongWangStep.dS2_dt(S2, r2)` | Use with the population-2 rate to preserve the model's public `S2` gating equation. |
| `brainstate.nn.exp_euler_step(fn, state, *args)` | Use under an active `dt` context to advance each public gating derivative and return its next State value. |

```python
import brainmass
import brainstate
import brainunit as u

model = brainmass.WongWangStep(in_size=1)
brainstate.nn.init_all_states(model)

adaptation_1 = 0.02 * u.nA
adaptation_2 = 0.0 * u.nA
with brainstate.environ.context(dt=0.1 * u.ms):
    current_1, current_2 = model.compute_inputs(coherence=0.0)
    rate_1 = model.phi(current_1 - adaptation_1)
    rate_2 = model.phi(current_2 - adaptation_2)
    model.S1.value = u.math.clip(
        brainstate.nn.exp_euler_step(model.dS1_dt, model.S1.value, rate_1),
        0.0,
        1.0,
    )
    model.S2.value = u.math.clip(
        brainstate.nn.exp_euler_step(model.dS2_dt, model.S2.value, rate_2),
        0.0,
        1.0,
    )
```

Keep actual noise in `noise_1_val` and `noise_2_val`; apply adaptation, stimulation, or another named mechanism explicitly to the returned currents. Do not force a current through out-of-range coherence or copy the model equations when these public methods preserve them.

## Choose Wong-Wang decision semantics

Choose the decision rule from the scientific definition; thresholded firing-rate decisions and final gating dominance are not equivalent.

| API | Description |
|---|---|
| `WongWangStep.get_decision(threshold=15 * u.Hz)` | Use when a firing-rate threshold defines commitment; it returns `1`, `-1`, or `0` for population 1, population 2, or undecided. |
| `(model.S1.value - model.S2.value) > 0` | Use when final gating dominance intentionally forces every trial into a binary choice; keep it as explicit model-analysis logic. |

Do not replace one rule with the other without updating probability denominators, uncertainty calculations, and handling of undecided trials.

## Application scripts

Open only the script that matches the selected model or scientific regime.

| Script | Open when |
|---|---|
| `scripts/hopf-bifurcation-single-node.py` | Demonstrating oscillation onset across the Hopf bifurcation or checking settled amplitude against `sqrt(a)`. |
| `scripts/wilson-cowan-ei-dynamics.py` | Simulating E/I population rates, their phase portrait, or an excitatory-drive sweep. |
| `scripts/jansen-rit-eeg-proxy.py` | Simulating the Jansen-Rit EEG proxy, its spectrum, or an input-rate sweep. |
| `scripts/seizure-epileptor-case-study.py` | Comparing healthy and seizure-like Epileptor regimes across epileptogenicity `x0`. |
| `scripts/wong-wang-decision-making.py` | Running stochastic competing-population decisions or estimating a psychometric curve. |
| `scripts/kuramoto-synchronization.py` | Measuring synchronization through the Kuramoto order parameter while varying coupling strength. |
| `scripts/wong-wang-dmf-resting-state.py` | Inspecting resting excitatory rate and gating in the Wong-Wang excitatory-inhibitory dynamic mean-field model. |
| `scripts/linear-baseline-node.py` | Checking integration against the analytical linear decay before diagnosing a more complex model. |

## Common selection failures

- Choosing a complex six-State model before a cheap model establishes the regime.
- Treating `WongWangStep` and `WongWangExcInhStep` as interchangeable.
- Passing an arbitrary current through bounded Wong-Wang `coherence` or hiding a non-noise mechanism in `noise_1_val` or `noise_2_val`.
- Fitting a BOLD target without a suitable slow neural source and hemodynamic observation model.
- Treating a neural-mass model as a point-neuron or cellular model.
- Assuming the inventory from this file supersedes `list_models()` in another installed release.

## Official sources

- `https://brainx.chaobrain.com/brainmass/reference/models.html`
- `https://brainx.chaobrain.com/brainmass/reference/generated/brainmass.WongWangStep.html`
- `https://brainx.chaobrain.com/brainmass/reference/utilities.html`
- `https://brainx.chaobrain.com/brainmass/howto/choose_a_model.html`
