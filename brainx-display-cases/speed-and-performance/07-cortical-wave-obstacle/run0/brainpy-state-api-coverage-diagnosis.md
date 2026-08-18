# BrainX diagnosis: cortical wave obstacle

## Evidence studied

- Generated artifacts: `README.md`, `cortical_wave.py`, `test_cortical_wave.py`, `pyproject.toml`, `results/outcomes.csv`, `results/phase_metrics.npz`, `results/wave_storyboard.png`, and `results/phase_map.png`.
- Execution: the default 42-condition run completed in the required BrainX virtualenv; all three focused tests passed by direct invocation. The reproduced CSV, NPZ, and storyboard matched the archived files byte for byte. The phase-map pixels differed after a fresh Matplotlib font-cache build, but its data and visual content matched.
- Owning skills: `brainx-general-guard`, `brainpy-state`, `brainevent`, `brainstate`, and `brainunit`.
- Routed references: BrainPy-State `component-selection.md` and `projection-patterns.md`; BrainEvent `sparse-formats.md`; BrainState `transformation-vmap-expansion.md`; BrainUnit `array-creation.md` and `array-mechanics.md`.
- Closest examples: `skills/brainpy-state/references/scripts/103_COBA_2005.py`, `skills/brainevent/references/scripts/coba_ei_teaching.py`, and `skills/brainpy-state/references/scripts/sound_localization.py`.
- Official contracts: BrainPy-State generated `LIFRef`, `Expon`, `COBA`, and `Dynamics` pages; BrainEvent generated `BinaryArray`, `CSR`, and `coo2csr` pages; BrainState generated `vmap2`, `vmap_init_all_states`, and `for_loop` pages; BrainUnit generated `meshgrid`, `arange`, and `Quantity` pages.
- Missing-owner check: official BrainTools `braintools.conn` pages for `Grid2d`, `DistanceDependent`, and `ConnectionResult`; `braintools.init.StepProfile`; and the official `braintools.input` API indexed by `source_html_references/braintool_html_reference.md`.

## Executive diagnosis

The implementation is executable, deterministic, unit-aware, event-driven, and correctly maps independent dynamical State inside one transformed time loop. It produces all requested outcome labels and readable figures.

The main API-coverage failure is upstream of BrainEvent storage: `make_local_csr()` manually constructs spatial neighborhoods even though BrainTools owns grid and distance-dependent topology generation. The spark is also hand-coded as a time comparison despite BrainTools owning unit-aware current protocols. The general guard names these owners, but BrainPy-State provides no precise local route to either existing authoring reference, so the agent followed the detailed BrainEvent storage route and missed the higher-level topology and input APIs.

The main scientific weakness is causal interpretation. Survival changes almost entirely at one inhibition threshold: radii 0-10 mm all reach the right edge for gains 0.64-0.76 and all die for gains 0.78-0.80; the 11.5 mm disk always dies. The `bends` row occurs because a 10 mm disk leaves six upper-transect sites and zero lower-transect sites. Because route fractions use `ever_active`, the label proves that one geometrically available corridor fired sometime, not that a coherent front traversed that corridor and then reached downstream tissue in order.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `cortical_wave.py:290` | Outcome metrics reduce the complete trajectory to `ever_active`; upper/lower activity is measured at the disk center without an arrival-order or downstream-continuation requirement. | A transient or pre-obstacle event can count as a route, so `bends` and `splits` are not causal propagation tests. | Require ordered activation from left of the obstacle through a corridor to a post-obstacle or right-edge region, or derive a first-arrival map and classify from its connected advancing front. |
| P1 | `cortical_wave.py:38` and `results/phase_map.png` | The sampled inhibition boundary is nearly independent of radius. At gain 0.76, right-edge arrival stays near 32.5-33.75 ms for radii 0-10 mm; at 0.78 every such condition dies. | The map reports the requested grid but provides little evidence of an interaction between obstacle size and inhibition. | Calibrate the sweep around a joint boundary and report a continuous metric such as reach, delay, or transmitted fraction so radius-dependent effects remain visible before categorization. |
| P1 | `cortical_wave.py:297` | For radius 10 mm the lower route mask is empty by geometry, while the upper mask has six sites; radius 11.5 mm leaves neither route. | The reported bend is imposed by clipping a large disk against the sheet boundary rather than emerging from symmetric obstacle interaction. | Keep the diagnostic disk within the sheet for unbiased bend/split claims, or state that the off-center/clipped geometry intentionally selects one corridor and classify it as a boundary-routing condition. |
| P2 | `results/wave_storyboard.png` | The intact sequence contains separated edge lobes and later residual activity, not only a compact advancing front. | The figure can be read as edge-dominated or reverberant activity rather than one cortical wave. | Add front-coherence, first-arrival, or post-front quiescence checks and select frames from the same causal-front metric used for classification. |
| P2 | `README.md:3` | Parameters are calibrated in code but no provenance or phenomenological-model boundary is stated. | Readers may over-interpret the regime as a biologically calibrated cortical sheet. | Label it as a demonstrative phenomenological regime unless parameters are tied to a source; preserve the no-spark control, which produced zero excitatory spikes. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Point-neuron E/I dynamics | Two `brainpy.state.LIFRef` populations | BrainPy-State `LIFRef` | Correct | Keep; initialization, units, refractory State, and one-step calls match the official contract. |
| Synaptic filtering | Three postsynaptic `brainpy.state.Expon` instances | BrainPy-State `Expon` | Correct | Keep; one conductance State per target site is the efficient aligned pattern. |
| Conductance-to-current conversion | Three `brainpy.state.COBA` outputs registered with `add_current_input()` | BrainPy-State `COBA` and `Dynamics.add_current_input()` | Correct | Keep; excitation and inhibition use explicit reversal potentials. |
| Spatial topology generation | Nested Python loops in `make_local_csr()` | BrainTools `Grid2d` or `DistanceDependent(StepProfile(...))` | Bypassed owner | Generate edge lists with BrainTools; retain custom topology only when documented families cannot express the neighborhood. |
| Explicit sparse storage | Direct `brainevent.CSR` construction | BrainEvent `CSR` | Correct storage, low-level construction | Convert the BrainTools `ConnectionResult.pre_indices/post_indices/weights` through `brainevent.coo2csr()`, preserving `weights[order]`. |
| Spike communication | `brainevent.BinaryArray(spikes) @ CSR` | BrainEvent `BinaryArray` and `CSR` product | Correct | Keep and verify output shape and orientation. |
| Brief edge stimulus | `t < SPARK_DURATION` plus masked current arithmetic | BrainTools `braintools.input` current protocols plus BrainUnit broadcasting | Bypassed owner | Generate the unit-aware pulse or section once, then apply the spatial edge mask. Keep the lesion clamp as model-specific per-condition logic. |
| Lesion geometry | Unit-aware distance comparison | BrainUnit quantities and `brainunit.math` | Correct | Keep; the circular mask preserves length units. |
| Condition grid | `u.math.meshgrid(..., indexing="ij")` | BrainUnit `meshgrid` | Correct | Keep and retain explicit axis order. |
| Independent condition State | `vmap_init_all_states()` and filtered `vmap2` input/output State axes | BrainState mapped initialization and `vmap2` | Correct | Keep `unexpected_out_state_mapping="raise"`; it protects against undeclared writes. |
| Time evolution | mapped step inside `brainstate.transform.for_loop`, wrapped by `brainstate.transform.jit` | BrainState `for_loop` and `jit` | Correct | Keep one compiled rollout and time-major outputs. |
| Reproducibility | Fixed seed before construction | BrainState randomness | Adequate for deterministic initialization | Keep; add independent stochastic replicates only if noise is introduced. |
| Outcome statistics | NumPy host analysis after simulation | Legitimate host boundary | Mechanically valid, scientifically weak | Replace `ever_active` route tests with time-resolved causal-front metrics. |
| CSV/NPZ persistence | Python `csv` and NumPy | Legitimate host boundary | Correct | Keep. |
| Figures | High-level Matplotlib | Legitimate presentation boundary | Readable | Keep, but align storyboard frames with validated front metrics. |

## Missing, bypassed, or misused BrainX APIs

### `braintools.conn.Grid2d` and `DistanceDependent`

Use `Grid2d` for immediate four- or eight-neighbor sheet wiring. Use `DistanceDependent(StepProfile(threshold, inside_prob=1, outside_prob=0), ...)` when a physical cutoff must reproduce a wider deterministic neighborhood. These APIs replace the nested topology loops and return a `ConnectionResult` with aligned pre/post indices, weights, optional delays, and metadata.

The current BrainPy-State skill does not route the existing `braintools-references/braintools-connectivity.md`, even though topology selection precedes BrainEvent storage selection.

### `brainevent.coo2csr`

Use this at the BrainTools-to-BrainEvent handoff. Pass `ConnectionResult.pre_indices` and `post_indices`; construct `brainevent.CSR((weights[order], indices, indptr), shape=result.shape)`. The returned permutation is mandatory because `coo2csr()` groups edges by row.

### `braintools.input`

Use `Section`, `Constant`, `Step`, or `Spike` for reusable unit-aware current protocols. A 3 ms spark is a direct current protocol, not data preprocessing or a BrainPy autonomous spike source. The current BrainPy-State skill has no local route to `braintools-references/braintools-input-current.md`, so the agent implemented the pulse manually.

No material misuse was found in `LIFRef`, `Expon`, `COBA`, `BinaryArray`, `CSR`, `vmap2`, mapped State initialization, `for_loop`, `jit`, or BrainUnit quantities.

## Performance and code simplicity

The simulation structure is strong: all 42 independent conditions are mapped over semantic dynamical-State roles, time is one transformed loop, and one JIT boundary encloses the rollout. The default run completed in about twenty seconds during independent review, including initial compilation and figure generation.

The main avoidable complexity is `make_local_csr()`: two nested lattice loops, offset enumeration, boundary checks, row pointers, and manual index bookkeeping duplicate BrainTools topology generation. Replacing it with a named connectivity pattern makes the scientific topology visible and leaves BrainEvent responsible only for sparse event storage and multiplication.

Host-side classification, persistence, and plotting are valid boundaries. Do not force them into BrainX, but keep classification time-resolved enough to support the scientific claim.

## Skill improvements

1. Add `skills/brainpy-state/references/braintools/connectivity.md` from the existing authoring source, with a BrainEvent handoff section that uses `ConnectionResult` edge order, `coo2csr()`, `weights[order]`, and `brainevent.CSR`.
2. Add `skills/brainpy-state/references/braintools/input-current.md` from the existing authoring source for unit-aware stimulus protocols.
3. Route both references from `skills/brainpy-state/SKILL.md`. Route connectivity beside projection/storage selection and input currents beside the canonical rollout/input decision.
4. Update `component-selection.md` so spatial topology is chosen through BrainTools before BrainEvent representation, and direct current protocols route to the input-current reference rather than being conflated with spike sources or data encoders.
5. Update `plan.md` to declare the two new first-level BrainTools references and their official sources. Do not change `brainx-general-guard`: it already assigns connectivity and inputs to BrainTools.

Do not add cortical-wave-specific thresholds or scientific claims to the root skill. The next run should show whether the general routing improvements are sufficient.

## Checks for the next run

- The exact prompt bytes and evaluation conditions remain unchanged.
- Spatial topology is generated by a documented BrainTools connectivity pattern, with explicit BrainEvent storage and a verified edge-order handoff.
- The edge spark is generated by a documented BrainTools input-current API and remains unit-aware.
- `BinaryArray @ connectivity` shape/orientation tests pass.
- Each mapped condition owns independent `HiddenState` and `ShortTermState`; the State filters declare input and write-back axes.
- One transformed time loop executes the mapped complete step; no Python timestep loop appears.
- A no-spark control remains quiescent.
- Front arrival progresses from left to right, and outcome labels require causal corridor-to-downstream activation rather than `ever_active` alone.
- The phase map reports a continuous transmission metric and shows whether both radius and inhibition change the result; it must not claim an emergent bend when one corridor is absent by construction.
- The figures are opened and checked for coherent propagation, visible obstacle geometry, readable axes/units, and agreement with saved metrics.
