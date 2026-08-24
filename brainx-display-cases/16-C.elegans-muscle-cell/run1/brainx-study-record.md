# BrainX study record

## Represented scale and package ownership

- Biological scale: one isopotential muscle cell with ions, HH gates, calcium feedback, and current clamp.
- Modeling owner: `braincell.SingleCompartment`; no point-neuron network or neural-mass scale is represented.
- Physical quantities: `brainunit` quantities remain attached through parsing boundaries, parameter reconstruction, simulation, and comparison.
- Runtime state: BrainCell gate/voltage/calcium State is initialized or reset at every independent rollout and advanced with `brainstate.transform.for_loop` under `brainstate.environ.context`.

## Selected API and lifecycle

1. Parse the ATF table on the host, explicitly attach `u.ms`, `u.mV`, and `u.pA`, and retain an immutable SHA-256 identity.
2. Construct `braincell.SingleCompartment(size=condition_shape, C=..., area=..., solver=...)` with one independent cell per candidate or stimulus lane.
3. Attach fixed sodium and potassium ions, dynamic calcium, custom source-named HH channels, a K/Ca `MixIons` owner for SLO-2, and root-cell `braincell.channel.IL` leak.
4. Custom HH gates use BrainCell's `Gate`/`HH` lifecycle: steady state and tau functions, ion-supplied reversal/concentration data, and `g_max * gates * (E - V)` current sign.
5. Initialize all voltage, calcium, and gate State from the measured pre-stimulus voltage; reset these States between candidates and stimuli without carrying fitted values.
6. Advance the complete time-major protocol with `brainstate.transform.for_loop`; a host loop is allowed only across optimizer starts or independent result serialization.
7. Convert voltage to mV only at the explicit observation/scoring boundary. Use fixed spike extraction and raw-voltage metrics on aligned times.

## Fitting design

- Active coverage: parameter fitting.
- Parameter map is explicit and ordered. Fitted quantities are maximal conductances and capacitance; reversals and gate kinetics are fixed and disclosed because one training trace cannot constrain all channel equations.
- Use bounded Braintools `ScipyOptimizer` when its single-objective contract is sufficient; otherwise use SciPy only as the host-side search boundary while every candidate is evaluated through the BrainCell simulator.
- Candidate evaluation reconstructs units, resets all runtime State, runs exactly the 25 pA protocol, applies the direct-voltage observation model, and returns one scalar objective.
- The objective combines robust voltage discrepancy with fixed event-feature penalties so a low-amplitude average cannot beat the spike timing/count behavior. Three predeclared starts are retained.
- Exact-pipeline recovery uses the same duration, sampling, protocol, parameter map, bounds, objective, and start-selection rule. Recovery gates mechanistic interpretation per parameter, not the predictive result.

## Validation and implementation checks

- Formula: gates remain in [0, 1] within tolerance, derivatives point toward steady state, and each current is zero at its reversal potential.
- Lifecycle: init/reset determinism, independent condition lanes, finite nominal and boundary runs, no pre-stimulus spikes, correct total-current units, and time/output shape.
- Observation: exact 0.1 ms alignment to every second experimental sample, no held-out normalization, and fixed spike detector.
- Controls: passive leak-only response and zero-current baseline use identical initialization, solver, duration, and scoring path.
- Claims: predictive criteria are locked in `NeuroSpecification.md`; parameter identity is withheld when recovery is poor.

## Source patterns studied

- BrainX general guard and complete BrainCell, BrainUnit, and BrainState root skills.
- BrainCell single-compartment HH pattern, area scaling, ion library, channel library, custom HH authoring, K/Ca MixIons adaptation, and official HH fitting example.
- BrainX parameter-fitting workflow, BrainState constrained-parameter lifecycle, and Braintools optimizer selection.

## Implementation design decision

Use BrainCell-native batched condition State and a BrainState-transformed time loop. Keep fixed kinetic constants in plainly named channel classes, fit only the lower-dimensional physical parameter map, serialize every start and held-out prediction, and label the model as a predictive phenomenological HH model rather than an exact paper reimplementation.
