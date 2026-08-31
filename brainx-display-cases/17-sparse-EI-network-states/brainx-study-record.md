# BrainX study record

## Selected scale and packages

- Biological scale: point-neuron spiking network only.
- Modeling owner: BrainPy-State.
- Supporting infrastructure: BrainEvent for fixed-fan-in binary event communication, BrainState for Module State, delays, randomness, environments, JIT, and time control flow, and BrainUnit for voltage, time, and frequency quantities.
- Excluded scales: no ion/channel/compartment mechanism and no aggregate neural-mass State is represented.
- Execution dependency check: `brainpy`, `brainevent`, `brainstate`, and `brainunit` import successfully in the intended project Python environment.
- Optional training/fitting coverage: none; this is a forward simulation.

## Sources studied

- Brunel (2000), Eqs. 1-2, Model A parameters, Section 6, Fig. 8, and Table 1.
- `brainx-general-guard` and `brainx-modeling-loop`.
- BrainPy-State root skill, `references/projection-patterns.md`, `references/braintools/connectivity.md`, and `references/scripts/109_fast_global_oscillation.py`.
- BrainEvent root skill, `references/connectivity-variants.md`, and `references/scripts/coba_ei_teaching.py`.
- BrainState root skill, `references/brainstate/randomness-and-reproducibility.md`, `references/brainstate/brainstate-control-flow-patterns.md`, and `references/simulation-environment.md`.
- BrainUnit root skill.

## Scientific translation

The membrane follows `tau_m dV/dt = -V` between events. A delayed excitatory or external event increments `V` by `J`; a delayed inhibitory event increments it by `-gJ`. A threshold crossing emits a spike, sets `V` to `Vr`, and makes voltage insensitive to all stimulation for `tau_ref`. Both populations share the same Model A dynamics.

The paper specifies `NE=10000`, `NI=2500`, and probability 0.1, equivalently exact mean fan-ins `CE=1000` and `CI=250`. Encode these as exact fixed fan-in because it preserves the paper's analytical `CE` and `CI` at every target and makes the realized graph inspectable. Sample unique sources without replacement and exclude autapses. Represent external input by the superposition theorem: the sum of `Cext=1000` independent Poisson trains is one per-neuron Poisson count with mean `Cext*nu_ext*dt = eta` at `dt=0.1 ms`.

## API and lifecycle design

| Need | API and invariant |
|---|---|
| Neuron graph | Subclass `brainstate.nn.Module`; store voltage in `HiddenState`, and refractory/spike runtime values in `ShortTermState`. Preserve shape, dtype, and units across writes. |
| Exact leak | Apply `V * exp(-dt/tau_m)` outside refractory, then add event jumps in the same step. |
| Sparse communication | Wrap delayed E and I spikes in `brainevent.BinaryArray` and multiply by two `FixedNumPerPost` structures shaped `(NE, N)` and `(NI, N)`. |
| Delay | Use `brainstate.nn.Delay` with a fixed Boolean population shape and retrieve exactly 15 steps after emission. External Poisson drive is not delayed: it represents spikes arriving from outside at the current soma time, consistent with the paper's stationary external processes. |
| Randomness | Use independent seeded streams for connectivity, initial voltage, external Poisson input, and probe selection. Reseed and reset before independent replay. |
| Environment | Scope `dt` around construction and rollout; scope `t` and integer `i` inside every network update. |
| Time execution | Put all timesteps in one `brainstate.transform.for_loop`, wrapped once by `brainstate.transform.jit`; do not use a Python timestep loop. |
| Host boundary | Convert completed arrays to NumPy only for metrics, serialization, hashing, and plotting. Keep units until explicit `to_decimal(...)` boundaries. |

## Update order

At timestep `i`: retrieve delayed recurrent spikes; communicate delayed E/I events; draw the independent aggregate external count; leak non-refractory voltages and add recurrent/external jumps; suppress all voltage changes during refractory; threshold, reset, and store the current spike; then push that spike into the delay buffer. Under this canonical end-of-step emission convention, an impulse returned by source step 0 is observed at receiving array index 16: the interval from the end of source step 0 to the start of receiving step 16 is exactly 15 bins, or 1.5 ms.

## Outputs and validation

Return only the current full-population spike vector from the transformed step. After the run, discard burn-in on the host, reduce full spikes to E/I/global counts, preserve one fixed 50-neuron raster, compute per-neuron ISI CV and Welch spectra, and save compact raw arrays. Plot the paper's layout only from review-accepted raw evidence.

Focused tests cover units/rates, connectivity, delay, refractory/reset, deterministic replay, no-recurrence external drive, output shapes, metric behavior, and eager/compiled parity. A representative benchmark must establish the performance baseline before any acceleration change.

## Implementation boundary

Do not use BrainMass population-rate State, dense recurrent matrices, generated probabilistic connectivity, explicit external afferent populations, raw JAX transforms over State, Python timestep loops, or post-hoc tuning of the locked paper parameters or acceptance thresholds.
