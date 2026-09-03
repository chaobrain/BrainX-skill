# BrainX study record

## Scope and selected skills

- Represented scale: point neurons, fixed recurrent synapses, and a spiking network.
- Owning modeling skill: BrainPy-State.
- Supporting skills: BrainEvent for exact fixed fan-in event communication, BrainUnit for time/voltage/rate quantities, and BrainState for Module State, deterministic randomness, environment settings, and transformed time loops.
- Optional training/fitting coverage: none; this is forward simulation.
- Installation route: reuse `/home/yixinliu/anaconda3/envs/braincell-released`, whose `BrainX 2026.7.9` components match the compatibility matrix and whose JAX backend reports one CPU device.
- Excluded routes: BrainCell, BrainMass, BrainTrace, NEST compatibility, and legacy BrainPy do not own any represented mechanism. Bundled application scripts were not opened because the researcher explicitly prohibited deriving the model from existing code.

## Mental model and lifecycle

1. Construct one unit-aware `brainpy.state.LIFRef` population of 12,500 neurons with `V_rest = 0 mV`, the specified time constant, reset, threshold, and refractory interval.
2. Construct separate excitatory and inhibitory `brainevent.FixedNumPerPost` matrices. Their index arrays have shapes `(12,500, 1,000)` and `(12,500, 250)` and contain unique source indices per target. Exclude autapses within each source population.
3. Wrap each fixed-fan-in product in a BrainState Module and pass it to `brainpy.state.DeltaProj`. This preserves event-driven communication and makes each recurrent event an instantaneous voltage jump.
4. Maintain a 15-slot boolean ring buffer in `HiddenState`. At step `i`, read slot `pointer`, which contains spikes from step `i - 15`; deliver those events before the neuron update, then overwrite that slot with spikes emitted at step `i` and advance the pointer. An impulse test must prove the 15-step arrival.
5. Draw one aggregate external count for every target and step from `Poisson(Cext * nu_ext * dt)`. Superposition of the independent afferents makes this exactly the requested aggregate drive. A separate BrainState `RandomState` owns this stream and is reset to the matched external seed before each condition.
6. Call recurrent projections and the external delta projection before `LIFRef(...)`; the neuron then applies leak, accumulated delta jumps, threshold/reset, and refractory State in its native update. Set `dt`, `t`, and integer step `i` through `brainstate.environ.context`.
7. Run all 50,000 steps through one `brainstate.transform.for_loop` enclosed by `brainstate.transform.jit`. Use a small host loop only across the 12 independent seed-condition rollouts.
8. Reset all dynamical State, the delay buffer, online ISI accumulators, initial voltage, and the external RNG at every condition boundary. Change only `g` and `eta` within a seed; reuse its connectivity, initial voltage, and displayed neuron sample.

## Exact equations and grid convention

For non-refractory neuron `j`, BrainPy-State advances the current-based LIF state over one `dt = 0.1 ms` and applies accumulated delta input:

```text
tau_m dV_j/dt = -(V_j - 0 mV)
V_j <- V_j + JE * Kext_j + JE * nE_j - g * JE * nI_j
spike_j = V_j >= theta
V_j <- reset after a spike
```

Here `Kext_j ~ Poisson(Cext * nu_ext * dt)`, and `nE_j` and `nI_j` count delayed spikes among the target's exact recurrent afferents. Events generated in step `i` arrive in step `i + 15`. The native `LIFRef` owns the exact discrete refractory convention; a single-neuron check must verify that it prevents firing throughout the specified 2 ms interval.

The external rate remains unit-derived:

```text
nu_ext = eta * theta / (JE * CE * tau_m)
lambda_ext = Cext * nu_ext * dt
```

For the approved values `Cext = CE` and `dt / tau_m = JE / theta`, `lambda_ext = eta`, but the implementation computes the full expression and tests that simplification rather than hard-coding it.

## Randomness contract

- Root seeds: `11`, `29`, `47`.
- Derive independent connectivity, initialization/sample, and external-stream seeds from each root with a documented NumPy `SeedSequence` on the host boundary.
- Build connectivity and initial voltages once per root seed; reuse them unchanged across all conditions.
- Initialize voltage uniformly on `[10, 20) mV` and choose 50 global neuron indices without replacement.
- Reset the external BrainState stream to the same per-root external seed for every condition. Different `eta` values change the Poisson law but not the starting key.
- Repeat one smoke rollout after restoring its RNG seed and State; require exact output equality.

## Online observables

- Return E and I spike counts for every 0.1 ms step.
- Return spike indicators only for the fixed 50-neuron display sample.
- After the 1 s transient, maintain per-neuron previous spike step, ISI count, ISI sum, and squared-ISI sum in State. A CV is valid only with at least two complete intervals.
- Convert the final online accumulators to per-neuron ISI CV on the host. Do not pool intervals across neurons.
- Preserve raw post-transient rate/count series, sample raster events, valid CV values, and spectral arrays for every seed-condition run.

## Frozen spectrum and regime test

Compute Welch spectra from the demeaned 4 s post-transient whole-network rate at `fs = 10,000 Hz`, with a Hann window, `nperseg = 10,000`, `noverlap = 5,000`, constant detrending, and density scaling. Search `1-500 Hz`; never admit DC as a peak.

For each seed, define:

- dominant frequency: maximum PSD within `1-500 Hz`;
- background: median PSD in `1-500 Hz` excluding `+/-5 Hz` around that peak;
- peak prominence ratio: peak PSD divided by that background;
- narrowband fraction: PSD within `+/-2 Hz` of the peak divided by total PSD in `1-500 Hz`.

A seed has a significant narrowband peak only when prominence is at least `5` and narrowband fraction is at least `0.05`. A condition is:

- `synchronous` when all three seeds have significant peaks and their peak range is no more than `max(5 Hz, 0.2 * median peak frequency)`;
- `asynchronous` when none of the three seeds has a significant peak;
- `synchrony-indeterminate` otherwise.

Pool valid per-neuron CV values across seeds only after computing every neuron's CV separately. Require at least 10% of all neurons per seed to have valid CV. Label firing `regular` for pooled median CV `<= 0.5`, `irregular` for `>= 0.8`, and `regularity-indeterminate` otherwise. For synchronous activity, label the median dominant frequency `slow` below `30 Hz` and `fast` at or above `30 Hz`.

Map predicates to names exactly:

| Regime | Required predicates |
|---|---|
| synchronous regular | synchronous + regular |
| fast synchronous irregular | synchronous + irregular + fast |
| asynchronous irregular | asynchronous + irregular |
| slow synchronous irregular | synchronous + irregular + slow |

Any other combination is reported literally or as inconclusive; requested names are never forced.

## Implementation design and checks

- Keep physical parameters as BrainUnit quantities until explicit NumPy/SciPy/Matplotlib boundaries.
- Store scalar fixed-fan-in weights rather than materializing one weight per edge.
- Keep connectivity indices in `int32`, delay history and sample spikes in boolean arrays, refractory State in the native neuron, and count/ISI accumulators in compact numeric State.
- Test exact indegrees, uniqueness, source bounds, no autapses, inhibitory sign, unit propagation through BrainEvent, delta summation, 15-step delay, external count statistics, State reset, deterministic replay, output shapes, and finite metrics.
- Save an immutable JSON configuration, environment/package record, log, per-run NPZ artifacts, metrics JSON/CSV, connectivity hashes, and an artifact manifest.

