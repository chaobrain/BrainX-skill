# BrainX study record

## Selected scale and ownership

- Represent only the point-neuron network scale. Use BrainPy-State for the LIF population and neuronal lifecycle; do not introduce neural-mass State.
- Use BrainEvent for binary recurrent communication and exact fixed-fan-in storage.
- Use BrainUnit for voltage, time, and frequency through the simulation boundary.
- Use BrainState for Module registration, dynamical and random State, delay buffers, environments, reset, and the compiled time loop.
- Use the BrainX acceleration workflow for the large population and time axes.
- Active training or fitting coverage: none; this is forward simulation.

## Source model and units

Use the locked Brunel model-A contract in `NeuroSpecification.md`. Between events, integrate `tau_m dV/dt = -V` exactly over one fixed step. Apply the complete E, I, and external delta-voltage jump outside refractoriness. Emit a spike at `V >= theta`, set `V=V_reset`, and ignore leak and jumps until the absolute refractory interval has elapsed.

Keep `tau_m`, `tau_ref`, delay, `dt`, voltage, and rates as BrainUnit quantities. Convert to raw milliseconds, seconds, millivolts, or hertz only at host-side analysis and serialization boundaries. The supplied equation gives `nu_threshold = 10 Hz` per external afferent and aggregate per-neuron Poisson means `eta` per 0.1 ms step.

## API and lifecycle design

| Need | BrainX API and invariant |
|---|---|
| Point-neuron dynamics | Subclass `brainpy.state.Neuron`; register `V`, last-spike time, and current spike as BrainState runtime State. Use an exact exponential leak because the drift is linear. |
| Exact fixed indegree | Use `brainevent.FixedNumPerPost(data, indices, shape=(N_pre, N_post))`; store source indices as `(N_post, indegree)`, sampled without replacement, and remove within-population autapses. |
| Event communication | Apply `brainevent.BinaryArray(delayed_spikes) @ connectivity`; its result is one recurrent input count per postsynaptic neuron. Keep topology attached to the Module before tracing. |
| Delta jumps | Multiply E counts by `JE` and I counts by `-g*JE`, add delayed external multiplicities times `JE`, then pass one voltage quantity to the neuron. No synaptic filter or current conversion belongs in this model. |
| Delay | Use `brainstate.nn.Delay` with boolean recurrent and integer external targets. Insert current values before `retrieve_at_step(15)`, so step 15 is exactly 1.5 ms old. Initialize and reset buffers with the network. |
| External drive | Use an independent `brainstate.random.RandomState` or documented clone per network; draw Poisson multiplicities of shape `(12500,)` with mean `eta`, and reseed before every independent condition. |
| Initialization | Enter `brainstate.environ.context(dt=0.1*u.ms, precision=32)` before construction. Initialize all State, reset all State before a condition, restore one intentional uniform `[reset, threshold)` voltage realization, and reseed external randomness. |
| Rollout | Put every timestep in `brainstate.transform.for_loop` and wrap the complete fixed-length rollout in `brainstate.transform.jit`. Keep condition and seed orchestration as a small host loop because each rollout must reset writable State and releases a 312.5 MB spike history before the next condition. |
| Analysis | Use NumPy and SciPy only after device-to-host transfer. Compute full-population ISI CV before discarding the full spike array; save only fixed 50-neuron rasters, E/I count traces, spectra, metrics, and graph hashes. |

## Update order

For step `n`:

1. Draw one independent external Poisson multiplicity per neuron and insert it into the external delay.
2. Insert the spike vector from the previous completed neuronal update into the recurrent delay.
3. Retrieve both exact 15-step taps.
4. Communicate delayed E and I spikes through their fixed-fan-in matrices and construct the signed delta-voltage input.
5. Advance leak and apply the jump only outside refractoriness, threshold, reset, and publish the new spike vector.
6. Return the new spikes. Reduce, analyze, serialize, and plot only outside the compiled stateful path.

Lock the insert-before-retrieve convention with an impulse test that is false before step 15, true at step 15, and false afterward.

## Randomness and matched conditions

Use repeat seeds `[1729, 2718, 3141, 5772, 8119]`. Derive topology, initial-voltage, and external seeds by fixed offsets. Within a repeat, hold topology and initial voltage fixed across all four conditions and restart the external stream from the same seed; only `(g, eta)` changes. Use sample seed `8675309` once to choose 40 E and 10 I probe neurons, then preserve those 50 indices across every condition and repeat.

## Analysis and classification

- Analyze only the final 2,000 ms.
- Compute E, I, and global 0.1 ms rate traces from spike counts and population sizes.
- Compute E/I mean firing rates and per-neuron ISI CV for neurons with at least four analyzed spikes; preserve eligible counts and distributions.
- Form 1 ms global rate by summing ten native bins and compute its temporal CV.
- Estimate global-rate power with `scipy.signal.welch`: sampling rate 10 kHz, Hann window, `nperseg=8192`, 50% overlap, constant detrend, density scaling.
- Search 1-1000 Hz for the dominant bin. Define spectral background as the median outside `dominant +/- 5 Hz` within the search band.
- Apply the locked predicates without post-outcome edits. Preserve every repeat label, aggregate median metrics, matching-repeat count, and robust/non-robust outcome.

## Focused checks before production

- Unit and drive conversion: `nu_threshold=10 Hz`; aggregate external rates are 20, 40, 20, and 9 kHz per neuron, with per-step means 2, 4, 2, and 0.9.
- Connectivity shapes, exact indegrees, uniqueness, range, absence of autapses, and signed jump construction.
- Delay impulse at exactly 15 steps.
- Leak, threshold/reset, and refractory jump insensitivity.
- Independent reset and deterministic replay.
- Small-network eager/compiled parity including final voltage.
- No-recurrence external-drive activity and finite State.
- Locked configuration round-trip and fixed 50-neuron sample composition.

## References studied

- BrainX modeling loop and general guard.
- BrainPy-State root skill, projection patterns, delay protocol, `109_fast_global_oscillation.py`, and the NEST-compatible `brunel_delta.py` application.
- BrainEvent root skill, connectivity variants, and `coba_ei_teaching.py`.
- BrainUnit root skill.
- BrainState root skill, simulation environment, randomness/reproducibility, and control-flow patterns.
- BrainX acceleration root skill.
- BrainX run and monitor experiment references.
- The matching committed case-17 specification, study record, implementation, tests, and acceleration evidence, inspected read-only while its worktree deletions remained untouched.
