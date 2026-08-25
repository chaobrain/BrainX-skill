# BrainX study record

## Selected scale and ownership

- Model only the point-neuron scale. BrainPy-State owns the LIF population and point-neuron lifecycle.
- BrainEvent owns binary recurrent communication and exact fixed fan-in storage.
- BrainUnit owns voltage, time, and frequency quantities through simulation boundaries.
- BrainState owns Module registration, mutable dynamical State, delay buffers, reproducible random State, and the compiled time loop.
- Active optional training or fitting coverage: none; this is forward simulation.

## Source model

Use Brunel model A exactly at the supplied Figure 8 points. The neuron equation is `tau_m dV/dt = -V` between events. An excitatory or external arrival adds `J`; an inhibitory arrival adds `-g J`. Threshold crossing emits a spike, reset follows the refractory interval convention, and inputs arriving during refractoriness are ignored. Use the locked parameter and analysis contract in `NeuroSpecification.md`.

## API and lifecycle decisions

| Need | BrainX API and invariant |
|---|---|
| Point-neuron component | Subclass `brainpy.state.Neuron` because the requested delta-voltage equation has no current conversion or synaptic filter. Register `V`, last-spike time, and current spike as BrainState runtime State. |
| Exact fixed indegree | Construct `brainevent.FixedNumPerPost(data, indices, shape=(N_pre, N_post))`; `indices` has shape `(N_post, indegree)`. Sample each row without replacement and exclude the corresponding source for within-population targets. |
| Event communication | Apply `brainevent.BinaryArray(delayed_spikes) @ connectivity`; the output has one recurrent spike count per postsynaptic neuron. Store each repeat's explicit topology on its Module before tracing because BrainEvent treats fixed-degree indices as static metadata. Multiply E counts by `J` and I counts by `-g J` after communication so one graph-specific compiled runner is reusable across all four `g` values. |
| Recurrent delay | Use `brainstate.nn.Delay` with a boolean target descriptor and `D = 1.5 ms`. Retrieve the 15-step tap before the neuronal update and insert emitted spikes afterward, following the canonical delayed `DeltaProj` example. |
| External Poisson drive | Use `brainpy.state.poisson_generator(in_size=N, rate=C_ext * eta * nu_thr, rng_seed=...)`. It emits independent per-neuron Poisson multiplicities each step. Apply the same 15-step delay before converting counts to voltage jumps. |
| State initialization | Enter `brainstate.environ.context(dt=DT)` before construction and initialization. Call `brainstate.nn.init_all_states(net)` once before tracing, then use `brainstate.nn.reset_all_states(net)`, restore the intentional seeded voltage, and reseed the external RNG before every independent run. Never replace a compiled callable's State objects or carry voltage, refractory, spike, delay, or RNG values across runs. |
| Stateful rollout | Put the complete step in `brainstate.transform.for_loop` and compile the complete rollout with `brainstate.transform.jit`. Do not use a Python timestep loop. |
| Unit boundary | Keep `V`, `J`, thresholds, reset, `dt`, delay, refractory time, and rates as BrainUnit quantities. Convert to milliseconds, seconds, millivolts, or hertz only for host analysis and serialization. |

## Update order

For time step `n`:

1. Draw the current external Poisson multiplicity and insert it into its delay buffer.
2. Insert the spike from the previous completed neuronal update into the recurrent delay buffer.
3. Retrieve both 15-step taps; insert-before-retrieve makes step 0 current and step 15 exactly 15 completed updates earlier.
4. Apply E and I event communication and form the signed voltage jump.
5. Advance leak, apply the complete instantaneous jump only outside refractoriness, threshold, reset, and record the new spike for insertion at the next step.
6. Return only the new E/I spikes; derive population counts inside the compiled loop when a reduced-output check does not require full rasters.

An impulse test must demonstrate that an event inserted at step 0 is retrieved at step 15 and nowhere earlier.

## Randomness and graph reuse

Use repeat seeds `[1729, 2718, 3141, 5772, 8119]`. Derive graph, initial-voltage, and external seeds deterministically from each repeat seed. Hold the graph and initial-voltage realization fixed across all four conditions within one repeat. Reset the external generator to the same seed sequence for each condition; changing `eta` changes only its Poisson intensity. Replaying one complete seed and condition must reproduce spikes and metrics exactly.

## Metrics fixed before production

- Retain full spikes only long enough to calculate per-neuron statistics; save fixed 50-E/50-I raster probes and E/I population counts for every repeat.
- Compute E and I firing rates from analyzed spike counts divided by population size and 2 s.
- Compute each neuron's ISI CV when it has at least four spikes (three intervals), then report the mean and median separately for E and I. Apply the locked regularity predicate to the mean across all eligible E and I neurons.
- Form the 1 ms rate from ten 0.1 ms bins and compute its temporal CV.
- Run `scipy.signal.welch` on the mean-centered 0.1 ms population rate with a Hann window, `nperseg=8192`, 50% overlap, constant detrending, and density scaling. Search non-DC frequencies from 1 to 1,000 Hz. Define background power as the median in that band outside `dominant_frequency +/- 5 Hz`.
- Apply every classification predicate from the locked specification without changing thresholds or bands. Use the median of each continuous metric for the aggregate five-repeat predicate and require at least four matching repeat labels.

## Focused implementation checks

- Unit and external-rate conversion: `nu_thr = 10 Hz`; aggregate rates are 20, 40, 20, and 9 kHz per neuron.
- Exact E/I matrix shapes, indegrees, unique row indices, in-range indices, no autapses, and signed jump construction.
- Delay impulse timing at 15 steps.
- LIF leak, threshold, reset, and refractory insensitivity to jumps.
- Independent reset and deterministic replay.
- Small-network full-spike eager/compiled parity.
- No-recurrence external-drive sanity against a single-neuron mean-input expectation without using that control as a regime claim.

## References studied

- BrainX general guard and modeling-loop contracts.
- BrainPy-State root skill, component selection, projection patterns, delay protocol, and `109_fast_global_oscillation.py`.
- BrainEvent root skill, connectivity variants, `coba_ei_teaching.py`, and the delayed-communication portion of `204_joglekar_2018_propagation.py`.
- BrainUnit root skill.
- BrainState root skill, simulation environment, and randomness/reproducibility references.
- Brunel (2000), model definition, Figure 7, Figure 8, Table 1, and finite-size discussion.
