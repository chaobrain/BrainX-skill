# BrainX study record

## Scale and package ownership

- Represent one point-neuron E/I spiking network; use no ion/channel, compartment, morphology, or neural-mass mechanism.
- Use `brainpy.state.Neuron` as the owner of the exact Brunel point-neuron dynamics.
- Use BrainEvent only for binary event communication, BrainState for mutable State, delay, RNG, environments, JIT, and time control flow, and BrainUnit for voltage/time/frequency quantities.
- Required packages import successfully in the selected Python environment.
- Training/fitting coverage: none.

## Sources studied

- Brunel (2000), Eqs. 1-2, Model A, Section 6, Fig. 8, and Table 1.
- BrainX general guard and modeling loop.
- BrainPy-State root skill, projection patterns, connectivity guidance, and delayed delta-network example.
- BrainEvent root skill, fixed-degree variants, and E/I event-communication example.
- BrainState root skill, randomness/replay, collective lifecycle, control flow, and environment references.
- BrainUnit root skill.

## Model translation

Between events, apply the exact discrete leak `V <- V exp(-dt/tau_m)`. Add delayed recurrent jumps `J*nE - gJ*nI` and the current aggregate external jump `J*nExt`. Hold voltage fixed during the absolute refractory interval. On threshold, emit a Boolean spike, reset to `Vr`, and record the spike time.

Sample exactly 1,000 unique E sources and 250 unique I sources per target, excluding same-population autapses. Store source indices with `brainevent.FixedNumPerPost` and communicate through `BinaryArray @ connectivity`. Use one inspectable graph for all four parameter points and reset every dynamical/RNG State between them.

## API lifecycle

| Need | API and invariant |
|---|---|
| Point neuron | Subclass `brainpy.state.Neuron`; allocate `HiddenState` voltage and `ShortTermState` last-spike/current-spike values in `init_state()`. |
| Sparse fan-in | Use two `brainevent.FixedNumPerPost` objects shaped `(NE,N)` and `(NI,N)` with source-index arrays shaped `(N,CE)` and `(N,CI)`. |
| Recurrent delay | Use one `brainstate.nn.Delay` and the canonical retrieve-before-neuron, push-after-neuron protocol; verify 1.5 ms physical elapsed time. |
| External randomness | Use one independent BrainState random stream, reseeded per condition; a Poisson count of mean `eta` represents the sum of 1,000 external afferents. |
| Execution | Scope `dt`, `t`, and `i` in BrainState environments; put the complete fixed-length State-driven sequence in `brainstate.transform.for_loop` under one `brainstate.transform.jit`. |
| Reset | Initialize once, use collective reset between conditions, replace voltage with the seeded initial distribution, and reseed external RNG. |
| Host boundary | Convert the completed spike history only for fixed analyses, compact raw serialization, hashing, and eventual plotting. |

## Validation and artifacts

Test units/rates, topology, delay, refractory/reset, deterministic replay, eager/JIT parity, external-only activity, acceptance logic, and fixed probes. Benchmark cold and warm execution with final-State parity before production. Save raw arrays and metrics without rendering. Bind the implementation, specification, command, run contract, environment, and results through one outer artifact manifest before review.

Only after a fresh read-only Codex review passes may the renderer read accepted raw files and produce Fig. 8.
