# BrainX acceleration audit

| File/area | Pattern | Evidence | Axis | Impact | Risk | Rewrite | Confidence |
|---|---|---|---|---|---|---|---|
| `SparseEINetwork.update` | Population/event path already fused | One vectorized LIF update and two `BinaryArray @ FixedNumPerPost` products; no scalar neuron or synapse loop | N | High | Low | Keep unchanged | High |
| `build_runner` | Correct largest stable JIT boundary | The complete fixed-length `for_loop` is inside one `brainstate.transform.jit`; no Python timestep loop or transform construction occurs inside execution | T | High | Low | Keep unchanged | High |
| Per-condition execution | Small host loop around independent compiled rollouts | Four conditions reuse one graph-specific compiled runner, but each needs independent writable voltage, refractory, delay, and RNG State | E | Medium | High | Keep sequential to avoid four full State/spike histories and uncertain fixed-degree batch kernels | High |
| Per-repeat execution | Graph-specific compilation | `FixedNumPerPost` indices are static topology metadata, so each independently sampled exact-indegree graph requires a fresh Module trace | E | Medium | Medium | Accept five cold compilations; JITC would change exact indegree and dynamic topology arguments fail the BrainEvent PyTree contract | High |
| Full spike output | `memory-T-full-history` required by analysis | Production output is 25,000 x 12,500 booleans (312.5 MB) before host analysis; exact per-neuron ISI CV requires spike timing | T,N | Medium | Medium | Retain for correctness, immediately reduce to raster probes/counts/CV and discard after each condition | High |
| Host topology and metrics | Host-only Python/NumPy | Fixed-indegree sampling, ISI grouping, Welch spectra, serialization, and condition/repeat loops occur outside the transformed rollout | N,E | Low | Low | Keep as explicit host boundary | High |

## Patch / rewrite plan

No acceleration rewrite is justified after the audit. The implementation already uses the BrainX-native optimum for this workload: population State, exact fixed-degree event communication, and one state-aware compiled time loop. Preserve one graph-specific compiled runner per repeat and pass only scalar `g` and `eta` dynamically across its four independent condition calls.

The fixed-degree topology cannot be passed dynamically through the installed BrainEvent/BrainState JIT cache because its indices are static PyTree metadata. Attaching topology to the Module before tracing is therefore an execution contract, not cosmetic specialization.

## Validation plan and result

- Focused parity suite: `8 passed in 23.12s` on CPU, including exact eager/compiled output parity, deterministic compiled replay, delay timing, and State reset behavior.
- Representative benchmark: 800 E + 200 I neurons, indegrees 80/20, 100 ms total, fixed seed 1729, `(g, eta) = (5, 2)`.
- Cold execution: 12.3306 s, including trace/compile.
- Warm execution after complete reset/reseed: 0.9256 s.
- Output parity: spike histories bit-identical; final membrane voltage maximum absolute difference `0.0 mV`; output shape `(1000, 1000)` with 8,625 spikes.
- Memory: no dense recurrent matrix is created. Production fixed-degree indices require about 62.5 MB as int32; one full boolean spike history requires 312.5 MB and is released condition by condition.

## Remaining risks

- Production is CPU-only and five graph-specific compilations are required; total runtime may be substantial.
- Full spike history is the dominant result buffer. Online ISI moments could reduce memory, but that rewrite would add State and analysis complexity without evidence that memory is currently limiting.
- Mapping conditions or repeats would multiply dynamical State, RNG, delay, and output memory. It is intentionally rejected on the single CPU device.
- BrainEvent kernel performance depends on active-spike density; the synchronous-regular state may be the most expensive condition.
