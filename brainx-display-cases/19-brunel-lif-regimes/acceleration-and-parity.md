# BrainX acceleration audit

| File/area | Pattern | Evidence | Axis | Impact | Risk | Rewrite | Confidence |
|---|---|---|---|---|---|---|---|
| `SparseEINetwork.update` | `already-fused` | One vectorized LIF transition and two `BinaryArray @ FixedNumPerPost` products; no scalar neuron or synapse loop | N | High | Low | Keep unchanged | High |
| `build_runner` | Correct stable JIT boundary | The complete fixed-length `for_loop` is inside one `brainstate.transform.jit`; no Python timestep loop or transform construction occurs during execution | T | High | Low | Keep unchanged | High |
| Condition loop | Small host loop over independent rollouts | Every condition resets voltage, refractory, spike, two delays, and external RNG State; each full output is 312.5 MB | E | Medium | High | Keep sequential and release each full history after analysis | High |
| Repeat loop | Graph-specific construction and compilation | Each repeat owns a separately sampled exact-indegree topology stored as BrainEvent static metadata | E | Medium | Medium | Accept one cold compile per graph; generated JITC connectivity would violate exact indegree and edge inspection | High |
| Full spike output | `memory-T-full-history` required | Full analyzed spike timing is needed for per-neuron ISI CV, while compact probes/counts alone are insufficient | T,N | Medium | Medium | Retain once per condition, reduce immediately, and delete the host array | High |
| Topology and analyses | Host-only NumPy/SciPy | Unique fixed-fan-in sampling, ISI grouping, Welch analysis, CSV/JSON/NPZ, and hashing occur outside transforms | N,E | Low | Low | Keep as explicit host boundaries | High |

## Patch / rewrite plan

No further acceleration rewrite is justified. Preserve population State, exact fixed-degree event communication, and one state-aware compiled time loop. Reuse the graph-specific compiled runner while passing scalar `g` and `eta`; reset every writable State before each call.

The exact semantics came from the BrainState control-flow reference and the BrainPy-State delay protocol. `vmap` is intentionally not used: mapping five graphs or four conditions would duplicate independent delay buffers, random streams, neuronal State, and full spike histories on a 15 GiB CPU host.

## Validation plan

- Focused suite: `9 passed in 18.39s` on CPU, including delay timing, reset/replay, eager/compiled spikes, and final-voltage parity.
- Representative benchmark: 800 E plus 200 I neurons, exact indegrees 80/20, 100 ms total, seed 1729, `(g, eta)=(5,2)`.
- Cold run: 9.5112 s, including trace and compilation.
- Warm run after complete reset/reseed: 0.8086 s.
- Output shape: `(1000, 1000)` with 8,625 spikes.
- Parity: spike histories bit-identical and final membrane-voltage maximum absolute difference `0.0 mV`.
- Memory: no dense recurrent matrix. Production fixed-degree int32 indices use about 62.5 MB; one production boolean spike history uses 312.5 MB and is released condition by condition.

## Remaining risks

- Production is CPU-only and requires five graph-specific compilations.
- Full spike history is the largest result buffer; replacing it with online ISI moments would add State and analysis risk without a demonstrated memory failure.
- BrainEvent runtime depends on active-spike density, so conditions can have materially different warm runtimes.
- Host fixed-fan-in sampling is sequential but occurs once per graph outside the transformed simulation.

## Iteration 2 parity

The production hot path did not change. Repeating the representative benchmark after the smoke-only correction produced bit-identical spikes, `0.0 mV` final-voltage difference, a 15.9942 s cold run, and a 1.7014 s warm run on the same CPU backend. Timing variability does not affect the unchanged acceleration decision.

## Iteration 3 continuation parity

Condition-boundary continuation changes host orchestration only. The repeated representative simulation remains bit-identical after reset with `0.0 mV` final-voltage difference; measured cold and warm times were 27.3126 s and 3.5460 s under concurrent test load. The actual stopped parent passed the resume gate for all 13 completed artifacts. The remaining seven conditions retain their original full-history execution and are not mapped or shortened.

## Iteration 4 provenance and validation parity

The iteration-4 patch changes only host-side inheritance validation and process provenance. The representative simulation again produced 8,625 bit-identical spikes with a `0.0 mV` final-voltage difference. Cold and warm CPU times were 10.0524 s and 0.9344 s. All 20 completed source artifacts pass the stricter source-manifest and scientific-contract gate, so no simulation-axis rewrite or new mapping is justified.
