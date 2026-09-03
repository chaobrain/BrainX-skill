# BrainX acceleration audit, iteration 2

| File/area | Pattern | Evidence | Axis | Impact | Risk | Rewrite | Confidence |
|---|---|---|---|---|---|---|---|
| Native delay | BrainX-owned State | `brainstate.nn.Delay` replaces manual buffer/pointer State and exactly preserves all iteration-1 smoke arrays. | T x N | Medium | Low | Keep native Delay. | High |
| Complete rollout | correct JIT boundary | One `brainstate.transform.jit` encloses one `for_loop` over all timesteps; no Python timestep loop. | T | High | Low | Keep unchanged. | High |
| Communication | structured events | Exact target-major fixed fan-in remains `FixedNumPerPost`; prior CSR benchmark showed no consistent gain. | N | High | Low | Keep unchanged. | High |
| Histories | summary/sampled only | Online ISI moments plus E/I counts and 50 sampled neurons avoid full population history. | T x N | High | Low | Keep unchanged. | High |
| Three seeds | independent host processes | Disjoint immutable State, RNG, connectivity, and output paths support concurrent CPU execution. | E | Medium | Low | Retain three disjoint core sets. | High |

## Validation

- Representative `800 E + 200 I`, 2,000-step result: `15.436 s` cold, `1.541 s` warm replay, and `1.403 s` for a changed condition on the same compiled graph.
- Exact equality holds for E counts, I counts, sampled spikes, and final membrane voltage after reset/reseed.
- Full-scale 200-step result: `41.173 s` cold and `5.661 s` warm, projecting `1,415.2 s` per 5-second condition and `16,981.8 s` for twelve serial conditions.
- Native-delay smoke output is exactly equal to the iteration-1 ring result for every preserved raw array.

## Decision and remaining risk

Keep the corrected implementation unchanged. The native Delay increases the full-scale short benchmark relative to iteration 1, but replacing it would violate the reviewed BrainX-native requirement. Production remains CPU-bound and condition runtime remains spike-density dependent. Run three independent seeds concurrently on the already validated disjoint CPU core sets; keep four conditions serial within each seed to reuse connectivity and compilation while bounding memory.

