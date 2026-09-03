# BrainX acceleration audit

| File/area | Pattern | Evidence | Axis | Impact | Risk | Rewrite | Confidence |
|---|---|---|---|---|---|---|---|
| `CurrentLIFNetwork.update` | already fused | Population-shaped LIF State, BrainEvent products, Poisson draw, ISI reductions, and delay update execute inside one transformed step. | N | High | Low | Keep unchanged. | High |
| `create_rollout` | correct JIT boundary | All 50,000 recurrent steps run through `brainstate.transform.for_loop` enclosed by one `brainstate.transform.jit`. | T | High | Low | Keep unchanged. | High |
| `run_experiment` condition/seed loop | `loop-E-sweep` | Twelve independent rollouts use a small host loop; each seed reuses one connectivity realization across four conditions. | E | Medium | High | Do not `vmap`: mapped dynamical State and three seed-specific fixed-fan-in matrices would substantially increase peak memory, while CPU exposes no parallel device. | High |
| `_fixed_sources` | `loop-N-scalar` outside hot path | One host loop constructs each target's exact unique sources once per seed before compilation. | N | Low runtime frequency | Medium | Keep NumPy host construction; a transformed replacement would require large random ranking arrays and would not accelerate simulation. | High |
| returned history | bounded output history | Rollout returns two count vectors and 50 sampled spike indicators, not all 12,500 spikes or voltages. | T x N | High memory benefit | Low | Keep online ISI moments and sampled raster output. | High |
| connectivity | structured event representation | `FixedNumPerPost` stores exact fan-in indices and one scalar weight per pathway; it never forms a dense `12,500 x 12,500` matrix. | N | High | Low | Keep structured representation. | High |
| stochastic drive | State-aware RNG | One `RandomState` advances inside the compiled loop and is reseeded at independent rollout boundaries. | T | High | Medium | Keep and verify exact replay. | High |

## Patch / rewrite plan

No production hot-path rewrite is justified. The implementation already applies the acceleration skill's highest-impact transformations and memory reductions. The BrainState control-flow and randomness references establish the existing `jit(for_loop)` and reset/reseed boundaries. Preserve the current code and add only a benchmark artifact that separates compilation from warm execution and checks exact output/final-voltage parity after reset.

## Validation plan

- Construct a representative `800 E + 200 I`, `80 E + 20 I` fixed-fan-in network for 200 ms.
- Time the first compiled rollout separately from a reset-and-replayed warm rollout.
- Synchronize before stopping each timer.
- Require exact equality for E counts, I counts, sampled spikes, and final voltage after State/RNG reset.
- Change `(g, eta)` without changing shapes and time one additional warm call to demonstrate reuse of the compiled graph.
- Retain boolean delay history, `int32` connectivity indices, scalar pathway weights, and online ISI moments; do not materialize dense connectivity or full spike history.

## Remaining risks

- Production runtime is dominated by 15.625 million logical fixed-fan-in edges over 50,000 steps on one CPU device; the representative benchmark can estimate but not eliminate that cost.
- BrainEvent CPU performance depends on event sparsity and condition, so warm runtime can differ substantially across regimes.
- The condition loop is intentionally serial to cap memory and preserve simple, separately reset RNG and dynamical State.

## Measured outcome

- Representative `800 E + 200 I`, 2,000-step benchmark: `10.450 s` cold, `1.846 s` warm, and exact equality of all outputs and final voltage after reset.
- Full-scale 200-step benchmark: `27.578 s` cold and `4.714 s` warm, projecting `1,178.6 s` per 5-second rollout and `14,142.6 s` for twelve serial runs.
- Equivalent CSR event products preserved counts exactly but ranged from `0.93x` to `1.39x` the `FixedNumPerPost` speed across tested spike densities; no consistent material gain justified the extra conversion and storage.
- Final acceleration decision: keep the scientific implementation unchanged and execute the three independent seeds concurrently on disjoint CPU core sets. Keep the four conditions within each seed serial so they reuse one compiled graph and one shared connectivity realization.
