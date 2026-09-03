# BrainX iteration 2 study record

## Refusal finding map

| Finding | Restudied owner | Iteration-2 decision |
|---|---|---|
| `BX-API-001` | BrainTools connectivity and metric references | Keep only two explicit host boundaries and add `BrainToolsAPIGap.md` with checked APIs and parity. The frozen exact fan-in/no-autapse rule is not represented by a listed BrainTools pattern, and the frozen Welch window/detrend/scaling contract is not configurable in `power_spectral_density`. |
| `BX-NATIVE-002` | BrainPy-State delay protocol and BrainState collective lifecycle | Replace `RingDelay` with `brainstate.nn.Delay`; initialize/reset it through collective lifecycle operations and prove exact 15-step arrival plus full smoke-rollout parity. |
| `BX-CTRL-003` | BrainState randomness, BrainPy-State `DeltaProj`/`LIFRef` | Add fixed-seed Poisson moment checks, signed E/I delta summation through projections, and a hand-computed transition check spanning leak, delay delivery, threshold, hard reset, and refractory exclusion. |
| `BX-STATE-004` | BrainState State lifecycle and run evidence | Save every run's complete final membrane vector, SHA-256, range, and finiteness result; make collection reject missing, mismatched, or non-finite State evidence. |

## Native delay lifecycle

Use `brainstate.nn.Delay(target_info, max_time=1.5 ms)` with a boolean vector target shape. At the start of simulation step `i`, `neurons.get_spike()` is the event emitted by the completed step `i - 1`. Insert it with `delay.update(previous_spike)`, then retrieve `delay.retrieve_at_step(14)`. The retrieved event is `s[i - 15]`: the buffer's step 0 is `s[i - 1]`, so step 14 is exactly 15 simulation steps behind the current target update. This is an insertion-time convention, not an unexplained shortening of the 1.5 ms delay.

Require two checks:

1. An impulse emitted at source step 0 appears in the target input at step 15 and nowhere earlier.
2. With identical fixed random structures and external stream, every E count, I count, and sampled spike in the iteration-2 smoke rollout exactly matches the preserved iteration-1 ring-buffer smoke output.

## Control design

- Draw a fixed 250,000-sample BrainState Poisson snapshot for each production lambda and require sample mean and variance within five analytic standard errors of lambda. Preserve the observed values so this is reproducible rather than an unfrozen stochastic test.
- Construct a hand-checkable three-source/one-target projection. Deliver two `+0.1 mV` excitatory events and one `-0.3 mV` inhibitory event after leak from `10 mV`; require the native update to equal `10 * exp(-0.1 / 20) - 0.1 mV`.
- Extend the transition check with a delayed suprathreshold jump, require hard reset to `10 mV`, and continue forcing input during the next 2 ms while requiring no spike before the refractory boundary.

## Production evidence change

After each rollout and synchronization, convert the complete final `neurons.V` quantity explicitly to mV at the host boundary. Reject non-finite values immediately. Save `final_voltage_mV` in the raw NPZ and save its SHA-256, minimum, maximum, and `all_finite = true` in the per-run metric JSON. The combined collector recomputes and matches the hash and rejects any invalid vector.

## Acceptance for re-execution

Do not reuse iteration-1 scientific results as iteration-2 evidence even if smoke parity is exact. Freeze new code snapshots and re-execute all twelve production rollouts because the required final State evidence cannot be recovered from completed iteration-1 processes.
