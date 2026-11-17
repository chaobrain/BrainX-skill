# Select a verified NEST delay path

In BrainSpikes 0.2.0, delay support differs between the default static projection, explicit synapse specifications, and mutable connection setters. Verify actual transmission before replacing one path with another.

## Choose the construction path

| API | Verified use or limit |
|---|---|
| `sim.connect(pre, post, weight=..., delay=array)` | The default static path accepts the tested one-source/two-target vector and delivers at distinct delays. |
| Separate scalar-delay `connect()` calls | The tested two-by-two source/target delay matrix works when expressed as four scalar connections. |
| Matrix delay in one `connect()` | The tested ordinary-neuron two-by-two matrix changes target State shape and fails with a scan carry-type error. Do not assume arbitrary array shapes are supported. |
| `static_synapse(delay=array)` / `cont_delay_synapse(delay=array)` | Both tested explicit-spec array paths fail scalar conversion. |
| `get_connections().set("delay", ...)` on the tested static-synapse spec | Rejected because the underlying spec projection fixes delay at construction. This does not establish every setter path's behavior. |

Open `scripts/nest_boundary_workflows.py` for `delay_arrivals` and `neuron_delay_matrix`. The two-by-two grouping fixture preserves the specified edge pairs and delays; it is a correctness fallback, not a scalable implementation of arbitrary large delay matrices.

## Verify fractional transmission with continuous traces

The current-based `iaf_psc_exp` fixture uses one source spike, a `0.1 ms` integration step, and a `250 pA` connection. For delay `0.24 ms`, both the default path and `cont_delay_synapse` produce the weighted trajectory `0.6*V(0.2 ms)+0.4*V(0.3 ms)`, with maximum error about `4.4e-16 mV` against default-path endpoints.

This verifies a grid-level fractional-amplitude interpolation for that continuous observable. It is not an exact continuously timed binary event. A high-amplitude delta target can report only one later grid spike while the underlying drive spans two bins; spike time alone is insufficient to establish fractional behavior.

**Release-scoped anomaly:** with the same source/current target at delay `0.3 ms`, the explicit `static_synapse` and `cont_delay_synapse` traces differ markedly from the default static connection. The observed maximum discrepancies are approximately `1.331` and `1.323 mV`; the explicit-spec peak is about twice the default peak. Do not treat all scalar spec delays as interchangeable or assume this is only timestamp quantization. Use the default path for the verified fixture and require an impulse/amplitude check at the intended delay before choosing another spec. No package implementation was modified to hide this behavior.

Open `scripts/nest_boundary_workflows.py` for `voltage_trace` and the executable counterexample. The pinned regression records these failures as boundaries, not as capabilities that should be generalized to every later release.

Open `references/native-delays.md` for integer native buffer integration, and `references/nest-compatible/divergence-and-parity.md` before claiming parity with live NEST or its precise-spiking models.

## Source

[Continuous-delay synapse](https://brainx.chaobrain.com/brainspike/apis/generated/brainspike.cont_delay_synapse.html). This live page describes the intended delay scheme; the local release-scoped experiments establish the narrower tested behavior and counterexample above.
