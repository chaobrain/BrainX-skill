# Integration categories

Use this reference when determining how a NEST-compatible model family is numerically updated and which parity reference to open next. Integration categories describe how a model is solved; validation categories describe how its output is compared with NEST.

## Category map

| Integration category | Numerical behavior | Model families | Downstream decision |
|---|---|---|---|
| A | Adaptive Runge-Kutta-Fehlberg (`AdaptiveRungeKuttaStep`) controls local error with internal adaptive steps. | `aeif_*`, `gif_*`, `glif_*`, `iaf_cond_*` | Expect adaptive-integrator trace validation; open `divergence-and-parity.md` for category-A tolerances. |
| B | Exact analytic propagation over the fixed simulation step; no Runge-Kutta iteration. | `iaf_psc_*`, including alpha, exponential, and delta forms | Expect near-exact deterministic trace parity under matching inputs and `dt`. |
| C | Adaptive Runge-Kutta-Fehlberg handles Hodgkin-Huxley gating dynamics. | `hh_psc_*`, `hh_cond_*`, `ht_neuron` | Use the documented HH/conductance validation band rather than category-B exactness. |
| D | A vectorized population update advances rate State each step without a spike-resolved ODE. | `lin_rate_*`, `tanh_rate_*`, `sigmoid_rate_*`, `threshold_lin_rate_*`, `siegert_neuron` | Validate the relevant rate or distributional observable rather than a spike-resolved voltage trace. |
| E | Discrete update rules run without ODE integration. | Generators, recorders, detectors, static synapses, gap junctions, STP, STDP, voltage-based learning rules | Validate event time/count, distribution, or documented plasticity behavior according to the component. |

## Apply the category

1. Identify the concrete model or component family.
2. Use the table to determine the numerical update mechanism.
3. Keep the simulation `dt` and device alignment identical when constructing a NEST comparison.
4. Open `divergence-and-parity.md` and choose the validation mode and tolerance for the actual observable.

Examples:

| Model | Integration result | Validation consequence |
|---|---|---|
| `iaf_psc_alpha` | Category B analytic propagator | Compare a deterministic trace near-exactly under category B. |
| `aeif_cond_exp` | Category A adaptive RKF45 | Use the adaptive-integrator trace tolerance. |
| `hh_cond_exp_traub` | Category C Hodgkin-Huxley RKF45 | Use the documented HH/conductance tolerance. |
| `lin_rate_ipn` | Category D vectorized rate update | Compare rate behavior; use distributional parity when the drive is stochastic. |
| `stdp_synapse` | Category E event-driven rule | Compare at the documented event lifecycle point and check STDP divergences. |

## Common failures

- Do not infer numerical behavior from `psc` versus `cond` alone without checking the family table.
- Do not call integration category D "distributional validation category D"; the names align by design but answer different questions.
- Do not assign one tolerance to every model in an integration family; the observable and stochasticity still determine the validation mode.
- Do not demand bit-for-bit agreement from an adaptive, stochastic, or event-aligned workflow when the parity reference specifies a trace band, distribution, or event tolerance.

## Official source

- https://brainx.chaobrain.com/brainpy-state/nest-style/integration-categories.html
