# Model library

Use this reference when selecting a NEST-compatible neuron family, locating an exact `brainpy.state` model name, or recognizing the model-specific capability that changes construction or routing. Use upstream NEST documentation for equations, parameter meanings, physiology, and scientific references.

All constructor rows use the canonical `Simulator.create()` surface. Pass a reusable NEST-style parameter mapping with `params=parameters`, or replace it with explicit keyword parameters.

## Current-based IAF neurons

Use these models for linear integrate-and-fire dynamics driven by postsynaptic currents; choose delta, alpha, exponential, precise-spiking, or multisynapse behavior from the requested NEST model.

| Constructor | Use when |
|---|---|
| `sim.create(bp.iaf_psc_delta, size, params=parameters)` | Use delta-shaped input that produces an instantaneous membrane-voltage jump. |
| `sim.create(bp.iaf_psc_delta_ps, size, params=parameters)` | Use delta input with precise spike timing between grid points. |
| `sim.create(bp.iaf_psc_alpha, size, params=parameters)` | Use alpha-shaped excitatory and inhibitory current kernels. |
| `sim.create(bp.iaf_psc_alpha_multisynapse, size, params=parameters)` | Use alpha currents with multiple receptor ports or time constants. |
| `sim.create(bp.iaf_psc_alpha_ps, size, params=parameters)` | Use alpha currents with precise spike timing. |
| `sim.create(bp.iaf_psc_exp, size, params=parameters)` | Use exponential postsynaptic-current kernels. |
| `sim.create(bp.iaf_psc_exp_multisynapse, size, params=parameters)` | Use exponential currents with multiple receptor ports or time constants. |
| `sim.create(bp.iaf_psc_exp_htum, size, params=parameters)` | Use the NEST `iaf_psc_exp_htum` variant required by the model being ported. |
| `sim.create(bp.iaf_psc_exp_ps, size, params=parameters)` | Use exponential currents with precise spike timing. |
| `sim.create(bp.iaf_psc_exp_ps_lossless, size, params=parameters)` | Use exponential precise-spiking dynamics with lossless spike detection. |

**Invariant:** Use voltage weights such as `mV` for `iaf_psc_delta`; alpha and exponential current-event examples use current weights such as `pA`.

## Conductance-based IAF neurons

Use these models when synaptic input changes conductance rather than adding a postsynaptic current.

| Constructor | Use when |
|---|---|
| `sim.create(bp.iaf_cond_alpha, size, params=parameters)` | Use alpha-shaped synaptic conductances. |
| `sim.create(bp.iaf_cond_alpha_mc, size, params=parameters)` | Use the multi-compartment alpha-conductance variant; check parity coverage before claiming NEST validation. |
| `sim.create(bp.iaf_cond_beta, size, params=parameters)` | Use beta-shaped synaptic conductances. |
| `sim.create(bp.iaf_cond_exp, size, params=parameters)` | Use exponential synaptic conductances. |
| `sim.create(bp.iaf_cond_exp_sfa_rr, size, params=parameters)` | Use exponential conductances with spike-frequency adaptation and relative refractory dynamics. |

## Specialized IAF neurons

Use these APIs only when the requested paper or source NEST model names the specialized variant.

| Constructor | Use when |
|---|---|
| `sim.create(bp.iaf_bw_2001, size, params=parameters)` | Use the NEST-compatible `iaf_bw_2001` model. |
| `sim.create(bp.iaf_bw_2001_exact, size, params=parameters)` | Use the exact per-synapse NMDA form of `iaf_bw_2001`. |
| `sim.create(bp.iaf_chs_2007, size, params=parameters)` | Use the `iaf_chs_2007` spike-response model. |
| `sim.create(bp.iaf_chxk_2008, size, params=parameters)` | Use alpha synapses with the `iaf_chxk_2008` precise AHP timing. |
| `sim.create(bp.iaf_tum_2000, size, params=parameters)` | Use the `iaf_tum_2000` neuron required by the port. |

## Adaptive exponential IF neurons

Use these APIs for Brette-Gerstner adaptive exponential dynamics, selecting current versus conductance input and any required astrocyte, multisynapse, or Clopath interface.

| Constructor | Use when |
|---|---|
| `sim.create(bp.aeif_cond_alpha, size, params=parameters)` | Use adaptive exponential dynamics with alpha-shaped conductances. |
| `sim.create(bp.aeif_cond_alpha_astro, size, params=parameters)` | Use the astrocyte-compatible alpha-conductance variant. |
| `sim.create(bp.aeif_cond_alpha_multisynapse, size, params=parameters)` | Use alpha conductances with multiple receptor ports. |
| `sim.create(bp.aeif_cond_beta_multisynapse, size, params=parameters)` | Use beta conductances with multiple receptor ports. |
| `sim.create(bp.aeif_cond_exp, size, params=parameters)` | Use adaptive exponential dynamics with exponential conductances. |
| `sim.create(bp.aeif_psc_alpha, size, params=parameters)` | Use adaptive exponential dynamics with alpha-shaped currents. |
| `sim.create(bp.aeif_psc_delta, size, params=parameters)` | Use adaptive exponential dynamics with delta input. |
| `sim.create(bp.aeif_psc_delta_clopath, size, params=parameters)` | Use delta input while exposing the voltage filters consumed by `clopath_synapse`. |
| `sim.create(bp.aeif_psc_exp, size, params=parameters)` | Use adaptive exponential dynamics with exponential currents. |

## Generalized IF neurons

Use these APIs for generalized integrate-and-fire dynamics, choosing current versus conductance input, population dynamics, or multiple synaptic time constants.

| Constructor | Use when |
|---|---|
| `sim.create(bp.gif_cond_exp, size, params=parameters)` | Use conductance-based GIF dynamics with exponential synapses. |
| `sim.create(bp.gif_cond_exp_multisynapse, size, params=parameters)` | Use conductance-based GIF dynamics with multiple synaptic time constants. |
| `sim.create(bp.gif_pop_psc_exp, size, params=parameters)` | Use population GIF dynamics with exponential postsynaptic currents. |
| `sim.create(bp.gif_psc_exp, size, params=parameters)` | Use current-based GIF dynamics with exponential postsynaptic currents. |
| `sim.create(bp.gif_psc_exp_multisynapse, size, params=parameters)` | Use current-based GIF dynamics with multiple synaptic time constants. |

## Generalized LIF neurons

Use these APIs for Allen Institute generalized leaky integrate-and-fire dynamics.

| Constructor | Use when |
|---|---|
| `sim.create(bp.glif_cond, size, params=parameters)` | Use conductance-based GLIF dynamics. |
| `sim.create(bp.glif_psc, size, params=parameters)` | Use current-based GLIF dynamics. |
| `sim.create(bp.glif_psc_double_alpha, size, params=parameters)` | Use current-based GLIF dynamics with double-alpha postsynaptic currents. |

## Multi-timescale adaptive-threshold neurons

Use these APIs when the requested NEST model uses MAT or adaptive MAT threshold dynamics.

| Constructor | Use when |
|---|---|
| `sim.create(bp.mat2_psc_exp, size, params=parameters)` | Use the two-timescale MAT model with exponential currents. |
| `sim.create(bp.amat2_psc_exp, size, params=parameters)` | Use the adaptive MAT variant with exponential currents. |

## Hodgkin-Huxley neurons

Use these APIs for biophysical conductance-based dynamics, selecting Clopath or gap-junction support only when the network requires it.

| Constructor | Use when |
|---|---|
| `sim.create(bp.hh_psc_alpha, size, params=parameters)` | Use Hodgkin-Huxley dynamics with alpha-shaped postsynaptic currents. |
| `sim.create(bp.hh_psc_alpha_clopath, size, params=parameters)` | Use alpha currents while exposing voltage State required by `clopath_synapse`. |
| `sim.create(bp.hh_psc_alpha_gap, size, params=parameters)` | Use alpha currents in a gap-junction network. |
| `sim.create(bp.hh_cond_exp_traub, size, params=parameters)` | Use the Traub Hodgkin-Huxley model with exponential conductances. |
| `sim.create(bp.hh_cond_beta_gap_traub, size, params=parameters)` | Use Traub dynamics with beta conductances and gap junctions. |
| `sim.create(bp.ht_neuron, size, params=parameters)` | Use the Hill-Tononi thalamocortical neuron. |

## Izhikevich neurons

Use this family when the source model explicitly requires NEST-compatible Izhikevich dynamics.

| Constructor | Use when |
|---|---|
| `sim.create(bp.izhikevich, size, params=parameters)` | Use the NEST-compatible Izhikevich neuron. |

## Point-process neurons

Use these APIs for delta point-process dynamics or the two-compartment interface required by Urbanczik-Senn learning.

| Constructor | Use when |
|---|---|
| `sim.create(bp.pp_psc_delta, size, params=parameters)` | Use a point-process neuron with leaky integration of delta-shaped currents. |
| `sim.create(bp.pp_cond_exp_mc_urbanczik, size, params=parameters)` | Use the two-compartment conductance model read by `urbanczik_synapse`. |

## Rate neurons

Use these APIs for vectorized rate, noisy-rate, template, or mean-field dynamics rather than spike-resolved neurons.

| Constructor | Use when |
|---|---|
| `sim.create(bp.lin_rate_ipn, size, params=parameters)` | Use a linear rate unit with input noise. |
| `sim.create(bp.lin_rate_opn, size, params=parameters)` | Use a linear rate unit with output noise. |
| `sim.create(bp.tanh_rate_ipn, size, params=parameters)` | Use a tanh rate unit with input noise. |
| `sim.create(bp.tanh_rate_opn, size, params=parameters)` | Use a tanh rate unit with output noise. |
| `sim.create(bp.sigmoid_rate_ipn, size, params=parameters)` | Use a sigmoid rate unit with input noise. |
| `sim.create(bp.sigmoid_rate_gg_1998_ipn, size, params=parameters)` | Use the `sigmoid_rate_gg_1998_ipn` nonlinear-rate variant with input noise. |
| `sim.create(bp.gauss_rate_ipn, size, params=parameters)` | Use a Gaussian rate unit with input noise. |
| `sim.create(bp.threshold_lin_rate_ipn, size, params=parameters)` | Use threshold-linear rate dynamics with input noise. |
| `sim.create(bp.threshold_lin_rate_opn, size, params=parameters)` | Use threshold-linear rate dynamics with output noise. |
| `sim.create(bp.rate_neuron_ipn, size, params=parameters)` | Use the input-noise rate-neuron template. |
| `sim.create(bp.rate_neuron_opn, size, params=parameters)` | Use the output-noise rate-neuron template. |
| `sim.create(bp.rate_transformer_node, size, params=parameters)` | Use the NEST-compatible rate-transformer template. |
| `sim.create(bp.siegert_neuron, size, params=parameters)` | Use the Siegert mean-field diffusion-rate model. |

## Binary neurons

Use these APIs for deterministic or stochastic binary-state dynamics.

| Constructor | Use when |
|---|---|
| `sim.create(bp.mcculloch_pitts_neuron, size, params=parameters)` | Use deterministic binary activation with a Heaviside gain. |
| `sim.create(bp.ginzburg_neuron, size, params=parameters)` | Use stochastic binary activation with a sigmoidal or affine gain. |
| `sim.create(bp.erfc_neuron, size, params=parameters)` | Use stochastic binary activation with a complementary error-function gain. |

## Relay and utility neurons

Use these APIs to relay spikes without transforming them or to emit fixed-interval spikes.

| Constructor | Use when |
|---|---|
| `sim.create(bp.parrot_neuron, size, params=parameters)` | Use a one-to-one spike relay, including ports that need an intermediate neuron. |
| `sim.create(bp.ignore_and_fire, size, params=parameters)` | Use a fixed-interval spike source. |

## Canonical construction

Use NEST parameter names with BrainUnit quantities. The same `Simulator.create()` pattern applies across all families.

```python
import brainunit as u
from brainpy import state as bp

sim = bp.Simulator(dt=0.1 * u.ms)
parameters = {
    "C_m": 250.0 * u.pF,
    "tau_m": 20.0 * u.ms,
    "V_th": -55.0 * u.mV,
    "I_e": 350.0 * u.pA,
}
population = sim.create(bp.iaf_psc_alpha, 100, params=parameters)
```

**Invariant:** Preserve NEST parameter names but never strip their physical units. This directory identifies the BrainPy-State API surface; use upstream NEST documentation for model physiology.

## Selection boundaries

- Choose `iaf_psc_delta`, `iaf_psc_alpha`, or `iaf_psc_exp` from the requested postsynaptic response; when the response shape is unspecified, preserve the upstream NEST model rather than guessing a substitute.
- Do not replace a `psc` model with a `cond` model during a port.
- Choose a `_ps` variant only when sub-grid spike timing matters; route its validation to event-count and spike-time parity.
- Choose a multisynapse variant when receptor ports or multiple time constants are part of the model; do not silently collapse channels.
- Use `aeif_psc_delta_clopath` or `hh_psc_alpha_clopath` with `clopath_synapse`.
- Use `pp_cond_exp_mc_urbanczik` with `urbanczik_synapse`.
- Open `integration-categories.md` after selecting a family when numerical behavior matters.
- Open `divergence-and-parity.md` before claiming parity for multi-compartment, stochastic, precise-spiking, plastic, or network-level behavior.

## Official sources

- https://brainx.chaobrain.com/brainpy-state/nest-style/models.html
- https://brainx.chaobrain.com/brainpy-state/apis/nest-neurons.html
