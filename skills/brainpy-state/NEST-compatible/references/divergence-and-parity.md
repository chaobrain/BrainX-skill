# Divergence and parity

Use this reference when porting NEST code, locating plasticity parameters, explaining a NEST mismatch, selecting trace versus distributional comparison, or deciding which documented tolerance and coverage claim applies.

## Divergence map

| Divergence | Porting decision | Failure prevented |
|---|---|---|
| Learning State location | Move NEST neuron-owned learning parameters to the BrainPy-State synapse spec or broadcast node where documented. | Setting a recognized name on the wrong object and silently changing the learning rule. |
| Parameter-location maps | Check each STDP or Clopath parameter rather than assuming all parameters move together. | Moving voltage-filter State that must remain on the neuron. |
| Online versus deferred updates | Compare at the documented event times and use the documented numerical band. | Reporting a known timing residual as an implementation failure. |
| Device and recording conventions | Preserve analog observer direction, current-versus-spike source semantics, and in-memory readback. | Empty recordings, one-step misalignment, or unsupported file output. |
| Independent PRNG streams | Compare stochastic drives, random connectivity, and stochastic neurons distributionally across seeds. | Requiring impossible sample-by-sample identity between NEST and JAX. |

## STDP parameter placement

BrainPy-State evaluates plasticity kernels every time step on an event-driven substrate. It keeps presynaptic and postsynaptic traces on the plasticity side so rules can run without a NEST archiving-capable postsynaptic neuron.

| Parameter or State | NEST location | BrainPy-State location |
|---|---|---|
| `tau_minus` for trace-based STDP | Postsynaptic archiving neuron | Synapse spec |
| `tau_minus_triplet` | Postsynaptic archiving neuron | `stdp_triplet_synapse` spec |
| `A_LTP`, `A_LTD`, `theta_plus`, `theta_minus` | Clopath-capable postsynaptic neuron | `clopath_synapse` spec |
| `tau_u_bar_plus`, `tau_u_bar_minus` | Clopath postsynaptic neuron | Remain on the compatible `_clopath` neuron |
| `delay_u_bars` | Postsynaptic analog ring buffer | `clopath_synapse` spec; the online reader uses a one-step lag |
| Dopamine level `n` | Per-synapse State | `volume_transmitter` broadcast State |
| `tau_n` | Common property bound to the transmitter | `volume_transmitter`; the synapse spec value must match |
| Dopamine STDP weight parameters | Common synapse properties | `stdp_dopamine_synapse` spec |

```python
# Port tau_minus from the NEST postsynaptic neuron to the synapse spec.
plasticity = bp.stdp_synapse(
    tau_plus=20.0 * u.ms,
    tau_minus=20.0 * u.ms,
)
```

**Invariant:** Move only the parameters named by the location map. Clopath voltage-filter time constants remain neuron State even though Clopath learning amplitudes and thresholds move to the synapse spec.

## Nearest-neighbour STDP conventions

Use these constructors when the port requires a specific nearest-neighbour or hardware-emulated STDP pairing convention.

| Constructor | Use when |
|---|---|
| `bp.stdp_nn_symm_synapse(*args, **kwargs)` | Use when each spike must pair only with the nearest preceding spike on the opposite side. |
| `bp.stdp_nn_restr_synapse(*args, **kwargs)` | Use symmetric nearest-neighbour pairing with a one-pairing-per-spike availability restriction. |
| `bp.stdp_nn_pre_centered_synapse(*args, **kwargs)` | Use when a postsynaptic spike must consume all presynaptic spikes accumulated since the preceding postsynaptic spike, while a presynaptic spike pairs with the nearest preceding postsynaptic spike. |
| `bp.stdp_facetshw_synapse_hom(*args, **kwargs)` | Use when causal and acausal charge must accumulate before quantized LUT readout on the configured cycle. |

Do not replace one nearest-neighbour variant with another because their class names look similar; the pairing convention changes the weight trajectory.

## Documented numerical divergences

The direction and ordering of each effect remain exact; only the documented magnitude or initialization behavior diverges.

| Rule or behavior | Cause | Accepted result |
|---|---|---|
| Clopath online LTP | BrainPy-State reads online with an intrinsic one-step lag; NEST reads a delayed analog ring buffer and defers the history sum. | Use `delay_u_bars=0.1 ms` for parity and accept the documented `<= 5%` band; the NEST `4.0 ms` default is not reproduced online. |
| Dopamine-modulated STDP | BrainPy-State integrates every step; NEST integrates lazily at send/update time. | Accept the documented approximately `0.2%` band while requiring exact direction and ordering. |
| Nearest-neighbour phantom pre at `t=0` | NEST initializes the first send time at zero; BrainPy-State represents that no real presynaptic spike has occurred. | Do not require the phantom facilitation for `stdp_nn_symm_synapse` or `stdp_nn_restr_synapse`. |

For ordinary online-versus-deferred pair STDP, compare weights at presynaptic send times, where the operations coincide.

## Choose the parity mode

| Mode | Use when | Compare |
|---|---|---|
| Trace | Drive and dynamics are deterministic under the same `dt` and fixed inputs. | Per-sample or maximum absolute error, with an optional one-step recorder alignment. |
| Distributional | Poisson input, random connectivity, or stochastic neurons use independent NEST and JAX PRNG streams. | Seed-aggregated statistics; use at least five seeds for the documented category-D protocol. |

The comparison uses `|actual - reference| <= atol + rtol * |reference|`; do not divide by the reference, because a valid reference can be zero.

## Validation categories

Validation categories describe how parity is asserted. They are related to, but distinct from, the integration categories that describe how a model is solved.

| Category | Kind | Representative metric | Documented tolerance |
|---|---|---|---|
| A | Adaptive numerical integrator | AdEx, HH, or Izhikevich `V_m` trace | Absolute `1e-3 mV` trace tolerance |
| B | Analytic exact propagator | Linear `iaf_psc_*` voltage or PSC trace | Absolute `1e-6 mV`; aligned variant `5e-2 mV` with at most one step |
| C | Conductance, coupled, or mean-field | `iaf_cond_*` voltage or `siegert_neuron` rate | `1e-3 mV` trace or `5%` rate |
| D | PRNG-divergent distribution | Network firing rate or ISI CV | Seed mean within `5%` over at least five seeds |
| E | Spike time or event count | Precise spiking or PSC peak timing | Event-count difference at most two and step difference at most one |

## Mismatch workflow

1. Confirm that both sides use the same model name, parameter values, physical units, `dt`, connection rule, and device window.
2. Check the STDP and Clopath parameter-location map before comparing learning behavior.
3. Determine whether the observable is deterministic or PRNG-divergent.
4. Select the matching validation category and recorder alignment.
5. Compare the same observable at the same lifecycle point; for deferred plasticity, use the documented send-time comparison.
6. Apply only a documented band. Do not derive a new tolerance from one run.
7. Report known coverage gaps separately from parity failures.

Run the live-NEST harness when the package source and NEST are available:

```bash
python -m pytest brainpy_state/_nest_validation/ -m requires_nest -q
```

Tests marked `requires_nest` skip when NEST cannot be imported; harness unit tests still run without NEST.

## Known coverage gaps

- `iaf_cond_alpha_mc` is not yet validated against NEST; `pp_cond_exp_mc_urbanczik` does have per-compartment parity coverage.
- File-backed recording is not implemented; use the in-memory `SimulationResult`.
- Some legacy distributional tests use one realization rather than the documented five-seed protocol.

## Official sources

- https://brainx.chaobrain.com/brainpy-state/nest-style/divergences/index.html
- https://brainx.chaobrain.com/brainpy-state/nest-style/validation-status.html
- https://brainx.chaobrain.com/brainpy-state/nest-style/divergences/stdp.html
