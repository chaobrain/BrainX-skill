# BrainX study record

## Represented scale and package ownership

- The model explicitly represents ion-channel gates and currents at one isopotential cellular membrane. BrainCell owns the biological model.
- BrainUnit owns voltage, time, conductance, and current quantities and the conversion boundaries used by SciPy and plotting.
- BrainState owns mutable channel gate state and transformed execution. No point-neuron network, event projection, neural mass, morphology, or training graph is represented.

## Relevant abstractions and invariants

- Implement SHK-1 and EGL-19 as `braincell.channel._base.HH` subclasses. Declare independent `Gate` objects, implement one `inf/tau` kinetic form per gate, and compute current through the ion-provided reversal potential.
- Attach SHK-1 to `braincell.ion.PotassiumFixed` and EGL-19 to `braincell.ion.CalciumFixed`; do not embed reversal-potential ownership in mutable channel state.
- Keep the BrainCell current convention `g * gates * (E - V)`. Convert to the experimental outward-positive convention only at the observation boundary.
- Initialize every voltage-step rollout at gate steady state for -60 mV. Independent voltage steps never share runtime gate state.
- Preserve quantities through model evaluation. Convert explicitly to mV, ms, nS, and pA only at SciPy, JSON, NPZ, and Matplotlib boundaries.
- Use bounded robust least squares because Igor parsing and host-side curve fitting are nondifferentiable boundaries. Preserve every start and select the finite minimum-loss result by the locked rule.

## Fitting design

- SHK-1: fit each baseline-corrected trace to a fourth-power activation response, derive conductance-normalized `n_inf` points and activation `tau_n` points, then fit a monotone tanh steady-state function and a decreasing logistic time-constant function. Fix `E_K=-30 mV` from the source model.
- EGL-19: fit the isolated WT traces jointly with `m^2 h` kinetics, a sigmoid activation gate, a residual sigmoid inactivation gate, a bell-shaped activation time constant, and a constant inactivation time constant. Fix `E_Ca=60 mV` from the source model.
- Run synthetic recovery through the same trace times, voltage commands, exclusions, bounds, starts, and objective before accepting observed-data parameter interpretation.
- Report conductance and kinetic values as protocol-calibrated parameters. Withhold unique biological parameter claims because only population averages are available.

## Execution and validation design

- Parse packed files with `igor2`, preserving the raw files unchanged.
- Fit on host with deterministic SciPy starts; evaluate all protocols in one shape-stable array operation.
- Validate formula direction, gate bounds, unit-bearing current, reversal behavior, channel initialization/reset, derivative direction, protocol timing, wave mapping, encode/decode order, and synthetic recovery.
- Record raw extracted targets, fitted traces, per-voltage gate observations, parameters, metrics, provenance, and an artifact manifest before review.

## Sources studied

- `brainx-general-guard`, `braincell`, `brainunit`, `brainstate`, and `brainx-acceleration` root skills.
- BrainCell custom HH authoring, area-scaling, and complete HH fitting examples.
- BrainX modeling-loop parameter-fitting workflow.
- Du et al. (2025), PLOS Computational Biology 21(1):e1012318, especially the voltage-clamp protocol, Fig. 2 SHK-1 workflow, Fig. 3 EGL-19 workflow, and fixed reversal potentials.
