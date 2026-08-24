# BrainX study record

## Scope and ownership

- Biological scale: one isopotential body-wall muscle cell with conductance-based channel and intracellular calcium dynamics.
- BrainCell: `SingleCompartment`, a custom `braincell.Channel`, `DiffEqState`, `init_state()`, and `update()`.
- BrainUnit: total conductance (nS), capacitance (pF), current (pA), voltage (mV), calcium concentration (mM), time (ms), and explicit decimal conversion only at observation/file boundaries.
- BrainState: scoped `dt`/`t`, transformed `for_loop`, and one stable `jit` boundary around a complete rollout.
- Braintools: parameter broadcasting/initialization. Host NumPy/SciPy are limited to ATF loading, deterministic preprocessing, ABC proposal generation, summaries, and artifact serialization.

## Scientific source reconciliation

- Du et al. (2025), PLOS Computational Biology 21(1), e1012318, defines the HH model, four 15/20/25/30 pA protocols, and simulation-based inference.
- The authors' `HH_body_wall_SNVI.ipynb` fixes 500 ms duration, 57.8-257.8 ms stimulation, 0.1 ms simulation `dt`, `/0.75` current calibration, and six fitted coordinates `(gCa, gK, gL, C, gSLO2, V_th)`.
- The paper equation and `Muscle_cell_model.ipynb` include the voltage-dependent `z_inf^3` factor in SLO-2. The fitting helper `HH_helper_bp.py` omits that factor. This case follows the paper equation and records the discrepancy.
- The researcher requested exactly six currents. The small M-type current in the authors' helper is excluded; EGL-19, SHK-1, SLO-2, Kr, Na/NCA, and leak remain.

## Lifecycle and axis design

- Candidate axis: `SingleCompartment.size` is the number of independent parameter lanes. Each lane owns independent voltage and channel State; parameters and input arrays are shape-aligned.
- Time axis: one time-major current protocol enters `brainstate.transform.for_loop`; no Python timestep loop is used.
- Candidate initialization: a new cell is constructed and `init_state()` is called for every simulation batch, preventing State carryover across inference rounds or protocols.
- Observation boundary: voltage stays unit-bearing through the rollout and is converted to mV only when returned to host-side summaries.

## Fitting design and interpretation

- Backend: bounded sequential rejection ABC because spike-count/timing summaries are discontinuous. It is a simulation-based parameter-fitting method, but it does not reproduce the paper's neural likelihood-estimator training.
- Parameter map order is explicit: `g_egl19_nS`, `g_shk1_nS`, `g_leak_nS`, `capacitance_pF`, `g_slo2_nS`, `v_shift_mV`.
- The lowest-discrepancy retained candidate is a best-fit ABC sample, not MAP. Kernel-weighted retained samples provide an approximate calibration distribution, not an exact Bayesian posterior.
- Parameter recovery and boundary/sensitivity evidence gate mechanistic interpretation. Predictive behavior may still be assessed if parameters are weakly identified.

## Verification design

- Formula checks: six-current key set, current reversal/sign behavior, gate derivative direction, units, and finite calcium dynamics.
- Lifecycle checks: data split, `init_state`, reset independence, candidate-batch shape, finite nominal/boundary rollouts, and no pre-stimulus spikes.
- Numerical checks: compare 0.1 ms `exp_euler` results against a smaller-step reference or supported solver and preserve protocol spike counts.
- Predictive checks: held-out raw traces, residual metrics, spike counts, first-spike latency, ISI, peaks, and monotonic current-response trend.
- Claim boundary: report qualitative protocol agreement separately from waveform disagreement and withhold unique biological parameter claims when recovery is poor.
