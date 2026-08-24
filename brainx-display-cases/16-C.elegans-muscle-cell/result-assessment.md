# Deterministic result assessment

## Outcome

The fitted model shows qualitative protocol-level agreement but not waveform-level agreement on the three held-out currents. It reproduces held-out stimulus spike counts (3, 3, and 4) and the direction of decreasing interspike interval as current increases. It does not reproduce spike phase/timing or post-stimulus relaxation closely enough to claim full trace consistency.

## Parameter estimate

The locked selection rule chooses the lowest-summary-discrepancy ABC sample:

| Parameter | Best-fit value | Posterior mean | Boundary evidence |
|---|---:|---:|---|
| EGL-19 conductance | 12.400 nS | 14.512 nS | 14.96% retained mass near a bound |
| SHK-1 conductance | 45.000 nS | 42.655 nS | Best fit at upper bound; 49.07% near a bound |
| Leak conductance | 0.500 nS | 0.147 nS | Best fit at upper bound; 46.13% near a bound |
| Capacitance | 16.953 pF | 19.670 pF | 4.52% near a bound |
| SLO-2 conductance | 2.376 nS | 2.151 nS | 17.95% near a bound |
| Voltage shift | 10.520 mV | 11.025 mV | no retained mass near a bound |

These are calibration estimates, not uniquely identified biological measurements. The upper-bound SHK-1/leak solution and broad retained distributions are explicit warnings against mechanistic interpretation.

## Held-out evidence

| Trace/current | Spikes exp/model | ISI exp/model (ms) | First latency exp/model (ms) | RMSE (mV) | Correlation | Post-stim spikes exp/model |
|---|---:|---:|---:|---:|---:|---:|
| #6 / 15 pA | 3 / 3 | 70.15 / 61.05 | 45.2 / 75.9 | 15.310 | 0.116 | 0 / 1 |
| #7 / 20 pA | 3 / 3 | 60.50 / 55.00 | 37.3 / 60.2 | 14.745 | 0.117 | 0 / 1 |
| #8 / 25 pA | 4 / 4 | 54.73 / 49.17 | 29.9 / 48.3 | 14.212 | 0.304 | 0 / 0 |

The first simulated held-out spikes are 18.4-30.7 ms late. The model's ISIs are 5.5-9.1 ms shorter. These timing errors explain the low whole-trace correlations despite correct stimulus spike counts.

## Controls and recovery

- The no-current trajectory is finite and has no spike before 57.8 ms. It later spikes autonomously at 310.5 and 440.5 ms, showing that late post-stimulus activity is not solely stimulus-driven.
- A 0.05 ms reference preserves all four stimulus spike counts; relative to 0.1 ms, voltage RMSE is 0.080 mV and maximum absolute error is 0.418 mV.
- Three exact-budget synthetic recovery cases are diagnostic only. SLO-2 recovery is poor (normalized RMSE 0.394, recovered-vs-true correlation -0.018); the small case count precludes parameter-identifiability claims for every coordinate.

## Claim-evidence matrix

| Claim | Evidence | Verdict |
|---|---|---|
| Six requested currents are represented | Current-component inventory and unit test | Supported |
| Model reproduces stimulus spike counts at held-out currents | Exact 3/3, 3/3, and 4/4 comparisons | Supported |
| Model reproduces current-frequency direction | Experimental and simulated ISIs both decrease from 15 to 25 pA | Supported |
| Model reproduces held-out voltage waveforms | RMSE 14.2-15.3 mV, correlation 0.12-0.30, late spikes | Refuted under locked thresholds |
| Fitted values identify biological conductances | Boundary concentration and limited/poor recovery | Not supported |
| Numerical step drives the conclusion | 0.05 ms parity preserves all spike-count outcomes | Refuted |
