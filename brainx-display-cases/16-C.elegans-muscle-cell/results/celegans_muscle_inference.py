"""Fit and validate the Du et al. C. elegans body-wall muscle model.

The published model is seven-dimensional (V, m, h, n, p, q, Ca) and has
six ionic currents: EGL-19, SHK-1, SLO-2, Kr, NCA sodium, and leak.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import braincell
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc


DATA_FILE = Path(__file__).with_name("Fig4A-D.txt")
OUTPUT_DIR = Path(__file__).with_name("results")

DT = 0.1 * u.ms
STIMULUS_ON = 57.8 * u.ms
STIMULUS_OFF = 257.8 * u.ms
SPIKE_THRESHOLD = 10.0
INPUT_CALIBRATION = 0.75

CURRENT_BY_TRACE = {6: 15.0, 7: 20.0, 8: 25.0, 9: 30.0}
PARAMETER_NAMES = (
    "g_egl19_nS",
    "g_shk1_nS",
    "g_leak_nS",
    "capacitance_pF",
    "g_slo2_nS",
    "v_shift_mV",
)
PRIOR_LOW = np.array([10.0, 10.5, 1.0e-4, 10.0, 0.1, 7.0])
PRIOR_HIGH = np.array([25.0, 45.0, 0.5, 30.0, 4.0, 13.0])

SUMMARY_NAMES = (
    "stimulus_spike_count",
    "stimulus_mean_isi_ms",
    "first_spike_latency_ms",
    "resting_mean_mV",
    "stimulus_mean_mV",
    "stimulus_std_mV",
    "stimulus_peak_mV",
    "post_stimulus_spike_count",
)
SUMMARY_SCALES = np.array([1.0, 20.0, 10.0, 2.0, 5.0, 5.0, 5.0, 0.5])


class CelegansMuscleCurrents(braincell.Channel):
    """Coupled six-current mechanism used by the authors for Figure 4."""

    root_type = braincell.HHTypedNeuron

    def __init__(
        self,
        size,
        *,
        g_egl19,
        g_shk1,
        g_slo2,
        g_leak,
        v_shift,
    ):
        super().__init__(size=size)
        self.g_egl19 = braintools.init.param(g_egl19, self.varshape)
        self.g_shk1 = braintools.init.param(g_shk1, self.varshape)
        self.g_slo2 = braintools.init.param(g_slo2, self.varshape)
        self.g_leak = braintools.init.param(g_leak, self.varshape)
        self.v_shift = braintools.init.param(v_shift, self.varshape)

        self.g_kr = 3.2 * u.nS
        self.g_na = 0.1 * u.nS
        self.e_ca_offset = 60.0 * u.mV
        self.e_k_offset = -40.0 * u.mV
        self.e_na_offset = 15.0 * u.mV
        self.e_leak_offset = -24.0 * u.mV

        self.phi_m = 1.2
        self.phi_n = 1.2
        self.phi_slo2 = 0.04
        self.alpha_slo2 = 43.0
        self.beta_slo2 = 0.09
        self.ca_rest = 0.001 * u.mM
        self.ca_recovery = 0.075 / u.ms
        self.ca_influx_per_current = 1.5e-5 * u.mM / u.ms / u.pA

    def init_state(self, V, batch_size=None):
        del V, batch_size
        shape = self.varshape
        self.m = braincell.DiffEqState(u.math.full(shape, 0.01))
        self.h = braincell.DiffEqState(u.math.full(shape, 0.6))
        self.n = braincell.DiffEqState(u.math.full(shape, 0.99))
        self.p = braincell.DiffEqState(u.math.zeros(shape))
        self.q = braincell.DiffEqState(u.math.zeros(shape))
        self.ca = braincell.DiffEqState(u.math.zeros(shape) * u.mM)

    def reset_state(self, V, batch_size=None):
        del V, batch_size
        self.m.value = u.math.full(self.varshape, 0.01)
        self.h.value = u.math.full(self.varshape, 0.6)
        self.n.value = u.math.full(self.varshape, 0.99)
        self.p.value = u.math.zeros(self.varshape)
        self.q.value = u.math.zeros(self.varshape)
        self.ca.value = u.math.zeros(self.varshape) * u.mM

    def _voltage_terms(self, V):
        voltage_mV = V.to_decimal(u.mV)
        shift_mV = self.v_shift.to_decimal(u.mV)

        m_inf = 1.0 / (1.0 + u.math.exp(-(voltage_mV + 8.0 - shift_mV) / 8.6))
        tau_m = 0.4 + 0.7 / (
            u.math.exp(-(voltage_mV + 5.0 - shift_mV) / 15.0)
            + u.math.exp((voltage_mV + 5.0 - shift_mV) / 15.0)
        )
        h_inf = 0.42 / (
            1.0 + u.math.exp((voltage_mV + 11.0 - shift_mV) / 2.0)
        ) + 0.28

        n_inf = 0.5 * (
            u.math.tanh((voltage_mV - shift_mV + 15.2) / 36.22) + 1.0
        )
        tau_n = 1.18 + 511.78 / (
            1.0 + u.math.exp((voltage_mV - shift_mV + 89.3) / 21.92)
        )

        q_inf = 0.5 * (
            u.math.tanh((voltage_mV - shift_mV + 42.0) / 5.0) + 1.0
        )
        z_inf = 1.0 / (1.0 + u.math.exp(-(voltage_mV + 33.4) / 3.2))
        return m_inf, tau_m, h_inf, n_inf, tau_n, q_inf, z_inf

    def current_components(self, V):
        _, _, _, _, _, q_inf, z_inf = self._voltage_terms(V)
        i_egl19 = self.g_egl19 * self.m.value**2 * self.h.value * (
            self.v_shift + self.e_ca_offset - V
        )
        i_shk1 = self.g_shk1 * self.n.value**4 * (
            self.v_shift + self.e_k_offset - V
        )
        i_slo2 = self.g_slo2 * self.p.value * z_inf**3 * (self.e_k_offset - V)
        i_kr = self.g_kr * (1.0 - self.q.value) * q_inf * (
            self.e_k_offset - V
        )
        i_na = self.g_na * (self.v_shift + self.e_na_offset - V)
        i_leak = self.g_leak * (self.v_shift + self.e_leak_offset - V)
        return {
            "EGL-19": i_egl19,
            "SHK-1": i_shk1,
            "SLO-2": i_slo2,
            "Kr": i_kr,
            "Na": i_na,
            "Leak": i_leak,
        }

    def current(self, V):
        currents = self.current_components(V)
        return sum(currents.values(), 0.0 * u.pA)

    def compute_derivative(self, V):
        m_inf, tau_m, h_inf, n_inf, tau_n, q_inf, _ = self._voltage_terms(V)
        ca_mM = self.ca.value.to_decimal(u.mM)
        ca_rate = self.alpha_slo2 * ca_mM**2 + self.beta_slo2
        p_inf = self.alpha_slo2 * ca_mM**2 / ca_rate

        self.m.derivative = self.phi_m * (m_inf - self.m.value) / tau_m / u.ms
        self.h.derivative = (h_inf - self.h.value) / (24.0 * u.ms)
        self.n.derivative = self.phi_n * (n_inf - self.n.value) / tau_n / u.ms
        self.p.derivative = (
            self.phi_slo2 * (p_inf - self.p.value) * ca_rate / u.ms
        )
        self.q.derivative = (q_inf - self.q.value) / (62.0 * u.ms)

        i_egl19 = self.current_components(V)["EGL-19"]
        influx = self.ca_influx_per_current * i_egl19
        self.ca.derivative = influx - self.ca_recovery * (
            self.ca.value - self.ca_rest
        )


class CelegansMuscleCell(braincell.SingleCompartment):
    """Single isopotential body-wall muscle cell with total pA/pF/nS units."""

    def __init__(self, parameters: np.ndarray, solver: str = "exp_euler"):
        parameters = np.asarray(parameters, dtype=np.float32)
        if parameters.ndim == 1:
            parameters = parameters[None, :]
        if parameters.shape[1] != len(PARAMETER_NAMES):
            raise ValueError(f"expected six parameters, got shape {parameters.shape}")

        size = parameters.shape[0]
        v_shift = jnp.asarray(parameters[:, 5]) * u.mV
        super().__init__(
            size,
            C=jnp.asarray(parameters[:, 3]) * u.pF,
            V_th=v_shift,
            V_initializer=braintools.init.Constant(-30.0 * u.mV),
            solver=solver,
        )
        self.currents = CelegansMuscleCurrents(
            size,
            g_egl19=jnp.asarray(parameters[:, 0]) * u.nS,
            g_shk1=jnp.asarray(parameters[:, 1]) * u.nS,
            g_leak=jnp.asarray(parameters[:, 2]) * u.nS,
            g_slo2=jnp.asarray(parameters[:, 4]) * u.nS,
            v_shift=v_shift,
        )


def load_experimental_traces(path: Path = DATA_FILE):
    """Read the Axon Text File and retain the four Fig. 4 current traces."""
    data = np.loadtxt(path, delimiter="\t", skiprows=11)
    raw_time_ms = data[:, 0] * 1000.0
    stride = int(round(DT.to_decimal(u.ms) / np.median(np.diff(raw_time_ms))))
    if stride < 1 or not np.isclose(stride * np.median(np.diff(raw_time_ms)), 0.1):
        raise ValueError("experimental sampling interval is incompatible with 0.1 ms")
    time_ms = raw_time_ms[::stride]
    traces = {trace: data[::stride, trace] for trace in CURRENT_BY_TRACE}
    return time_ms, traces


def make_current_protocol(current_pA, time_ms):
    active = (time_ms >= STIMULUS_ON.to_decimal(u.ms)) & (
        time_ms < STIMULUS_OFF.to_decimal(u.ms)
    )
    return np.where(active[:, None], np.asarray(current_pA)[None, :], 0.0) * u.pA


def simulate(
    parameters: np.ndarray,
    current_pA,
    time_ms: np.ndarray,
    *,
    solver: str = "exp_euler",
):
    """Run independent parameter/current lanes through one BrainCell rollout."""
    parameters = np.asarray(parameters, dtype=np.float32)
    if parameters.ndim == 1:
        parameters = parameters[None, :]
    currents = np.broadcast_to(np.asarray(current_pA, dtype=np.float32), (len(parameters),))
    cell = CelegansMuscleCell(parameters, solver=solver)
    cell.init_state()

    times = jnp.asarray(time_ms) * u.ms
    protocol = make_current_protocol(currents, time_ms)

    def step(t, current):
        with brainstate.environ.context(t=t):
            cell.update(current / INPUT_CALIBRATION)
        return cell.V.value

    @brainstate.transform.jit
    def rollout():
        return brainstate.transform.for_loop(step, times, protocol)

    with brainstate.environ.context(dt=DT):
        voltages = rollout()
    return np.asarray(voltages.to_decimal(u.mV))


def _spike_times(voltage_mV: np.ndarray, time_ms: np.ndarray, stop_ms=None):
    crossings = (voltage_mV[1:] >= SPIKE_THRESHOLD) & (
        voltage_mV[:-1] < SPIKE_THRESHOLD
    )
    indices = np.flatnonzero(crossings) + 1
    on = STIMULUS_ON.to_decimal(u.ms)
    selected = time_ms[indices] >= on
    if stop_ms is not None:
        selected &= time_ms[indices] < stop_ms
    return time_ms[indices[selected]]


def summarize_traces(voltages_mV: np.ndarray, time_ms: np.ndarray):
    """Compute summary statistics row-wise for shape (time, lanes)."""
    if voltages_mV.ndim == 1:
        voltages_mV = voltages_mV[:, None]
    on = STIMULUS_ON.to_decimal(u.ms)
    off = STIMULUS_OFF.to_decimal(u.ms)
    rest_mask = (time_ms >= 0.8 * on) & (time_ms < on)
    stim_mask = (time_ms >= on) & (time_ms < off)

    summaries = np.empty((voltages_mV.shape[1], len(SUMMARY_NAMES)))
    for lane in range(voltages_mV.shape[1]):
        spike_times = _spike_times(voltages_mV[:, lane], time_ms, stop_ms=off)
        all_spike_times = _spike_times(voltages_mV[:, lane], time_ms)
        summaries[lane] = (
            len(spike_times),
            np.mean(np.diff(spike_times)) if len(spike_times) > 1 else 0.0,
            spike_times[0] - on if len(spike_times) else off - on,
            np.mean(voltages_mV[rest_mask, lane]),
            np.mean(voltages_mV[stim_mask, lane]),
            np.std(voltages_mV[stim_mask, lane]),
            np.max(voltages_mV[stim_mask, lane]),
            len(all_spike_times) - len(spike_times),
        )
    return summaries


def summary_distance(simulated: np.ndarray, observed: np.ndarray):
    residual = (simulated - observed[None, :]) / SUMMARY_SCALES
    invalid = ~np.all(np.isfinite(simulated), axis=1)
    distance = np.sqrt(np.mean(residual**2, axis=1))
    distance[invalid] = np.inf
    return distance


def _local_samples(rng, elite, count):
    normalized = (elite - PRIOR_LOW) / (PRIOR_HIGH - PRIOR_LOW)
    covariance = np.cov(normalized, rowvar=False) + np.eye(normalized.shape[1]) * 0.0025
    center = np.mean(normalized, axis=0)
    samples = rng.multivariate_normal(center, covariance, size=count)
    samples = np.clip(samples, 0.0, 1.0)
    return PRIOR_LOW + samples * (PRIOR_HIGH - PRIOR_LOW)


def infer_parameters(
    observed_voltage_mV,
    time_ms,
    *,
    samples_per_round=1024,
    rounds=3,
    seed=2025,
):
    """Approximate the posterior with sequential rejection ABC."""
    observed_summary = summarize_traces(observed_voltage_mV, time_ms)[0]
    rng = np.random.default_rng(seed)
    all_parameters = []
    all_summaries = []
    all_distances = []
    round_reports = []
    proposal = None

    for round_index in range(rounds):
        if proposal is None:
            unit_samples = qmc.LatinHypercube(
                d=len(PARAMETER_NAMES), seed=seed
            ).random(samples_per_round)
            parameters = qmc.scale(unit_samples, PRIOR_LOW, PRIOR_HIGH)
        else:
            parameters = _local_samples(rng, proposal, samples_per_round)

        voltages = simulate(parameters, np.full(samples_per_round, 30.0), time_ms)
        summaries = summarize_traces(voltages, time_ms)
        distances = summary_distance(summaries, observed_summary)
        elite_indices = np.argsort(distances)[: max(32, samples_per_round // 16)]
        proposal = parameters[elite_indices]

        all_parameters.append(parameters)
        all_summaries.append(summaries)
        all_distances.append(distances)
        round_reports.append(
            {
                "round": round_index + 1,
                "minimum_distance": float(distances[elite_indices[0]]),
                "elite_distance": float(distances[elite_indices[-1]]),
            }
        )

    parameters = np.concatenate(all_parameters)
    summaries = np.concatenate(all_summaries)
    distances = np.concatenate(all_distances)
    finite_indices = np.flatnonzero(np.isfinite(distances))
    posterior_indices = finite_indices[np.argsort(distances[finite_indices])[:128]]
    posterior_parameters = parameters[posterior_indices]
    posterior_summaries = summaries[posterior_indices]
    posterior_distances = distances[posterior_indices]

    bandwidth = max(float(np.median(posterior_distances)), 1.0e-6)
    weights = np.exp(-0.5 * (posterior_distances / bandwidth) ** 2)
    weights /= weights.sum()
    estimate = np.average(posterior_parameters, axis=0, weights=weights)
    map_parameters = posterior_parameters[np.argmin(posterior_distances)]
    return {
        "estimate": estimate,
        "map": map_parameters,
        "observed_summary": observed_summary,
        "posterior_parameters": posterior_parameters,
        "posterior_summaries": posterior_summaries,
        "posterior_distances": posterior_distances,
        "weights": weights,
        "rounds": round_reports,
    }


def trace_metrics(experimental, simulated, time_ms):
    experimental_summary = summarize_traces(experimental, time_ms)[0]
    simulated_summary = summarize_traces(simulated, time_ms)[0]
    rmse = float(np.sqrt(np.mean((experimental - simulated) ** 2)))
    correlation = float(np.corrcoef(experimental, simulated)[0, 1])
    stimulus_off = STIMULUS_OFF.to_decimal(u.ms)

    def protocol_counts(voltage):
        stimulus_spikes = _spike_times(voltage, time_ms, stop_ms=stimulus_off)
        all_spikes = _spike_times(voltage, time_ms)
        return {
            "stimulus_spike_count": int(len(stimulus_spikes)),
            "stimulus_mean_isi_ms": (
                float(np.mean(np.diff(stimulus_spikes)))
                if len(stimulus_spikes) > 1
                else 0.0
            ),
            "post_stimulus_spike_count": int(len(all_spikes) - len(stimulus_spikes)),
        }

    return {
        "rmse_mV": rmse,
        "correlation": correlation,
        "experimental": dict(zip(SUMMARY_NAMES, experimental_summary.tolist())),
        "simulated": dict(zip(SUMMARY_NAMES, simulated_summary.tolist())),
        "experimental_protocol": protocol_counts(experimental),
        "simulated_protocol": protocol_counts(simulated),
    }


def save_outputs(result, time_ms, traces, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    estimate = result["map"]
    currents = np.array(list(CURRENT_BY_TRACE.values()))
    simulated = simulate(np.repeat(estimate[None, :], 4, axis=0), currents, time_ms)

    metrics = {}
    for lane, trace_number in enumerate(CURRENT_BY_TRACE):
        split = "train" if trace_number == 9 else "test"
        metrics[str(trace_number)] = {
            "current_pA": CURRENT_BY_TRACE[trace_number],
            "split": split,
            **trace_metrics(traces[trace_number], simulated[:, lane], time_ms),
        }

    report = {
        "model": {
            "dynamic_states": ["V", "m", "h", "n", "p", "q", "Ca"],
            "ionic_currents": ["EGL-19", "SHK-1", "SLO-2", "Kr", "Na", "Leak"],
            "training_trace": 9,
            "training_current_pA": 30.0,
            "test_traces": [6, 7, 8],
        },
        "inference": {
            "method": "sequential rejection approximate Bayesian computation",
            "parameter_names": list(PARAMETER_NAMES),
            "prior_low": dict(zip(PARAMETER_NAMES, PRIOR_LOW.tolist())),
            "prior_high": dict(zip(PARAMETER_NAMES, PRIOR_HIGH.tolist())),
            "posterior_mean": dict(zip(PARAMETER_NAMES, result["estimate"].tolist())),
            "map": dict(zip(PARAMETER_NAMES, result["map"].tolist())),
            "validation_parameters": "map",
            "rounds": result["rounds"],
        },
        "metrics": metrics,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    with (output_dir / "posterior_samples.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([*PARAMETER_NAMES, "distance", "weight"])
        for parameter, distance, weight in zip(
            result["posterior_parameters"],
            result["posterior_distances"],
            result["weights"],
        ):
            writer.writerow([*parameter, distance, weight])

    with (output_dir / "trace_predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        header = ["time_ms"]
        for trace_number in CURRENT_BY_TRACE:
            header.extend([f"trace_{trace_number}_experimental_mV", f"trace_{trace_number}_simulated_mV"])
        writer.writerow(header)
        for row in range(len(time_ms)):
            values = [time_ms[row]]
            for lane, trace_number in enumerate(CURRENT_BY_TRACE):
                values.extend([traces[trace_number][row], simulated[row, lane]])
            writer.writerow(values)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, sharey=True)
    for lane, (trace_number, current_pA) in enumerate(CURRENT_BY_TRACE.items()):
        axis = axes[lane]
        axis.plot(time_ms, traces[trace_number], color="#b33a3a", lw=1.2, label="experiment")
        axis.plot(time_ms, simulated[:, lane], color="#176b87", lw=1.2, label="model")
        axis.axvspan(
            STIMULUS_ON.to_decimal(u.ms),
            STIMULUS_OFF.to_decimal(u.ms),
            color="#dadada",
            alpha=0.45,
            linewidth=0,
        )
        split = "training" if trace_number == 9 else "held-out"
        axis.set_title(f"{current_pA:.0f} pA, Trace #{trace_number} ({split})", fontsize=10)
        axis.set_ylabel("mV")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(output_dir / "held_out_validation.png", dpi=180)
    plt.close(fig)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-round", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    brainstate.random.seed(args.seed)
    time_ms, traces = load_experimental_traces()
    result = infer_parameters(
        traces[9],
        time_ms,
        samples_per_round=args.samples_per_round,
        rounds=args.rounds,
        seed=args.seed,
    )
    report = save_outputs(result, time_ms, traces, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
