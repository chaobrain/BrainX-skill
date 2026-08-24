"""Single-cell perturbation in a feature-structured V1 layer 2/3 E/I network.

The parameter set and qualitative scoring rule below were fixed before the first
perturbation run.  This is a compact mechanistic demonstration, not a fitted
model of the Chettih-Harvey data.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import brainevent
import brainpy
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "results"
SEED = 20260812
N_E, N_I = 80, 20
DT = 0.5 * u.ms
DURATION = 500.0 * u.ms
N_STIMULI, N_TRIALS = 16, 3
P_CONNECT = 0.25
TARGET = 0
BASE_CURRENT = 17.5 * u.mA
STIM_CURRENT = 7.0 * u.mA
NOISE_SD = 2.0 * u.mA
TAU_E, TAU_I = 5.0 * u.ms, 10.0 * u.ms
PERTURB_STEPS = np.rint(np.linspace(100.0, 350.0, 6) / DT.to_decimal(u.ms)).astype(int)
PERTURB_STRENGTH = len(PERTURB_STEPS)
CORR_EDGES = np.array([-1.0, -0.35, 0.0, 0.35, 0.65, 0.85, 1.000001])

# strength is the mean nonzero event conductance; specificity multiplies
# cos(preference-direction difference). Values were frozen before outcomes.
REGIMES = {
    "inhibition_dominant": {
        "ee_strength_mS": 0.55, "ee_specificity": 0.0,
        "ei_strength_mS": 0.75, "ei_specificity": 0.0,
        "inh_strength_mS": 7.0, "inh_specificity": 0.0,
    },
    "broad_inhibition": {
        "ee_strength_mS": 0.75, "ee_specificity": 0.75,
        "ei_strength_mS": 1.0, "ei_specificity": 0.0,
        "inh_strength_mS": 5.5, "inh_specificity": 0.0,
    },
    "specific_strong_ei": {
        "ee_strength_mS": 0.75, "ee_specificity": 0.75,
        "ei_strength_mS": 1.25, "ei_specificity": 0.90,
        "inh_strength_mS": 5.5, "inh_specificity": 0.85,
    },
}


class DenseEventComm(brainstate.nn.Module):
    """Apply a fixed heterogeneous BrainEvent weight matrix to binary spikes."""

    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def update(self, spikes):
        return brainevent.BinaryArray(spikes) @ self.weights


def preferred_vectors(rng):
    """Return random 2-D unit receptive-field preference vectors."""
    angles = rng.uniform(0.0, 2.0 * np.pi, N_E + N_I)
    return angles, np.column_stack((np.cos(angles), np.sin(angles)))


def weight_matrix(pre_angles, post_angles, mask, strength_mS, specificity):
    similarity = np.cos(pre_angles[:, None] - post_angles[None, :])
    modulation = np.clip(1.0 + specificity * similarity, 0.05, None)
    return jnp.asarray(mask * strength_mS * modulation) * u.mS


class V1Network(brainstate.nn.Module):
    def __init__(self, angles, masks, regime):
        super().__init__()
        init = braintools.init.Constant(-60.0 * u.mV)
        neuron_args = dict(
            V_rest=-60.0 * u.mV, V_th=-50.0 * u.mV,
            V_reset=-60.0 * u.mV, tau=20.0 * u.ms,
            tau_ref=3.0 * u.ms, V_initializer=init,
        )
        self.e = brainpy.state.LIFRef(N_E, **neuron_args)
        self.i = brainpy.state.LIFRef(N_I, **neuron_args)
        ae, ai = angles[:N_E], angles[N_E:]

        w_ee = weight_matrix(ae, ae, masks["ee"], regime["ee_strength_mS"], regime["ee_specificity"])
        w_ei = weight_matrix(ae, ai, masks["ei"], regime["ei_strength_mS"], regime["ei_specificity"])
        w_ie = weight_matrix(ai, ae, masks["ie"], regime["inh_strength_mS"], regime["inh_specificity"])
        w_ii = weight_matrix(ai, ai, masks["ii"], regime["inh_strength_mS"], regime["inh_specificity"])

        self.ee = self._projection(w_ee, TAU_E, 0.0 * u.mV, self.e)
        self.ei = self._projection(w_ei, TAU_E, 0.0 * u.mV, self.i)
        self.ie = self._projection(w_ie, TAU_I, -80.0 * u.mV, self.e)
        self.ii = self._projection(w_ii, TAU_I, -80.0 * u.mV, self.i)
        # Separate channels make every forced event additive even when the
        # target also happens to spike spontaneously on the same time step.
        self.forced_ee = self._projection(w_ee, TAU_E, 0.0 * u.mV, self.e)
        self.forced_ei = self._projection(w_ei, TAU_E, 0.0 * u.mV, self.i)

    @staticmethod
    def _projection(weights, tau, reversal, post):
        return brainpy.state.AlignPostProj(
            comm=DenseEventComm(weights),
            # Concrete instances prevent the independent ordinary and forced
            # channels from being merged by AlignPost descriptor sharing.
            syn=brainpy.state.Expon(weights.shape[1], tau=tau),
            out=brainpy.state.COBA(E=reversal),
            post=post,
        )

    def update(self, t, drive_e, drive_i, forced_target_event):
        with brainstate.environ.context(t=t):
            e_spikes = self.e.get_spike() != 0.0
            i_spikes = self.i.get_spike() != 0.0
            forced_e = jnp.zeros(N_E, dtype=bool).at[TARGET].set(forced_target_event)
            self.ee(e_spikes)
            self.ei(e_spikes)
            self.ie(i_spikes)
            self.ii(i_spikes)
            self.forced_ee(forced_e)
            self.forced_ei(forced_e)
            self.e(drive_e)
            self.i(drive_i)
            return self.e.get_spike()


def make_masks(rng):
    masks = {
        "ee": rng.random((N_E, N_E)) < P_CONNECT,
        "ei": rng.random((N_E, N_I)) < P_CONNECT,
        "ie": rng.random((N_I, N_E)) < P_CONNECT,
        "ii": rng.random((N_I, N_I)) < P_CONNECT,
    }
    np.fill_diagonal(masks["ee"], False)
    np.fill_diagonal(masks["ii"], False)
    return masks


def stimulus_drive(vectors, stimulus_angle):
    stimulus = np.array([np.cos(stimulus_angle), np.sin(stimulus_angle)])
    tuning = np.maximum(0.0, vectors @ stimulus)
    return BASE_CURRENT + STIM_CURRENT * jnp.asarray(tuning)


def make_runner(net, times):
    @brainstate.transform.jit
    def run(drive_e, drive_i, events):
        return brainstate.transform.for_loop(net.update, times, drive_e, drive_i, events)

    return run


def restore_snapshot(net, snapshot):
    unexpected, missing = brainstate.nn.assign_state_values(net, snapshot)
    if unexpected or missing:
        raise RuntimeError(f"state restore mismatch: unexpected={unexpected}, missing={missing}")


def simulate_pair(net, snapshot, run, drive_e, drive_i, forced):
    def rollout(events):
        restore_snapshot(net, snapshot)
        return np.asarray(jax.block_until_ready(run(drive_e, drive_i, events)), dtype=bool)

    baseline = rollout(jnp.zeros_like(forced))
    perturbed = rollout(forced)
    return baseline, perturbed


def verify_reset_replay(net, snapshot, run, drive_e, drive_i, no_events):
    restore_snapshot(net, snapshot)
    first = np.asarray(jax.block_until_ready(run(drive_e, drive_i, no_events)), dtype=bool)
    restore_snapshot(net, snapshot)
    replay = np.asarray(jax.block_until_ready(run(drive_e, drive_i, no_events)), dtype=bool)
    if not np.array_equal(first, replay):
        raise RuntimeError("independent-rollout state reset did not replay baseline exactly")


def bin_curve(signal_corr, influence):
    rows = []
    for lo, hi in zip(CORR_EDGES[:-1], CORR_EDGES[1:]):
        take = (signal_corr >= lo) & (signal_corr < hi)
        values = influence[take]
        rows.append({
            "corr_low": float(lo), "corr_high": float(min(hi, 1.0)),
            "corr_center": float((lo + min(hi, 1.0)) / 2.0),
            "n": int(values.size),
            "mean_influence": float(np.nanmean(values)) if values.size else None,
            "sem_influence": float(np.nanstd(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else None,
        })
    return rows


def qualitative_score(rows):
    means = np.array([np.nan if row["mean_influence"] is None else row["mean_influence"] for row in rows])
    intermediate = means[3:5]
    high = means[-1]
    return bool(np.any(intermediate < 0.0) and high > 0.0 and high > np.nanmin(intermediate))


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"immutable result directory is not empty: {OUT}")
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    angles, vectors = preferred_vectors(rng)
    masks = make_masks(rng)
    stimulus_angles = np.linspace(0.0, 2.0 * np.pi, N_STIMULI, endpoint=False)
    n_steps = int(round(DURATION.to_decimal(u.ms) / DT.to_decimal(u.ms)))
    times = u.math.arange(0.0 * u.ms, DURATION, DT)
    forced = np.zeros(n_steps, dtype=bool)
    forced[PERTURB_STEPS] = True

    # Freeze all trial noise once and reuse it for every regime and paired run.
    noise_e = rng.normal(0.0, NOISE_SD.to_decimal(u.mA), (N_STIMULI, N_TRIALS, n_steps, N_E))
    noise_i = rng.normal(0.0, NOISE_SD.to_decimal(u.mA), (N_STIMULI, N_TRIALS, n_steps, N_I))
    all_curves, diagnostics = {}, {}

    with brainstate.environ.context(dt=DT):
        for name, regime in REGIMES.items():
            net = V1Network(angles, masks, regime)
            brainstate.nn.init_all_states(net)
            snapshot = {path: state.value for path, state in net.states().items()}
            run = make_runner(net, times)
            check_drive = stimulus_drive(vectors, stimulus_angles[0])
            check_e = u.math.broadcast_to(check_drive[:N_E], (n_steps, N_E))
            check_i = u.math.broadcast_to(check_drive[N_E:], (n_steps, N_I))
            verify_reset_replay(net, snapshot, run, check_e, check_i, jnp.zeros(n_steps, dtype=bool))
            baseline_counts = np.zeros((N_STIMULI, N_TRIALS, N_E))
            perturbed_counts = np.zeros_like(baseline_counts)
            for s, stimulus_angle in enumerate(stimulus_angles):
                mean_drive = stimulus_drive(vectors, stimulus_angle)
                for trial in range(N_TRIALS):
                    drive_e = mean_drive[:N_E] + jnp.asarray(noise_e[s, trial]) * u.mA
                    drive_i = mean_drive[N_E:] + jnp.asarray(noise_i[s, trial]) * u.mA
                    base, pert = simulate_pair(net, snapshot, run, drive_e, drive_i, jnp.asarray(forced))
                    baseline_counts[s, trial] = base.sum(axis=0)
                    perturbed_counts[s, trial] = pert.sum(axis=0)

            baseline_by_stimulus = baseline_counts.mean(axis=1)
            delta = (perturbed_counts - baseline_counts).mean(axis=(0, 1))
            influence = delta / PERTURB_STRENGTH
            target_tuning = baseline_by_stimulus[:, TARGET]
            signal_corr = np.array([
                np.corrcoef(target_tuning, baseline_by_stimulus[:, cell])[0, 1]
                if np.std(baseline_by_stimulus[:, cell]) > 0.0 and np.std(target_tuning) > 0.0 else np.nan
                for cell in range(N_E)
            ])
            valid = (np.arange(N_E) != TARGET) & np.isfinite(signal_corr)
            rows = bin_curve(signal_corr[valid], influence[valid])
            all_curves[name] = rows
            diagnostics[name] = {
                "baseline_mean_rate_hz": float(baseline_counts.mean() / DURATION.to_decimal(u.second)),
                "perturbed_mean_rate_hz": float(perturbed_counts.mean() / DURATION.to_decimal(u.second)),
                "target_baseline_spikes_per_trial": float(baseline_counts[:, :, TARGET].mean()),
                "valid_neighbor_count": int(valid.sum()),
                "qualitative_pattern": qualitative_score(rows),
            }
            np.savez_compressed(
                OUT / f"{name}_observables.npz", baseline_counts=baseline_counts,
                perturbed_counts=perturbed_counts, influence=influence,
                signal_correlation=signal_corr,
            )

    with (OUT / "influence_curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["regime", *next(iter(all_curves.values()))[0].keys()])
        writer.writeheader()
        for name, rows in all_curves.items():
            writer.writerows({"regime": name, **row} for row in rows)

    parameter_report = {
        "seed": SEED, "n_excitatory": N_E, "n_inhibitory": N_I,
        "dt_ms": DT.to_decimal(u.ms), "duration_ms": DURATION.to_decimal(u.ms),
        "n_stimuli": N_STIMULI, "n_trials": N_TRIALS,
        "connection_probability": P_CONNECT, "target_excitatory_index": TARGET,
        "base_current_mA": BASE_CURRENT.to_decimal(u.mA),
        "stimulus_current_mA": STIM_CURRENT.to_decimal(u.mA),
        "noise_sd_mA": NOISE_SD.to_decimal(u.mA),
        "perturbation_event_times_ms": (PERTURB_STEPS * DT.to_decimal(u.ms)).tolist(),
        "perturbation_strength_events": PERTURB_STRENGTH,
        "correlation_bin_edges": CORR_EDGES.tolist(), "regimes": REGIMES,
    }
    (OUT / "parameters.json").write_text(json.dumps(parameter_report, indent=2) + "\n")
    (OUT / "summary.json").write_text(json.dumps({"diagnostics": diagnostics, "curves": all_curves}, indent=2) + "\n")

    colors = {"inhibition_dominant": "#6b7280", "broad_inhibition": "#d97706", "specific_strong_ei": "#087f5b"}
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for name, rows in all_curves.items():
        x = np.array([row["corr_center"] for row in rows])
        y = np.array([np.nan if row["mean_influence"] is None else row["mean_influence"] for row in rows])
        sem = np.array([0.0 if row["sem_influence"] is None else row["sem_influence"] for row in rows])
        ax.errorbar(x, y, yerr=sem, marker="o", lw=1.8, capsize=3, color=colors[name], label=name.replace("_", " "))
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set(xlabel="Signal correlation with perturbed neuron", ylabel="Mean influence (spikes / delivered event)", title="Single-cell influence in feature-structured V1 E/I networks")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "influence_vs_signal_correlation.png", dpi=180)
    plt.close(fig)

    print(json.dumps(diagnostics, indent=2))
    print(f"results: {OUT}")


if __name__ == "__main__":
    main()
