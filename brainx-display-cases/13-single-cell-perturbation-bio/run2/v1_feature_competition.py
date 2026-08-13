"""Feature-dependent single-neuron influence in a spatial V1 L2/3 E/I network.

All parameters, seeds, bins, and qualitative criteria were fixed before the
first perturbation outcome was inspected. This is a phenomenological test of
the Chettih-Harvey signatures, not a fit to their data.
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


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SEED = 20260813
N_E, N_I = 120, 30
CORTICAL_FIELD_UM = 500.0
VISUAL_FIELD_DEG = 20.0
DT = 0.5 * u.ms
DURATION = 500.0 * u.ms
N_ORIENTATIONS, N_POSITIONS, N_TRIALS = 8, 3, 3
BASE_CURRENT = 10.3 * u.mA
STIM_CURRENT = 2.8 * u.mA
NOISE_SD = 0.7 * u.mA
RF_SIGMA_DEG = 7.0
ORIENTATION_KAPPA = 2.0
TAU_EXC, TAU_INH = 5.0 * u.ms, 10.0 * u.ms
P_FLOOR, P_SCALE = 0.015, 0.42
EXC_DISTANCE_UM, INH_DISTANCE_UM = 120.0, 220.0
EXC_WEIGHT, INH_WEIGHT = 0.75 * u.mA, -2.6 * u.mA
PERTURB_TIMES_MS = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0])
PERTURB_STRENGTH = int(PERTURB_TIMES_MS.size)

CORTICAL_BINS = np.array([0.0, 75.0, 150.0, 250.0, 400.0, np.inf])
RF_BINS = np.array([0.0, 4.0, 8.0, 12.0, 20.0, np.inf])
SIGNAL_BINS = np.array([-1.0, -0.3, 0.1, 0.4, 0.7, 0.9, 1.000001])
ORIENTATION_BINS_DEG = np.array([0.0, 15.0, 30.0, 45.0, 67.5, 90.0001])


class DenseEventComm(brainstate.nn.Module):
    """Multiply binary spikes by a fixed heterogeneous BrainEvent matrix."""

    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def update(self, spikes):
        return brainevent.BinaryArray(spikes) @ self.weights


class SpatialV1(brainstate.nn.Module):
    def __init__(self, exc_weights, inh_weights, forced_weights):
        super().__init__()
        init = braintools.init.Constant(-60.0 * u.mV)
        neuron_args = dict(
            R=1.0 * u.ohm,
            tau=20.0 * u.ms,
            tau_ref=3.0 * u.ms,
            V_rest=-60.0 * u.mV,
            V_th=-50.0 * u.mV,
            V_reset=-60.0 * u.mV,
            V_initializer=init,
        )
        self.neurons = brainpy.state.LIFRef(N_E + N_I, **neuron_args)
        zero_current = braintools.init.Constant(0.0 * u.mA)
        self.exc_syn = brainpy.state.Expon(
            N_E + N_I, tau=TAU_EXC, g_initializer=zero_current
        )
        self.inh_syn = brainpy.state.Expon(
            N_E + N_I, tau=TAU_INH, g_initializer=zero_current
        )
        self.forced_syn = brainpy.state.Expon(
            N_E + N_I, tau=TAU_EXC, g_initializer=zero_current
        )
        self.exc_out = brainpy.state.CUBA(scale=1.0)
        self.inh_out = brainpy.state.CUBA(scale=1.0)
        self.forced_out = brainpy.state.CUBA(scale=1.0)
        self.neurons.add_current_input("exc", self.exc_out)
        self.neurons.add_current_input("inh", self.inh_out)
        self.neurons.add_current_input("forced", self.forced_out)
        self.exc_weights = exc_weights
        self.inh_weights = inh_weights
        self.forced_weights = forced_weights

    def update(self, t, drive, forced_event):
        with brainstate.environ.context(t=t):
            spikes = self.neurons.get_spike() != 0.0
            e_spikes = spikes[:N_E]
            i_spikes = spikes[N_E:]
            exc_current = self.exc_syn(
                brainevent.BinaryArray(e_spikes) @ self.exc_weights
            )
            inh_current = self.inh_syn(
                brainevent.BinaryArray(i_spikes) @ self.inh_weights
            )
            forced_current = self.forced_syn(forced_event * self.forced_weights)
            self.exc_out.bind_cond(exc_current)
            self.inh_out.bind_cond(inh_current)
            self.forced_out.bind_cond(forced_current)
            return self.neurons(drive)


def orientation_difference(a, b):
    difference = np.abs(a - b) % np.pi
    return np.minimum(difference, np.pi - difference)


def pairwise_distance(points_a, points_b):
    return np.linalg.norm(points_a[:, None, :] - points_b[None, :, :], axis=-1)


def feature_similarity(rf_a, ori_a, rf_b, ori_b):
    rf_distance = pairwise_distance(rf_a, rf_b)
    rf_similarity = np.exp(-0.5 * (rf_distance / (2.0 * RF_SIGMA_DEG)) ** 2)
    ori_similarity = 0.5 * (1.0 + np.cos(2.0 * (ori_a[:, None] - ori_b[None, :])))
    return rf_similarity * ori_similarity


def make_connectivity(rng, cortical, rf_centers, preferred, target):
    post_cortical = cortical
    post_rf = rf_centers
    post_preferred = preferred

    def pathway(pre_slice, length_um, base_weight):
        cortical_distance = pairwise_distance(cortical[pre_slice], post_cortical)
        similarity = feature_similarity(
            rf_centers[pre_slice], preferred[pre_slice], post_rf, post_preferred
        )
        distance_factor = np.exp(-cortical_distance / length_um)
        probability = np.clip(
            P_FLOOR + P_SCALE * distance_factor * (0.25 + 0.75 * similarity),
            0.0,
            0.95,
        )
        mask = rng.random(probability.shape) < probability
        magnitude = distance_factor * (0.35 + 0.65 * similarity)
        return mask, probability, mask * magnitude * base_weight.to_decimal(u.mA)

    exc_mask, exc_probability, exc_raw = pathway(slice(0, N_E), EXC_DISTANCE_UM, EXC_WEIGHT)
    inh_mask, inh_probability, inh_raw = pathway(slice(N_E, None), INH_DISTANCE_UM, INH_WEIGHT)
    np.fill_diagonal(exc_mask[:, :N_E], False)
    exc_raw[np.arange(N_E), np.arange(N_E)] = 0.0
    np.fill_diagonal(inh_mask[:, N_E:], False)
    inh_raw[np.arange(N_I), N_E + np.arange(N_I)] = 0.0
    forced_raw = exc_raw[target].copy()
    return {
        "exc_mask": exc_mask,
        "inh_mask": inh_mask,
        "exc_probability": exc_probability,
        "inh_probability": inh_probability,
        "exc_raw_mA": exc_raw,
        "inh_raw_mA": inh_raw,
        "forced_raw_mA": forced_raw,
    }


def make_stimuli():
    orientations = np.linspace(0.0, np.pi, N_ORIENTATIONS, endpoint=False)
    positions = np.array([[-6.0, 0.0], [0.0, 0.0], [6.0, 0.0]])
    return np.array([(x, y, ori) for x, y in positions for ori in orientations])


def stimulus_profile(rf_centers, preferred, stimulus):
    position = stimulus[:2]
    orientation = stimulus[2]
    spatial = np.exp(-0.5 * np.sum((rf_centers - position) ** 2, axis=1) / RF_SIGMA_DEG**2)
    angular = np.exp(ORIENTATION_KAPPA * (np.cos(2.0 * (preferred - orientation)) - 1.0))
    return spatial * angular


def make_runner(net, times):
    @brainstate.transform.jit
    def run(drive, forced):
        return brainstate.transform.for_loop(net.update, times, drive, forced)

    return run


def restore(net, snapshot):
    unexpected, missing = brainstate.nn.assign_state_values(net, snapshot)
    if unexpected or missing:
        raise RuntimeError(f"state restore mismatch: unexpected={unexpected}, missing={missing}")


def rollout(net, snapshot, run, drive, forced):
    restore(net, snapshot)
    spikes = np.asarray(jax.block_until_ready(run(drive, forced)), dtype=bool)
    return spikes[:, :N_E], spikes[:, N_E:]


def binned(values, influence, edges):
    rows = []
    finite = np.isfinite(values) & np.isfinite(influence)
    for low, high in zip(edges[:-1], edges[1:]):
        take = finite & (values >= low) & (values < high)
        sample = influence[take]
        rows.append({
            "low": float(low),
            "high": None if np.isinf(high) else float(high),
            "center": float(low if np.isinf(high) else 0.5 * (low + high)),
            "n": int(sample.size),
            "mean": float(sample.mean()) if sample.size else None,
            "sem": float(sample.std(ddof=1) / np.sqrt(sample.size)) if sample.size > 1 else None,
        })
    return rows


def connection_bins(strength):
    positive = strength[strength > 0.0]
    if positive.size < 3:
        return np.array([-1e-12, 1e-12, np.inf])
    q1, q2 = np.quantile(positive, [1.0 / 3.0, 2.0 / 3.0])
    return np.array([-1e-12, 1e-12, q1, q2, np.inf])


def classification(condition, partial):
    return "reproduced" if condition else ("partially reproduced" if partial else "not reproduced")


def json_edges(edges):
    return [None if np.isinf(value) else float(value) for value in edges]


def qualitative_results(overall_mean, curves, stimulus_delta, target_preferred):
    cortical = np.array([np.nan if row["mean"] is None else row["mean"] for row in curves["cortical_distance_um"]])
    signal = np.array([np.nan if row["mean"] is None else row["mean"] for row in curves["signal_correlation"]])
    stimulus_diff = np.rad2deg(orientation_difference(stimulus_delta[:, 0], target_preferred))
    near_preferred = stimulus_delta[stimulus_diff <= 22.5, 1]
    nonpreferred = stimulus_delta[stimulus_diff >= 67.5, 1]
    center_surround = (
        np.isfinite(cortical[:3]).all()
        and cortical[0] > 0.0
        and np.nanmin(cortical[1:3]) < 0.0
        and abs(cortical[-1]) < abs(np.nanmin(cortical[1:3]))
    )
    similarity_suppression = np.isfinite(signal[[0, -1]]).all() and signal[-1] < signal[0]
    preferred_suppression = (
        near_preferred.size > 0
        and nonpreferred.size > 0
        and near_preferred.mean() < nonpreferred.mean()
    )
    return {
        "mean_suppression": classification(overall_mean < 0.0, overall_mean <= 0.0),
        "spatial_center_surround": classification(center_surround, np.nanmin(cortical) < 0.0),
        "similarity_dependent_suppression": classification(similarity_suppression, np.nanmin(signal) < 0.0),
        "preferred_stimulus_suppression": classification(preferred_suppression, near_preferred.mean() < 0.0),
    }


def write_curve_csv(curves):
    with (OUT / "influence_curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["analysis", "low", "high", "center", "n", "mean", "sem"])
        writer.writeheader()
        for analysis, rows in curves.items():
            writer.writerows({"analysis": analysis, **row} for row in rows)


def plot_geometry(cortical, rf_centers, preferred, target):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    axes[0].scatter(cortical[:N_E, 0], cortical[:N_E, 1], s=14, c="#3377aa", alpha=0.65, label="excitatory")
    axes[0].scatter(cortical[N_E:, 0], cortical[N_E:, 1], s=18, c="#cc6677", alpha=0.75, label="inhibitory")
    axes[0].scatter(*cortical[target], s=90, facecolors="none", edgecolors="black", linewidths=1.8, label="target")
    axes[0].set(xlabel="Cortical x (um)", ylabel="Cortical y (um)", title="Cortical positions")
    axes[0].legend(frameon=False)
    colors = preferred / np.pi
    axes[1].scatter(rf_centers[:, 0], rf_centers[:, 1], c=colors, cmap="hsv", s=22, alpha=0.8)
    axes[1].scatter(*rf_centers[target], s=90, facecolors="none", edgecolors="black", linewidths=1.8)
    axes[1].set(xlabel="Visual x (deg)", ylabel="Visual y (deg)", title="RF centers; color = orientation")
    fig.tight_layout()
    fig.savefig(OUT / "network_and_receptive_fields.png", dpi=180)
    plt.close(fig)


def plot_influence(curves, stimulus_delta, target_preferred):
    names = [
        ("cortical_distance_um", "Cortical distance (um)"),
        ("rf_distance_deg", "RF-center distance (deg)"),
        ("signal_correlation", "Signal correlation"),
        ("orientation_difference_deg", "Orientation difference (deg)"),
        ("direct_connection_mA", "Target direct weight (mA)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.4))
    for ax, (name, label) in zip(axes.flat, names):
        rows = curves[name]
        x = np.array([row["center"] for row in rows])
        y = np.array([np.nan if row["mean"] is None else row["mean"] for row in rows])
        sem = np.array([0.0 if row["sem"] is None else row["sem"] for row in rows])
        ax.errorbar(x, y, yerr=sem, color="#176b5b", marker="o", lw=1.7, capsize=3)
        ax.axhline(0.0, color="black", lw=0.7)
        ax.set(xlabel=label, ylabel="Mean influence (spikes/AP)")
        for xi, yi, row in zip(x, y, rows):
            if np.isfinite(yi):
                ax.annotate(f"n={row['n']}", (xi, yi), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
    diff = np.rad2deg(orientation_difference(stimulus_delta[:, 0], target_preferred))
    order = np.argsort(diff)
    axes[1, 2].plot(diff[order], stimulus_delta[order, 1], "o-", color="#a44a3f", lw=1.5, ms=4)
    axes[1, 2].axhline(0.0, color="black", lw=0.7)
    axes[1, 2].set(xlabel="Stimulus-target orientation difference (deg)", ylabel="Population response change (spikes)")
    fig.suptitle("Single-neuron influence in spatially and functionally structured V1")
    fig.tight_layout()
    fig.savefig(OUT / "influence_relationships.png", dpi=180)
    plt.close(fig)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"immutable result directory is not empty: {OUT}")
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    n_total = N_E + N_I
    cortical = rng.uniform(0.0, CORTICAL_FIELD_UM, (n_total, 2))
    rf_centers = rng.uniform(-0.5 * VISUAL_FIELD_DEG, 0.5 * VISUAL_FIELD_DEG, (n_total, 2))
    preferred = rng.uniform(0.0, np.pi, n_total)
    target = int(np.argmin(np.linalg.norm(cortical[:N_E] - 0.5 * CORTICAL_FIELD_UM, axis=1)))
    connectivity = make_connectivity(rng, cortical, rf_centers, preferred, target)
    stimuli = make_stimuli()
    n_stimuli = stimuli.shape[0]
    n_steps = int(round(DURATION.to_decimal(u.ms) / DT.to_decimal(u.ms)))
    times = u.math.arange(0.0 * u.ms, DURATION, DT)
    forced = np.zeros(n_steps, dtype=np.float32)
    forced[np.rint(PERTURB_TIMES_MS / DT.to_decimal(u.ms)).astype(int)] = 1.0
    profiles = np.array([stimulus_profile(rf_centers, preferred, stimulus) for stimulus in stimuli])
    noise = rng.normal(0.0, NOISE_SD.to_decimal(u.mA), (n_stimuli, N_TRIALS, n_steps, n_total))

    exc_weights = jnp.asarray(connectivity["exc_raw_mA"]) * u.mA
    inh_weights = jnp.asarray(connectivity["inh_raw_mA"]) * u.mA
    forced_weights = jnp.asarray(connectivity["forced_raw_mA"]) * u.mA
    baseline_e = np.zeros((n_stimuli, N_TRIALS, N_E), dtype=np.int16)
    perturbed_e = np.zeros_like(baseline_e)
    baseline_i = np.zeros((n_stimuli, N_TRIALS, N_I), dtype=np.int16)
    perturbed_i = np.zeros_like(baseline_i)

    with brainstate.environ.context(dt=DT):
        net = SpatialV1(exc_weights, inh_weights, forced_weights)
        brainstate.nn.init_all_states(net)
        snapshot = {path: state.value for path, state in net.states().items()}
        run = make_runner(net, times)
        for stimulus_index in range(n_stimuli):
            mean_drive = BASE_CURRENT + STIM_CURRENT * jnp.asarray(profiles[stimulus_index])
            for trial in range(N_TRIALS):
                drive = mean_drive + jnp.asarray(noise[stimulus_index, trial]) * u.mA
                base_e_spikes, base_i_spikes = rollout(
                    net, snapshot, run, drive, jnp.zeros(n_steps)
                )
                pert_e_spikes, pert_i_spikes = rollout(
                    net, snapshot, run, drive, jnp.asarray(forced)
                )
                baseline_e[stimulus_index, trial] = base_e_spikes.sum(axis=0)
                perturbed_e[stimulus_index, trial] = pert_e_spikes.sum(axis=0)
                baseline_i[stimulus_index, trial] = base_i_spikes.sum(axis=0)
                perturbed_i[stimulus_index, trial] = pert_i_spikes.sum(axis=0)

        # Exact branch replay is a required causal invariant.
        check_drive = BASE_CURRENT + STIM_CURRENT * jnp.asarray(profiles[0])
        check = u.math.broadcast_to(check_drive, (n_steps, n_total))
        first_e, first_i = rollout(net, snapshot, run, check, jnp.zeros(n_steps))
        replay_e, replay_i = rollout(net, snapshot, run, check, jnp.zeros(n_steps))
        if not (np.array_equal(first_e, replay_e) and np.array_equal(first_i, replay_i)):
            raise RuntimeError("exact baseline replay failed")

    baseline_tuning = baseline_e.mean(axis=1)
    target_tuning = baseline_tuning[:, target]
    signal_corr = np.array([
        np.corrcoef(target_tuning, baseline_tuning[:, neuron])[0, 1]
        if np.std(target_tuning) > 0.0 and np.std(baseline_tuning[:, neuron]) > 0.0 else np.nan
        for neuron in range(N_E)
    ])
    target_dose = (
        perturbed_e[:, :, target].astype(float)
        + PERTURB_STRENGTH
        - baseline_e[:, :, target]
    )
    valid_dose = target_dose > 0.0
    paired_influence = np.full((n_stimuli, N_TRIALS, N_E), np.nan)
    paired_influence[valid_dose] = (
        perturbed_e[valid_dose].astype(float) - baseline_e[valid_dose]
    ) / target_dose[valid_dose, None]
    influence = np.nanmean(paired_influence, axis=(0, 1))
    neighbors = np.arange(N_E) != target
    cortical_distance = np.linalg.norm(cortical[:N_E] - cortical[target], axis=1)
    rf_distance = np.linalg.norm(rf_centers[:N_E] - rf_centers[target], axis=1)
    orientation_difference_deg = np.rad2deg(orientation_difference(preferred[:N_E], preferred[target]))
    direct_strength = connectivity["exc_raw_mA"][target, :N_E]
    valid = neighbors & np.isfinite(signal_corr)
    direct_edges = connection_bins(direct_strength[valid])
    curves = {
        "cortical_distance_um": binned(cortical_distance[valid], influence[valid], CORTICAL_BINS),
        "rf_distance_deg": binned(rf_distance[valid], influence[valid], RF_BINS),
        "signal_correlation": binned(signal_corr[valid], influence[valid], SIGNAL_BINS),
        "orientation_difference_deg": binned(orientation_difference_deg[valid], influence[valid], ORIENTATION_BINS_DEG),
        "direct_connection_mA": binned(direct_strength[valid], influence[valid], direct_edges),
    }
    stimulus_delta = np.column_stack((
        stimuli[:, 2],
        (perturbed_e[:, :, neighbors] - baseline_e[:, :, neighbors]).mean(axis=(1, 2)),
    ))
    overall = influence[valid]
    summary = {
        "baseline_exc_rate_hz": float(baseline_e.mean() / DURATION.to_decimal(u.second)),
        "baseline_inh_rate_hz": float(baseline_i.mean() / DURATION.to_decimal(u.second)),
        "perturbed_exc_rate_hz": float(perturbed_e.mean() / DURATION.to_decimal(u.second)),
        "mean_neighbor_influence": float(overall.mean()),
        "neighbor_influence_sem": float(overall.std(ddof=1) / np.sqrt(overall.size)),
        "suppressed_fraction": float(np.mean(overall < 0.0)),
        "enhanced_fraction": float(np.mean(overall > 0.0)),
        "unchanged_fraction": float(np.mean(overall == 0.0)),
        "target_index": target,
        "forced_target_action_potentials_per_trial": PERTURB_STRENGTH,
        "measured_target_output_increase_mean": float(target_dose.mean()),
        "measured_target_output_increase_min": float(target_dose.min()),
        "measured_target_output_increase_max": float(target_dose.max()),
        "valid_target_dose_trial_fraction": float(valid_dose.mean()),
        "valid_neighbor_count": int(valid.sum()),
        "qualitative_results": qualitative_results(overall.mean(), curves, stimulus_delta, preferred[target]),
    }
    parameters = {
        "seed": SEED,
        "n_excitatory": N_E,
        "n_inhibitory": N_I,
        "cortical_field_um": CORTICAL_FIELD_UM,
        "visual_field_deg": VISUAL_FIELD_DEG,
        "dt_ms": DT.to_decimal(u.ms),
        "duration_ms": DURATION.to_decimal(u.ms),
        "n_orientations": N_ORIENTATIONS,
        "n_positions": N_POSITIONS,
        "n_trials": N_TRIALS,
        "base_current_mA": BASE_CURRENT.to_decimal(u.mA),
        "stimulus_current_mA": STIM_CURRENT.to_decimal(u.mA),
        "noise_sd_mA": NOISE_SD.to_decimal(u.mA),
        "rf_sigma_deg": RF_SIGMA_DEG,
        "orientation_kappa": ORIENTATION_KAPPA,
        "p_floor": P_FLOOR,
        "p_scale": P_SCALE,
        "exc_distance_um": EXC_DISTANCE_UM,
        "inh_distance_um": INH_DISTANCE_UM,
        "exc_weight_mA": EXC_WEIGHT.to_decimal(u.mA),
        "inh_weight_mA": INH_WEIGHT.to_decimal(u.mA),
        "tau_exc_ms": TAU_EXC.to_decimal(u.ms),
        "tau_inh_ms": TAU_INH.to_decimal(u.ms),
        "perturbation_times_ms": PERTURB_TIMES_MS.tolist(),
        "analysis_bins": {
            "cortical_distance_um": json_edges(CORTICAL_BINS),
            "rf_distance_deg": json_edges(RF_BINS),
            "signal_correlation": json_edges(SIGNAL_BINS),
            "orientation_difference_deg": json_edges(ORIENTATION_BINS_DEG),
            "direct_connection_mA": json_edges(direct_edges),
        },
    }

    metadata = np.column_stack((
        np.arange(N_E), cortical_distance, rf_distance,
        orientation_difference_deg, signal_corr, direct_strength, influence,
    ))
    np.savetxt(
        OUT / "neuron_pair_metadata.csv",
        metadata,
        delimiter=",",
        header="neuron,cortical_distance_um,rf_distance_deg,orientation_difference_deg,signal_correlation,direct_connection_mA,influence",
        comments="",
    )
    np.savez_compressed(
        OUT / "paired_responses.npz",
        baseline_exc=baseline_e,
        perturbed_exc=perturbed_e,
        baseline_inh=baseline_i,
        perturbed_inh=perturbed_i,
        stimuli=stimuli,
        cortical_positions_um=cortical,
        rf_centers_deg=rf_centers,
        preferred_orientation_rad=preferred,
        target_output_increase=target_dose,
        paired_influence=paired_influence,
        exc_weights_mA=connectivity["exc_raw_mA"],
        inh_weights_mA=connectivity["inh_raw_mA"],
    )
    write_curve_csv(curves)
    (OUT / "parameters.json").write_text(json.dumps(parameters, indent=2, allow_nan=False) + "\n")
    (OUT / "summary.json").write_text(json.dumps({"summary": summary, "curves": curves}, indent=2, allow_nan=False) + "\n")
    plot_geometry(cortical, rf_centers, preferred, target)
    plot_influence(curves, stimulus_delta, preferred[target])
    print(json.dumps(summary, indent=2))
    print(f"results: {OUT}")


if __name__ == "__main__":
    main()
