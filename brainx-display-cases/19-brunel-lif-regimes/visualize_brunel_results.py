"""Render final figures from the review-passed Brunel LIF run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CASE_ROOT = Path(__file__).resolve().parent
RUN_ID = "20260830T142816+0800-validation-continuation-brunel"
RESULTS = CASE_ROOT / "runs" / RUN_ID / "results"
OUTPUT = CASE_ROOT / "figures" / "iteration-4-final"
DISPLAY_REPEAT = 0
DISPLAY_SEED = 1729
DISPLAY_WINDOW_MS = 200.0

CONDITIONS = (
    ("synchronous_regular", "Synchronous regular", "SR", 3.0, 2.0),
    ("fast_synchronous_irregular", "Fast synchronous irregular", "fast SI", 6.0, 4.0),
    ("asynchronous_irregular", "Asynchronous irregular", "AI", 5.0, 2.0),
    ("slow_synchronous_irregular", "Slow synchronous irregular", "slow SI", 4.5, 0.9),
)
CONDITION_COLORS = {
    "synchronous_regular": "#0072B2",
    "fast_synchronous_irregular": "#D55E00",
    "asynchronous_irregular": "#009E73",
    "slow_synchronous_irregular": "#CC79A7",
}
GROUP_COLORS = {"E": "#0072B2", "I": "#D55E00"}
GROUP_MARKERS = {"E": "o", "I": "^"}

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "savefig.dpi": 300,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open(encoding="ascii") as stream:
        return json.load(stream)


def verify_accepted_sources() -> tuple[dict, list[dict], dict]:
    manifest = read_json(RESULTS / "artifact-manifest.json")
    for entry in manifest["files"]:
        source = RESULTS / entry["path"]
        assert source.is_file(), source
        assert source.stat().st_size == entry["bytes"], source
        assert sha256(source) == entry["sha256"], source

    config = read_json(RESULTS / "config.json")
    metrics = read_json(RESULTS / "metrics.json")
    robustness = read_json(RESULTS / "robustness.json")
    assert config["repeat_seeds"][DISPLAY_REPEAT] == DISPLAY_SEED
    assert config["dt_ms"] == 0.1
    assert config["network"]["analysis_ms"] == 2000.0
    assert len(metrics) == len(CONDITIONS) * len(config["repeat_seeds"])
    return config, metrics, robustness


def raw_path(repeat: int, condition: str) -> Path:
    return RESULTS / "raw" / f"repeat-{repeat:02d}_{condition}.npz"


def metric_lookup(metrics: list[dict]) -> dict[tuple[int, str], dict]:
    return {(row["repeat"], row["expected_state"]): row for row in metrics}


def load_raw_runs(config: dict) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    loaded = {}
    probe = None
    expected_steps = round(config["network"]["analysis_ms"] / config["dt_ms"])
    for repeat in range(len(config["repeat_seeds"])):
        for condition, *_ in CONDITIONS:
            with np.load(raw_path(repeat, condition), allow_pickle=False) as source:
                run = {key: source[key] for key in source.files}
            assert run["raster"].shape == (expected_steps, 50)
            assert run["exc_counts"].shape == (expected_steps,)
            assert run["inh_counts"].shape == (expected_steps,)
            if probe is None:
                probe = run["probe_indices"]
            else:
                np.testing.assert_array_equal(run["probe_indices"], probe)
            loaded[(repeat, condition)] = run
    assert np.count_nonzero(probe < config["network"]["n_exc"]) == 40
    assert np.count_nonzero(probe >= config["network"]["n_exc"]) == 10
    return loaded


def condition_title(label: str, g: float, eta: float, classified: str) -> str:
    measured = classified.replace("_", " ")
    return f"{label}\n$g={g:g}$, $\\eta={eta:g}$ | measured: {measured}"


def save_figure(fig, name: str) -> Path:
    path = OUTPUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_rate_and_raster(config: dict, metrics: dict, raw: dict) -> Path:
    dt_ms = config["dt_ms"]
    dt_s = dt_ms / 1000.0
    n_total = config["network"]["n_exc"] + config["network"]["n_inh"]
    time_ms = np.arange(raw[(DISPLAY_REPEAT, CONDITIONS[0][0])]["raster"].shape[0]) * dt_ms
    display_steps = round(DISPLAY_WINDOW_MS / dt_ms)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(15.8, 6.2),
        sharex="col",
        gridspec_kw={"height_ratios": (1.0, 1.25)},
        constrained_layout=True,
    )
    for column, (condition, label, _, g, eta) in enumerate(CONDITIONS):
        row = metrics[(DISPLAY_REPEAT, condition)]
        run = raw[(DISPLAY_REPEAT, condition)]
        color = CONDITION_COLORS[condition]
        global_rate = (run["exc_counts"] + run["inh_counts"]) / (n_total * dt_s)
        np.testing.assert_allclose(global_rate.mean(), row["overall_firing_rate_hz"], rtol=0, atol=1e-10)

        rate_ax = axes[0, column]
        rate_ax.plot(time_ms[:display_steps], global_rate[:display_steps], color=color, linewidth=0.55)
        rate_ax.axhline(
            row["overall_firing_rate_hz"],
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"mean {row['overall_firing_rate_hz']:.1f} Hz",
        )
        rate_ax.set_ylim(0.0, max(1.0, float(global_rate.max()) * 1.04))
        rate_ax.set_title(condition_title(label, g, eta, row["classified_state"]), pad=7)
        rate_ax.legend(loc="upper right", frameon=False, handlelength=1.6)
        rate_ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        if column == 0:
            rate_ax.set_ylabel("Global rate (Hz)\n0.1 ms bins")

        raster_ax = axes[1, column]
        steps, neurons = np.nonzero(run["raster"][:display_steps])
        excitatory = neurons < 40
        raster_ax.scatter(
            steps[excitatory] * dt_ms,
            neurons[excitatory] + 1,
            s=1.1,
            color=GROUP_COLORS["E"],
            marker=".",
            linewidths=0,
            rasterized=True,
        )
        raster_ax.scatter(
            steps[~excitatory] * dt_ms,
            neurons[~excitatory] + 1,
            s=1.6,
            color=GROUP_COLORS["I"],
            marker=".",
            linewidths=0,
            rasterized=True,
        )
        raster_ax.axhline(40.5, color="#666666", linewidth=0.7)
        raster_ax.set(xlim=(0.0, DISPLAY_WINDOW_MS), ylim=(0.5, 50.5), xlabel="Time (ms)")
        raster_ax.set_yticks((1, 20, 40, 41, 50))
        if column == 0:
            raster_ax.set_ylabel("Fixed probe row\n(40 E, 10 I)")
    fig.suptitle(
        f"Current-based LIF network activity | seed {DISPLAY_SEED} | first {DISPLAY_WINDOW_MS:g} ms; dashed mean uses full 2 s",
        fontsize=12,
    )
    return save_figure(fig, "four-condition-rate-raster.png")


def plot_repeat_summary(metrics: dict, value_keys: tuple[str, str], ylabel: str, name: str, log_y: bool) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    repeats = np.arange(5)
    jitter = np.linspace(-0.055, 0.055, repeats.size)
    for index, (condition, _, short, g, eta) in enumerate(CONDITIONS):
        e_values = np.array([metrics[(repeat, condition)][value_keys[0]] for repeat in repeats])
        i_values = np.array([metrics[(repeat, condition)][value_keys[1]] for repeat in repeats])
        for repeat in repeats:
            ax.plot(
                [index - 0.12 + jitter[repeat], index + 0.12 + jitter[repeat]],
                [e_values[repeat], i_values[repeat]],
                color="#A0A0A0",
                linewidth=0.65,
                zorder=1,
            )
        for group, offset, values in (("E", -0.12, e_values), ("I", 0.12, i_values)):
            ax.scatter(
                index + offset + jitter,
                values,
                s=24,
                marker=GROUP_MARKERS[group],
                color=GROUP_COLORS[group],
                edgecolor="white",
                linewidth=0.35,
                zorder=2,
                label=group if index == 0 else None,
            )
            ax.scatter(
                index + offset,
                np.median(values),
                s=78,
                marker=GROUP_MARKERS[group],
                facecolor="white",
                edgecolor=GROUP_COLORS[group],
                linewidth=1.5,
                zorder=3,
            )
    ax.set_xticks(
        np.arange(len(CONDITIONS)),
        [f"{short}\n$g={g:g}, \\eta={eta:g}$" for _, _, short, g, eta in CONDITIONS],
    )
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.55, len(CONDITIONS) - 0.45)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(title="Population", frameon=False, ncol=2, loc="upper right")
    ax.set_title("Five fixed seeds; small symbols are repeats, open symbols are medians")
    if log_y:
        ax.set_yscale("log")
    else:
        ax.axhline(0.5, color="#555555", linestyle=":", linewidth=1.0, label="regular boundary")
        ax.axhline(0.7, color="#555555", linestyle="--", linewidth=1.0, label="irregular boundary")
        ax.set_ylim(-0.025, 1.05)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="Population / criterion", frameon=False, ncol=2, loc="upper left")
    return save_figure(fig, name)


def plot_spectra(config: dict, robustness: dict, raw: dict) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.8), sharex=True, constrained_layout=True)
    for ax, (condition, label, _, g, eta) in zip(axes.flat, CONDITIONS):
        spectra = []
        for repeat in range(len(config["repeat_seeds"])):
            run = raw[(repeat, condition)]
            frequencies = run["frequencies_hz"].astype(np.float64)
            power = run["power_hz"].astype(np.float64)
            mask = (frequencies >= 1.0) & (frequencies <= 1000.0) & (power > 0.0)
            ax.plot(frequencies[mask], power[mask], color=CONDITION_COLORS[condition], alpha=0.22, linewidth=0.65)
            spectra.append(power)
        spectrum_median = np.median(np.stack(spectra), axis=0)
        mask = (frequencies >= 1.0) & (frequencies <= 1000.0) & (spectrum_median > 0.0)
        ax.plot(frequencies[mask], spectrum_median[mask], color=CONDITION_COLORS[condition], linewidth=1.7, label="median PSD")
        dominant = robustness[condition]["dominant_frequency_hz"]
        ax.axvline(dominant, color="black", linestyle="--", linewidth=1.0, label=f"median peak {dominant:.1f} Hz")
        ax.set(xscale="log", yscale="log", xlim=(1.0, 1000.0))
        ax.set_title(condition_title(label, g, eta, robustness[condition]["classified_state"]))
        ax.grid(which="major", color="#D9D9D9", linewidth=0.5)
        ax.legend(frameon=False, loc="best")
    axes[0, 0].set_ylabel("Global-rate PSD (Hz$^2$/Hz)")
    axes[1, 0].set_ylabel("Global-rate PSD (Hz$^2$/Hz)")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Global-rate spectra across five fixed seeds", fontsize=12)
    return save_figure(fig, "global-rate-spectrum.png")


def verify_spectral_peaks(config: dict, metrics: dict, raw: dict) -> None:
    for repeat in range(len(config["repeat_seeds"])):
        for condition, *_ in CONDITIONS:
            run = raw[(repeat, condition)]
            frequency = run["frequencies_hz"]
            power = run["power_hz"]
            mask = (frequency >= 1.0) & (frequency <= 1000.0)
            peak = float(frequency[np.flatnonzero(mask)[np.argmax(power[mask])]])
            assert peak == metrics[(repeat, condition)]["dominant_frequency_hz"]


def inspect_png(path: Path) -> dict:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    assert rgb.shape[0] >= 1000 and rgb.shape[1] >= 1600
    assert float(rgb.std()) > 5.0
    assert np.unique(rgb.reshape(-1, 3), axis=0).shape[0] > 32
    return {
        "path": str(path.relative_to(CASE_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "width_px": int(rgb.shape[1]),
        "height_px": int(rgb.shape[0]),
        "rgb_std": float(rgb.std()),
    }


def write_figure_manifest(checks: list[dict]) -> None:
    source_manifest_hash = sha256(RESULTS / "artifact-manifest.json")
    review_hash = sha256(CASE_ROOT / "reviews" / "iteration-4-review.md")
    shared = (
        f"- Source run: `{RUN_ID}` (accepted iteration-4 evidence).\n"
        f"- Source manifest SHA-256: `{source_manifest_hash}`.\n"
        f"- PASS review SHA-256: `{review_hash}`.\n"
        "- Output: final evidence, report destination, PNG at 300 dpi.\n"
    )
    roles = {
        "four-condition-rate-raster.png": (
            "Compare unsmoothed 0.1 ms global rate and fixed 50-neuron spike timing across conditions.",
            "Repeat 0 (seed 1729), common first 200 ms of the analysis interval; dashed mean uses the full 2,000 ms; no smoothing.",
        ),
        "ei-rates.png": (
            "Compare excitatory and inhibitory firing rates across conditions and seeds.",
            "Five paired repeat means per population with their medians; logarithmic rate axis.",
        ),
        "isi-cv.png": (
            "Compare E/I spike-train irregularity with the locked regular and irregular boundaries.",
            "Five paired repeat mean CVs per population with medians; source eligibility requires at least four spikes.",
        ),
        "global-rate-spectrum.png": (
            "Compare global-rate spectral structure and report each aggregate dominant frequency.",
            "Raw Welch PSD for all five repeats plus pointwise median; 1-1,000 Hz display, log-log axes, no normalization.",
        ),
    }
    lines = ["# Figure manifest", ""]
    for check in checks:
        filename = Path(check["path"]).name
        question, transform = roles[filename]
        lines.extend(
            [
                f"## `{check['path']}`",
                f"- Work type, evidence mode, role, and question: create; final; {question}",
                shared.rstrip(),
                f"- Variables and transformations: {transform}",
                "- Comparisons: fixed condition order and encodings; requested and measured labels remain distinct.",
                f"- Render checks: {check['width_px']} x {check['height_px']} px; RGB standard deviation {check['rgb_std']:.3f}; nonblank and numerically cross-checked against accepted source arrays.",
                f"- Output SHA-256: `{check['sha256']}`; {check['bytes']} bytes.",
                "",
            ]
        )
    (OUTPUT / "FIGURE_MANIFEST.md").write_text("\n".join(lines), encoding="ascii")
    (OUTPUT / "render-checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="ascii")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config, metric_rows, robustness = verify_accepted_sources()
    metrics = metric_lookup(metric_rows)
    raw = load_raw_runs(config)
    verify_spectral_peaks(config, metrics, raw)

    with plt.rc_context(STYLE):
        paths = [
            plot_rate_and_raster(config, metrics, raw),
            plot_repeat_summary(
                metrics,
                ("exc_firing_rate_hz", "inh_firing_rate_hz"),
                "Mean firing rate (Hz)",
                "ei-rates.png",
                log_y=True,
            ),
            plot_repeat_summary(
                metrics,
                ("exc_isi_cv_mean", "inh_isi_cv_mean"),
                "Mean ISI CV",
                "isi-cv.png",
                log_y=False,
            ),
            plot_spectra(config, robustness, raw),
        ]
    checks = [inspect_png(path) for path in paths]
    write_figure_manifest(checks)
    for check in checks:
        print(json.dumps(check, sort_keys=True))


if __name__ == "__main__":
    main()
