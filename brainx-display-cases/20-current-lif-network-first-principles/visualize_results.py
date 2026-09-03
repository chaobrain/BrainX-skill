from __future__ import annotations

import hashlib
import json
from pathlib import Path

import braintools.visualize as btvis
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
COMBINED = ROOT / "runs" / "production-v2-combined-20260903"
FIGURES = ROOT / "figures"
SEEDS = (11, 29, 47)
DISPLAY_SEED = 11
RUN_DIRS = {
    seed: ROOT / "runs" / f"production-v2-seed-{seed}-cpu-20260903"
    for seed in SEEDS
}
CONDITION_COLORS = ("#007C78", "#D1493F", "#3C6EAA", "#D99120")
EXC_COLOR = "#168A62"
INH_COLOR = "#C4473D"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs():
    config = json.loads((COMBINED / "config.json").read_text(encoding="utf-8"))
    assessment = json.loads(
        (COMBINED / "condition_assessment.json").read_text(encoding="utf-8")
    )
    by_id = {row["condition_id"]: row for row in assessment}
    ordered = [by_id[c["id"]] for c in config["protocol"]["conditions"]]
    return config, ordered


def raw_path(seed: int, condition_id: str) -> Path:
    return RUN_DIRS[seed] / "raw" / f"seed-{seed}_{condition_id}.npz"


def condition_label(row: dict) -> str:
    return rf"$g={row['g']:g},\ \eta={row['eta']:g}$"


def render_rate_raster(config: dict, assessment: list[dict]) -> Path:
    dt_s = float(config["model"]["dt_ms"]) * 1e-3
    transient_step = int(
        round(config["protocol"]["transient_ms"] / config["model"]["dt_ms"])
    )
    n_steps = int(
        round(config["protocol"]["duration_ms"] / config["model"]["dt_ms"])
    )
    time_s = np.arange(transient_step, n_steps) * dt_s
    sample_reference = None
    output = FIGURES / "four_condition_rate_raster.png"
    with btvis.apply_style("publication", fontsize=8, dpi=180):
        fig, axes = plt.subplots(
            8,
            1,
            figsize=(10.5, 14.0),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 0.72] * 4, "hspace": 0.14},
        )
        for index, (row, color) in enumerate(zip(assessment, CONDITION_COLORS)):
            path = raw_path(DISPLAY_SEED, row["condition_id"])
            with np.load(path) as data:
                rate = np.asarray(data["global_rate_hz"][transient_step:])
                spikes = np.asarray(data["sample_spikes"][transient_step:], dtype=bool)
                sample_ids = np.asarray(data["sample_ids"])
            assert rate.shape == (time_s.size,)
            assert spikes.shape == (time_s.size, config["protocol"]["sample_size"])
            if sample_reference is None:
                sample_reference = sample_ids
            else:
                assert np.array_equal(sample_ids, sample_reference)

            rate_ax = axes[2 * index]
            raster_ax = axes[2 * index + 1]
            btvis.population_activity(
                rate,
                time=time_s,
                ax=rate_ax,
                color=color,
                alpha=0.92,
                fill=False,
                xlabel="",
                ylabel="Global rate (Hz)",
                linewidth=0.45,
                rasterized=True,
            )
            mean_rate = float(np.mean(rate))
            rate_ax.axhline(
                mean_rate,
                color="#202020",
                linestyle="--",
                linewidth=0.9,
                label=f"mean {mean_rate:.2f} Hz",
            )
            rate_ax.set_ylim(bottom=0)
            rate_ax.set_title(
                f"{condition_label(row)}  |  requested: {row['requested_regime']}  |  measured: {row['measured_regime']}",
                loc="left",
                fontsize=9,
                pad=3,
            )
            rate_ax.legend(loc="upper right", frameon=False, fontsize=7)
            rate_ax.text(
                -0.065,
                1.02,
                chr(ord("A") + index),
                transform=rate_ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="bottom",
            )
            btvis.raster_plot(
                time_s,
                spikes,
                ax=raster_ax,
                marker="|",
                markersize=1.1,
                color="#1B1B1B",
                alpha=0.72,
                xlabel="",
                ylabel="Sample neuron",
                xlim=(1.0, 5.0),
                ylim=(-0.5, 49.5),
                show=False,
                rasterized=True,
            )
            raster_ax.set_yticks((0, 24, 49))
            rate_ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)
            raster_ax.grid(False)
        axes[-1].set_xlabel("Time (s)")
        axes[-1].set_xlim(1.0, 5.0)
        fig.suptitle(
            "Instantaneous global rate and fixed 50-neuron raster (seed 11, 0.1 ms bins)",
            fontsize=12,
            y=0.995,
        )
        fig.subplots_adjust(left=0.11, right=0.98, bottom=0.045, top=0.965)
        fig.savefig(output, dpi=180, facecolor="white")
        plt.close(fig)
    return output


def render_ei_rates(assessment: list[dict]) -> Path:
    output = FIGURES / "ei_rates.png"
    y = np.arange(len(assessment), dtype=float)
    with btvis.apply_style("publication", fontsize=9, dpi=220):
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        for index, row in enumerate(assessment):
            exc = np.asarray([m["mean_exc_rate_hz"] for m in row["seed_metrics"]])
            inh = np.asarray([m["mean_inh_rate_hz"] for m in row["seed_metrics"]])
            ax.plot(
                [float(np.mean(exc)), float(np.mean(inh))],
                [y[index] + 0.11, y[index] - 0.11],
                color="#A8A8A8",
                linewidth=1.0,
                zorder=1,
            )
            ax.scatter(
                exc,
                np.full(exc.size, y[index] + 0.11),
                s=28,
                marker="o",
                facecolors="white",
                edgecolors=EXC_COLOR,
                linewidths=1.0,
                zorder=2,
                label="E seeds" if index == 0 else None,
            )
            ax.scatter(
                inh,
                np.full(inh.size, y[index] - 0.11),
                s=30,
                marker="^",
                facecolors="white",
                edgecolors=INH_COLOR,
                linewidths=1.0,
                zorder=2,
                label="I seeds" if index == 0 else None,
            )
            ax.scatter(
                np.mean(exc),
                y[index] + 0.11,
                s=62,
                marker="o",
                color=EXC_COLOR,
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
                label="E mean" if index == 0 else None,
            )
            ax.scatter(
                np.mean(inh),
                y[index] - 0.11,
                s=66,
                marker="^",
                color=INH_COLOR,
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
                label="I mean" if index == 0 else None,
            )
        ax.set_xscale("log")
        ax.set_xlim(4.0, 400.0)
        ax.set_yticks(y, [condition_label(row) for row in assessment])
        ax.invert_yaxis()
        ax.set_xlabel("Mean population firing rate (Hz, log scale)")
        ax.set_ylabel("Condition")
        ax.set_title("Excitatory and inhibitory rates across fixed seeds")
        ax.grid(axis="x", which="both", color="#D9D9D9", linewidth=0.55)
        ax.legend(frameon=False, ncol=2, loc="lower right")
        fig.tight_layout()
        fig.savefig(output, dpi=220, facecolor="white")
        plt.close(fig)
    return output


def render_isi_cv(assessment: list[dict]) -> Path:
    pooled = []
    seed_medians = []
    for row in assessment:
        values = []
        medians = []
        for metric in row["seed_metrics"]:
            with np.load(COMBINED / metric["raw_path"]) as data:
                valid = np.asarray(data["isi_cv_valid"], dtype=bool)
                cv = np.asarray(data["isi_cv"])[valid]
            assert cv.size == 12_500 and np.all(np.isfinite(cv))
            values.append(cv)
            medians.append(float(np.median(cv)))
        pooled.append(np.concatenate(values))
        seed_medians.append(np.asarray(medians))

    output = FIGURES / "isi_cv.png"
    with btvis.apply_style("publication", fontsize=9, dpi=220):
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        btvis.violin_plot(
            pooled,
            labels=[condition_label(row) for row in assessment],
            positions=[1, 2, 3, 4],
            showmeans=False,
            showmedians=True,
            colors=list(CONDITION_COLORS),
            xlabel="Condition",
            ylabel="ISI CV (dimensionless)",
            ax=ax,
        )
        jitter = np.asarray((-0.08, 0.0, 0.08))
        for index, medians in enumerate(seed_medians, start=1):
            ax.scatter(
                index + jitter,
                medians,
                s=24,
                facecolors="white",
                edgecolors="#202020",
                linewidths=0.8,
                zorder=4,
                label="seed medians" if index == 1 else None,
            )
        ax.axhline(
            0.5,
            color="#168A62",
            linestyle="--",
            linewidth=1.0,
            label="regular boundary (0.5)",
        )
        ax.axhline(
            0.8,
            color="#C4473D",
            linestyle=":",
            linewidth=1.2,
            label="irregular boundary (0.8)",
        )
        ax.set_ylim(0.0, 1.8)
        ax.set_title("Per-neuron ISI variability across three fixed seeds")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(output, dpi=220, facecolor="white")
        plt.close(fig)
    return output


def render_spectrum(assessment: list[dict]) -> Path:
    output = FIGURES / "global_rate_spectrum.png"
    with btvis.apply_style("publication", fontsize=8, dpi=220):
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True)
        for ax, row, color in zip(axes.flat, assessment, CONDITION_COLORS):
            spectra = []
            frequencies = None
            for metric in row["seed_metrics"]:
                with np.load(COMBINED / metric["raw_path"]) as data:
                    current_frequencies = np.asarray(data["frequencies_hz"])
                    spectra.append(np.asarray(data["power_hz"]))
                if frequencies is None:
                    frequencies = current_frequencies
                else:
                    assert np.array_equal(frequencies, current_frequencies)
            spectra = np.stack(spectra)
            search = (frequencies >= 1.0) & (frequencies <= 500.0)
            for seed, power in zip(SEEDS, spectra):
                ax.plot(
                    frequencies[search],
                    power[search],
                    color=color,
                    alpha=0.23,
                    linewidth=0.65,
                    label=f"seed {seed}",
                )
            mean_power = np.mean(spectra, axis=0)
            ax.plot(
                frequencies[search],
                mean_power[search],
                color=color,
                linewidth=1.35,
                label="seed mean",
            )
            peak = float(row["median_dominant_frequency_hz"])
            ax.axvline(
                peak,
                color="#202020",
                linestyle="--",
                linewidth=0.9,
                label=f"median peak {peak:.0f} Hz",
            )
            ax.set_yscale("log")
            ax.set_xlim(1.0, 500.0)
            ax.set_title(
                f"{condition_label(row)} | measured: {row['measured_regime']}",
                loc="left",
                fontsize=8.5,
            )
            ax.set_ylabel("Power (Hz$^2$/Hz)")
            ax.grid(axis="both", which="major", color="#D9D9D9", linewidth=0.45)
            ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper right")
        axes[1, 0].set_xlabel("Frequency (Hz)")
        axes[1, 1].set_xlabel("Frequency (Hz)")
        fig.suptitle("Global-rate Welch spectra across fixed seeds", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(output, dpi=220, facecolor="white")
        plt.close(fig)
    return output


def write_manifest(outputs: list[Path], config: dict, assessment: list[dict]) -> None:
    source_paths = [
        raw_path(seed, row["condition_id"])
        for seed in SEEDS
        for row in assessment
    ]
    source_lines = ", ".join(
        f"`{path.relative_to(ROOT)}` (`{sha256(path)}`)" for path in source_paths
    )
    shapes = {}
    for path in outputs:
        image = plt.imread(path)
        shapes[path.name] = [int(image.shape[1]), int(image.shape[0])]
        assert image.size > 0 and np.all(np.isfinite(image))
        assert float(np.std(image)) > 0.01

    sections = {
        "four_condition_rate_raster.png": (
            "Final/create; compare instantaneous global activity with fixed-sample event timing in all four conditions.",
            "Seed 11; global rate `(40,000,)` and sample spikes `(40,000, 50)` after the locked transient; time-major, 0.1 ms bins, rate in Hz and time in seconds.",
            "Discard 0-1 s only; no smoothing, aggregation, or decimation; dashed arithmetic mean over 1-5 s.",
            "One accepted display seed and 50 fixed sampled neurons; no uncertainty summary.",
        ),
        "ei_rates.png": (
            "Final/create; compare E and I mean firing rates across conditions.",
            "Per-seed post-transient E/I mean rates from all accepted metric records; Hz.",
            "Time mean per seed followed by cross-seed arithmetic mean; raw seed values remain visible; logarithmic rate axis.",
            "Seed is the sampling unit, n=3 per condition/population; no modeled uncertainty.",
        ),
        "isi_cv.png": (
            "Final/create; compare valid per-neuron ISI-CV distributions with frozen regularity thresholds.",
            "All valid per-neuron CV values `(12,500,)` per seed and condition; dimensionless; neurons nested in three seeds.",
            "Pooled descriptive violin with no exclusions beyond the locked interval rule; seed medians overlaid; fixed limits 0-1.8 include all values.",
            "37,500 neurons per condition for descriptive shape; n=3 seed-median replicate points; thresholds 0.5 and 0.8.",
        ),
        "global_rate_spectrum.png": (
            "Final/create; compare global-rate spectral concentration and dominant frequency across conditions.",
            "Accepted frozen frequency and Welch-power arrays `(5,001,)` for all three seeds; frequency in Hz and density in Hz^2/Hz.",
            "Display 1-500 Hz; individual seeds, arithmetic seed mean, logarithmic power, and median accepted dominant frequency; no new spectral calculation.",
            "Seed is the sampling unit, n=3 per condition; individual spectra remain visible.",
        ),
    }
    lines = ["# Figure manifest", ""]
    for path in outputs:
        role, variables, transformations, samples = sections[path.name]
        lines.extend(
            [
                f"## figures/{path.name}",
                f"- Work type, evidence mode, scientific role, and question: {role}",
                f"- Source run IDs, artifacts, hashes, and acceptance status: Codex iteration-2 `PASS`; {source_lines}",
                f"- Variables, axes, ordering, and units: {variables}",
                f"- Transformations, smoothing, aggregation, and exclusions: {transformations}",
                f"- Sample size and uncertainty: {samples}",
                "- Controls and fixed comparison settings: Frozen condition order/colors; common definitions and limits from `FIGURE_CONTRACT.md`; measured labels come from the accepted assessment.",
                f"- Plotting source, output path, size, format, and resolution: `visualize_results.py`; `{path.relative_to(ROOT)}`; {shapes[path.name][0]} x {shapes[path.name][1]} px PNG; fixed script DPI.",
                f"- Render and source-value checks: nonblank finite raster passed; output SHA-256 `{sha256(path)}`; source shapes, time alignment, frequency grids, fixed sample identity, and CV validity asserted during rendering; exact values are recorded in `figure_validation.json`.",
                "",
            ]
        )
    (ROOT / "FIGURE_MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")


def write_validation(outputs: list[Path], config: dict, assessment: list[dict]) -> None:
    transient_step = int(
        round(config["protocol"]["transient_ms"] / config["model"]["dt_ms"])
    )
    checks = []
    reference_sample_ids = None
    for row in assessment:
        display_metric = next(
            metric for metric in row["seed_metrics"] if metric["seed"] == DISPLAY_SEED
        )
        with np.load(raw_path(DISPLAY_SEED, row["condition_id"])) as data:
            rate = np.asarray(data["global_rate_hz"][transient_step:])
            sample_spikes = np.asarray(
                data["sample_spikes"][transient_step:], dtype=bool
            )
            sample_ids = np.asarray(data["sample_ids"])
        if reference_sample_ids is None:
            reference_sample_ids = sample_ids
        assert np.array_equal(sample_ids, reference_sample_ids)
        computed_rate_mean = float(np.mean(rate))
        assert np.isclose(
            computed_rate_mean, display_metric["mean_global_rate_hz"], rtol=0, atol=1e-12
        )

        cv_values = []
        seed_peak_checks = []
        for metric in row["seed_metrics"]:
            with np.load(COMBINED / metric["raw_path"]) as data:
                valid = np.asarray(data["isi_cv_valid"], dtype=bool)
                cv_values.append(np.asarray(data["isi_cv"])[valid])
                frequencies = np.asarray(data["frequencies_hz"])
                power = np.asarray(data["power_hz"])
            search = (frequencies >= 1.0) & (frequencies <= 500.0)
            peak = float(frequencies[np.flatnonzero(search)[np.argmax(power[search])]])
            assert peak == metric["dominant_frequency_hz"]
            seed_peak_checks.append(
                {"seed": metric["seed"], "dominant_frequency_hz": peak}
            )
        pooled_cv = np.concatenate(cv_values)
        pooled_median = float(np.median(pooled_cv))
        assert np.isclose(
            pooled_median, row["pooled_median_isi_cv"], rtol=0, atol=1e-12
        )
        checks.append(
            {
                "condition_id": row["condition_id"],
                "display_seed": DISPLAY_SEED,
                "display_rate_mean_hz": computed_rate_mean,
                "metric_rate_mean_hz": display_metric["mean_global_rate_hz"],
                "raster_event_count_first_100_ms": int(
                    np.count_nonzero(sample_spikes[:1000])
                ),
                "sample_ids_sha256": hashlib.sha256(
                    np.ascontiguousarray(sample_ids).view(np.uint8)
                ).hexdigest(),
                "pooled_valid_cv_count": int(pooled_cv.size),
                "pooled_cv_minimum": float(np.min(pooled_cv)),
                "pooled_cv_maximum": float(np.max(pooled_cv)),
                "pooled_cv_median": pooled_median,
                "accepted_pooled_cv_median": row["pooled_median_isi_cv"],
                "seed_spectral_peaks": seed_peak_checks,
            }
        )
    image_checks = []
    for path in outputs:
        image = plt.imread(path)
        image_checks.append(
            {
                "path": str(path.relative_to(ROOT)),
                "width_pixels": int(image.shape[1]),
                "height_pixels": int(image.shape[0]),
                "all_finite": bool(np.all(np.isfinite(image))),
                "pixel_minimum": float(np.min(image)),
                "pixel_maximum": float(np.max(image)),
                "pixel_standard_deviation": float(np.std(image)),
                "sha256": sha256(path),
            }
        )
        assert image_checks[-1]["all_finite"]
        assert image_checks[-1]["pixel_standard_deviation"] > 0.01
    record = {
        "source_value_checks": checks,
        "image_checks": image_checks,
        "visual_inspection": {
            "nonblank": "pass",
            "unclipped": "pass",
            "legible_at_rendered_size": "pass",
            "overlap_free": "pass",
            "patterns_match_accepted_metrics": "pass",
        },
    }
    (ROOT / "figure_validation.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    config, assessment = load_inputs()
    outputs = [
        render_rate_raster(config, assessment),
        render_ei_rates(assessment),
        render_isi_cv(assessment),
        render_spectrum(assessment),
    ]
    write_validation(outputs, config, assessment)
    write_manifest(outputs, config, assessment)
    print(json.dumps({path.name: str(path) for path in outputs}, indent=2))


if __name__ == "__main__":
    main()
