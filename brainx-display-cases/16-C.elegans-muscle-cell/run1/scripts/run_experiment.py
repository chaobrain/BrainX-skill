"""Run the frozen C. elegans muscle fitting and held-out experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np
from scipy.stats import qmc

import brainstate
import brainunit as u

from cellegans_hh.data import (
    CURRENT_BY_TRACE_PA,
    FIT_TRACE,
    TEST_TRACES,
    current_protocol,
    initial_voltage,
    load_experiment,
)
from cellegans_hh.inference import (
    InferenceProblem,
    fit_passive_control,
    fit_problem,
    spike_times_ms,
    trace_metrics,
)
from cellegans_hh.model import PARAMETER_SPECS, parameter_bounds, simulate


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fit_to_dict(result):
    return {
        "seed": result.seed,
        "parameters": {
            spec.name: {"value": float(result.parameters[i]), "unit": str(spec.unit)}
            for i, spec in enumerate(PARAMETER_SPECS)
        },
        "parameter_vector": result.parameters.tolist(),
        "loss": result.loss,
        "success": result.success,
        "message": result.message,
        "nfev": result.nfev,
        "nit": result.nit,
        "candidate_batches": [asdict(record) for record in result.records],
        "initial_parameters": result.initial_parameters.tolist(),
        "initial_loss": result.initial_loss if np.isfinite(result.initial_loss) else None,
        "initial_components": result.initial_components,
        "loss_closed": result.loss_closed,
        "closure_reason": result.closure_reason,
        "candidate_evaluations": result.candidate_evaluations,
    }


def fit_one_target(config, time_axis, target, current, initial_v, label):
    problem = InferenceProblem(time_axis, target, current, initial_v)
    results = []
    for seed in config["optimizer"]["seeds"]:
        print(f"FIT_START label={label} seed={seed}", flush=True)
        result = fit_problem(
            problem,
            seed=int(seed),
            maxiter=int(config["optimizer"]["maxiter"]),
            popsize=int(config["optimizer"]["popsize"]),
            plateau_generations=int(config["optimizer"]["plateau_generations"]),
            plateau_tolerance=float(config["optimizer"]["plateau_tolerance"]),
        )
        results.append(result)
        print(
            f"FIT_DONE label={label} seed={seed} loss={result.loss:.6f} "
            f"nit={result.nit} nfev={result.nfev} status={result.message}",
            flush=True,
        )
    selected = min(results, key=lambda result: (result.loss, result.seed))
    return results, selected


def assess_predictions(metrics_by_trace):
    held_out = [metrics_by_trace[str(trace)] for trace in TEST_TRACES]
    count_passes = sum(metric["spike_count_error"] <= 1 for metric in held_out)
    latency_passes = sum(
        metric["first_spike_error_ms"] is not None
        and metric["first_spike_error_ms"] <= 15.0
        for metric in held_out
    )
    ordered = [metrics_by_trace[str(trace)] for trace in sorted(CURRENT_BY_TRACE_PA)]
    predicted_counts = [metric["predicted_spike_count"] for metric in ordered]
    predicted_latencies = [metric["predicted_first_spike_ms"] for metric in ordered]
    count_monotone = all(a <= b for a, b in zip(predicted_counts, predicted_counts[1:]))
    latency_monotone = all(
        a is not None and b is not None and a >= b
        for a, b in zip(predicted_latencies, predicted_latencies[1:])
    )
    return {
        "held_out_count_passes": count_passes,
        "held_out_latency_passes": latency_passes,
        "predicted_spike_counts_15_20_25_30_pA": predicted_counts,
        "predicted_first_spikes_ms_15_20_25_30_pA": predicted_latencies,
        "spike_count_monotone": count_monotone,
        "first_spike_latency_monotone": latency_monotone,
        "predictive_consistency_pass": (
            count_passes >= 2 and latency_passes >= 2 and count_monotone and latency_monotone
        ),
    }


def recovery_truths(config):
    count = int(config["recovery"]["truth_count"])
    if count == 0:
        return np.empty((0, len(PARAMETER_SPECS)))
    bounds = np.asarray(parameter_bounds())
    margin = float(config["recovery"]["domain_margin_fraction"])
    low = bounds[:, 0] + margin * (bounds[:, 1] - bounds[:, 0])
    high = bounds[:, 1] - margin * (bounds[:, 1] - bounds[:, 0])
    sampler = qmc.LatinHypercube(d=bounds.shape[0], seed=int(config["recovery"]["truth_seed"]))
    return qmc.scale(sampler.random(count), low, high)


def classify_recovery(config, recovery_rows):
    if not recovery_rows:
        return {}
    bounds = np.asarray(parameter_bounds())
    errors = np.asarray(
        [np.abs(np.asarray(row["selected_parameters"]) - np.asarray(row["truth"])) for row in recovery_rows]
    )
    normalized = errors / (bounds[:, 1] - bounds[:, 0])
    median_limit = float(config["recovery"]["median_normalized_error_limit"])
    max_limit = float(config["recovery"]["max_normalized_error_limit"])
    result = {}
    for index, spec in enumerate(PARAMETER_SPECS):
        median_error = float(np.median(normalized[:, index]))
        maximum_error = float(np.max(normalized[:, index]))
        passes_error_gate = median_error <= median_limit and maximum_error <= max_limit
        result[spec.name] = {
            "median_absolute_error": float(np.median(errors[:, index])),
            "median_normalized_error": median_error,
            "max_normalized_error": maximum_error,
            "passes_error_gate": bool(passes_error_gate),
            "classification": "non-identifiable-under-this-protocol",
        }
    return result


def recovery_tradeoffs(recovery_rows):
    if len(recovery_rows) < 2:
        return {}
    errors = np.asarray(
        [np.asarray(row["selected_parameters"]) - np.asarray(row["truth"]) for row in recovery_rows]
    )
    correlation = np.corrcoef(errors, rowvar=False)
    boundary_rate = {}
    for index, spec in enumerate(PARAMETER_SPECS):
        estimates = np.asarray([row["selected_parameters"][index] for row in recovery_rows])
        width = spec.upper - spec.lower
        boundary_rate[spec.name] = float(
            np.mean((estimates - spec.lower) / width <= 0.01)
            + np.mean((spec.upper - estimates) / width <= 0.01)
        )
    return {
        "parameter_order": [spec.name for spec in PARAMETER_SPECS],
        "signed_error_correlation": correlation.tolist(),
        "boundary_rate": boundary_rate,
        "fit_failure_rate": float(
            np.mean([not start["loss_closed"] for row in recovery_rows for start in row["starts"]])
        ),
    }


def state_validation(parameter_values, current, initial_v):
    states = simulate(parameter_values, current, initial_v, return_states=True)
    result = {}
    for name, values in states.items():
        if name == "voltage":
            numeric = np.asarray(values.to_decimal(u.mV))
            unit = "mV"
        elif name == "calcium_i":
            numeric = np.asarray(values.to_decimal(u.mM))
            unit = "mM"
        else:
            numeric = np.asarray(values)
            unit = "dimensionless"
        entry = {
            "finite": bool(np.isfinite(numeric).all()),
            "minimum": float(np.nanmin(numeric)),
            "maximum": float(np.nanmax(numeric)),
            "unit": unit,
        }
        if unit == "dimensionless":
            entry["in_gate_range"] = bool(numeric.min() >= -1e-6 and numeric.max() <= 1.0 + 1e-6)
        if name == "calcium_i":
            entry["positive"] = bool(numeric.min() > 0.0)
        result[name] = entry
    return result


def objective_profiles(problem, selected_parameters):
    profiles = {}
    selected_parameters = np.asarray(selected_parameters)
    for index, spec in enumerate(PARAMETER_SPECS):
        values = np.asarray(
            [
                spec.lower,
                0.5 * (spec.lower + selected_parameters[index]),
                selected_parameters[index],
                0.5 * (selected_parameters[index] + spec.upper),
                spec.upper,
            ]
        )
        candidates = np.repeat(selected_parameters[None, :], values.size, axis=0)
        candidates[:, index] = values
        components = problem.loss_components(candidates)
        profiles[spec.name] = {
            "unit": str(spec.unit),
            "values": values.tolist(),
            "components": {name: component.tolist() for name, component in components.items()},
        }
    return profiles


def artifact_manifest(run_dir: Path):
    entries = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {
            "artifact_manifest.json",
            "exit_code",
            "run.log",
            "status.json",
        }:
            continue
        entries.append(
            {
                "path": str(path.relative_to(run_dir)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run_dir = Path(config["run_dir"])
    raw_dir = run_dir / "raw"
    metrics_dir = run_dir / "metrics"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if jax.default_backend() != config["backend"]:
        raise RuntimeError(f"Expected backend {config['backend']}, got {jax.default_backend()}.")
    brainstate.random.seed(int(config["simulation_seed"]))
    started = time.time()
    print(f"RUN_START id={config['run_id']} level={config['run_level']}", flush=True)

    data = load_experiment(config["data_path"], stride=int(config["data_stride"]))
    target = data.voltage_by_trace[FIT_TRACE]
    train_initial_v = initial_voltage(target, data.time)
    train_current = current_protocol(data.time, data.current_by_trace[FIT_TRACE])
    fit_results, selected = fit_one_target(
        config, data.time, target, train_current, train_initial_v, "observed-trace-8"
    )
    write_json(raw_dir / "fit_starts.json", [fit_to_dict(result) for result in fit_results])
    write_json(raw_dir / "fitted_parameters.json", fit_to_dict(selected))
    write_json(
        raw_dir / "objective_profiles.json",
        objective_profiles(
            InferenceProblem(data.time, target, train_current, train_initial_v), selected.parameters
        ),
    )

    passive = fit_passive_control(
        InferenceProblem(data.time, target, train_current, train_initial_v),
        seed=int(config["passive_seed"]),
    )
    write_json(raw_dir / "passive_fit.json", fit_to_dict(passive))

    arrays = {"time_ms": np.asarray(data.time.to_decimal(u.ms))}
    per_start_arrays = {"time_ms": arrays["time_ms"]}
    metrics_by_trace = {}
    per_start_metrics = {}
    prestim_spikes_by_trace = {}
    all_predictions = {}
    for trace in sorted(CURRENT_BY_TRACE_PA):
        observed = data.voltage_by_trace[trace]
        prediction = simulate(
            selected.parameters,
            current_protocol(data.time, data.current_by_trace[trace]),
            initial_voltage(observed, data.time),
        )
        arrays[f"observed_trace_{trace}_mV"] = np.asarray(observed.to_decimal(u.mV))
        arrays[f"predicted_trace_{trace}_mV"] = np.asarray(prediction.to_decimal(u.mV))
        arrays[f"residual_trace_{trace}_mV"] = arrays[f"predicted_trace_{trace}_mV"] - arrays[
            f"observed_trace_{trace}_mV"
        ]
        metrics_by_trace[str(trace)] = trace_metrics(data.time, observed, prediction)
        all_predictions[trace] = prediction
        prestim_spikes_by_trace[str(trace)] = int(
            spike_times_ms(
                arrays["time_ms"],
                arrays[f"predicted_trace_{trace}_mV"],
                start_ms=None,
                end_ms=50.0,
            ).size
        )
    for result in fit_results:
        seed_metrics = {}
        for trace in sorted(CURRENT_BY_TRACE_PA):
            observed = data.voltage_by_trace[trace]
            prediction = simulate(
                result.parameters,
                current_protocol(data.time, data.current_by_trace[trace]),
                initial_voltage(observed, data.time),
            )
            per_start_arrays[f"seed_{result.seed}_trace_{trace}_mV"] = np.asarray(
                prediction.to_decimal(u.mV)
            )
            seed_metrics[str(trace)] = trace_metrics(data.time, observed, prediction)
        per_start_metrics[str(result.seed)] = seed_metrics
    np.savez_compressed(raw_dir / "predictions.npz", **arrays)
    np.savez_compressed(raw_dir / "per_start_predictions.npz", **per_start_arrays)
    write_json(metrics_dir / "per_start_metrics.json", per_start_metrics)

    passive_prediction = simulate(passive.parameters, train_current, train_initial_v)
    passive_metrics = trace_metrics(data.time, target, passive_prediction)
    zero_prediction = simulate(
        selected.parameters,
        u.math.zeros_like(train_current),
        train_initial_v,
    )
    zero_spikes = spike_times_ms(
        np.asarray(data.time.to_decimal(u.ms)),
        np.asarray(zero_prediction.to_decimal(u.mV)),
        start_ms=None,
        end_ms=50.0,
    )
    boundary_checks = {}
    full_state_checks = {}
    for label, values in (
        ("nominal", np.asarray([spec.initial for spec in PARAMETER_SPECS])),
        ("lower", np.asarray([spec.lower for spec in PARAMETER_SPECS])),
        ("upper", np.asarray([spec.upper for spec in PARAMETER_SPECS])),
    ):
        output = np.asarray(simulate(values, train_current, train_initial_v).to_decimal(u.mV))
        boundary_checks[label] = {
            "finite": bool(np.isfinite(output).all()),
            "minimum_mV": float(np.nanmin(output)),
            "maximum_mV": float(np.nanmax(output)),
        }
        full_state_checks[label] = state_validation(values, train_current, train_initial_v)

    fine_data = load_experiment(config["data_path"], stride=1)
    refinement = {}
    for trace in sorted(CURRENT_BY_TRACE_PA):
        fine_prediction = simulate(
            selected.parameters,
            current_protocol(fine_data.time, fine_data.current_by_trace[trace]),
            initial_voltage(fine_data.voltage_by_trace[trace], fine_data.time),
            dt=0.05 * u.ms,
        )
        fine_at_coarse_endpoints = fine_prediction[1::2]
        coarse_mV = np.asarray(all_predictions[trace].to_decimal(u.mV))
        fine_mV = np.asarray(fine_at_coarse_endpoints.to_decimal(u.mV))
        coarse_spikes = spike_times_ms(arrays["time_ms"], coarse_mV)
        fine_spikes = spike_times_ms(arrays["time_ms"], fine_mV)
        refinement[str(trace)] = {
            "coarse_dt_ms": 0.1,
            "fine_dt_ms": 0.05,
            "fine_downsampling": "indices 1::2, matching integration endpoints",
            "waveform_rmse_mV": float(np.sqrt(np.mean((fine_mV - coarse_mV) ** 2))),
            "coarse_spike_count": int(coarse_spikes.size),
            "fine_spike_count": int(fine_spikes.size),
            "spike_count_difference": int(fine_spikes.size - coarse_spikes.size),
            "coarse_first_spike_ms": float(coarse_spikes[0]) if coarse_spikes.size else None,
            "fine_first_spike_ms": float(fine_spikes[0]) if fine_spikes.size else None,
            "first_spike_shift_ms": (
                float(fine_spikes[0] - coarse_spikes[0])
                if fine_spikes.size and coarse_spikes.size
                else None
            ),
        }
    write_json(metrics_dir / "numerical_refinement.json", refinement)

    recovery_rows = []
    recovery_arrays = {"time_ms": arrays["time_ms"]}
    noise_sd_mV = float(np.std(target[data.time < 50.0 * u.ms].to_decimal(u.mV)))
    for truth_index, truth in enumerate(recovery_truths(config)):
        noise_rng = np.random.default_rng(int(config["recovery"]["noise_seed"]) + truth_index)
        latent = simulate(truth, train_current, train_initial_v)
        observed = latent + noise_rng.normal(0.0, noise_sd_mV, size=latent.shape) * u.mV
        truth_problem = InferenceProblem(data.time, observed, train_current, train_initial_v)
        truth_components = {
            name: float(values[0])
            for name, values in truth_problem.loss_components(truth[None, :]).items()
        }
        results, recovered = fit_one_target(
            config,
            data.time,
            observed,
            train_current,
            train_initial_v,
            f"recovery-{truth_index}",
        )
        recovery_rows.append(
            {
                "case": truth_index,
                "truth": truth.tolist(),
                "noise_sd_mV": noise_sd_mV,
                "truth_objective_components": truth_components,
                "starts": [fit_to_dict(result) for result in results],
                "selected_seed": recovered.seed,
                "selected_loss": recovered.loss,
                "selected_parameters": recovered.parameters.tolist(),
            }
        )
        recovery_arrays[f"case_{truth_index}_latent_mV"] = np.asarray(latent.to_decimal(u.mV))
        recovery_arrays[f"case_{truth_index}_observed_mV"] = np.asarray(observed.to_decimal(u.mV))
    recovery_classification = classify_recovery(config, recovery_rows)
    write_json(raw_dir / "recovery.json", recovery_rows)
    np.savez_compressed(raw_dir / "recovery_observations.npz", **recovery_arrays)
    tradeoffs = recovery_tradeoffs(recovery_rows)
    write_json(metrics_dir / "recovery_tradeoffs.json", tradeoffs)

    prediction_assessment = assess_predictions(metrics_by_trace)
    active_beats_passive = (
        metrics_by_trace[str(FIT_TRACE)]["rmse_mV"] < passive_metrics["rmse_mV"]
    )
    mechanics_valid = (
        all(item["finite"] for item in boundary_checks.values())
        and zero_spikes.size == 0
        and all(count == 0 for count in prestim_spikes_by_trace.values())
        and all(
            state["finite"]
            and state.get("in_gate_range", True)
            and state.get("positive", True)
            for condition in full_state_checks.values()
            for state in condition.values()
        )
        and all(np.isfinite(arrays[f"predicted_trace_{trace}_mV"]).all() for trace in CURRENT_BY_TRACE_PA)
    )
    metrics = {
        "trace_metrics": metrics_by_trace,
        "passive_trace_8_metrics": passive_metrics,
        "prediction_assessment": prediction_assessment,
        "active_model_beats_passive_training_rmse": active_beats_passive,
        "zero_current_spike_count": int(zero_spikes.size),
        "pre_stimulus_spike_count_by_trace": prestim_spikes_by_trace,
        "boundary_checks": boundary_checks,
        "full_state_checks": full_state_checks,
        "numerical_refinement": refinement,
        "per_start_trace_metrics": per_start_metrics,
        "observed_fit_loss_closed": bool(all(result.loss_closed for result in fit_results)),
        "observed_fit_closure": {
            str(result.seed): result.closure_reason for result in fit_results
        },
        "iteration_1_selected_training_objective": config["comparison"]["iteration_1_selected_loss"],
        "selected_loss_improvement_from_iteration_1": float(
            config["comparison"]["iteration_1_selected_loss"] - selected.loss
        ),
        "parameter_recovery": recovery_classification,
        "recovery_tradeoffs": tradeoffs,
        "selected_start_seed": selected.seed,
        "selected_training_objective": selected.loss,
    }
    assessment = {
        "mechanically_valid": mechanics_valid,
        "predictive_consistency_pass": prediction_assessment["predictive_consistency_pass"],
        "active_model_beats_passive": active_beats_passive,
        "overall_predictive_result": (
            "supported-under-tested-protocol"
            if mechanics_valid and active_beats_passive and prediction_assessment["predictive_consistency_pass"]
            else "not-supported-under-locked-criteria"
        ),
        "parameter_claims": {
            name: item["classification"] for name, item in recovery_classification.items()
        },
        "claim_evidence_matrix": {
            "numerical validity": ["metrics/metrics.json:full_state_checks", "metrics/numerical_refinement.json"],
            "training fit": ["raw/predictions.npz", "metrics.json:trace_metrics.8"],
            "held-out prediction": ["raw/per_start_predictions.npz", "metrics/per_start_metrics.json"],
            "parameter interpretation": ["raw/recovery.json", "raw/recovery_observations.npz", "metrics/recovery_tradeoffs.json"],
            "passive control": ["raw/passive_fit.json", "metrics.json:passive_trace_8_metrics"],
        },
        "explicit_non_claims": [
            "No unique biological parameter identification without recovery support.",
            "No exact reproduction of the Du et al. 2025 supplementary parameterization.",
            "No generalization outside 15-30 pA or beyond this single recording series.",
        ],
    }
    write_json(metrics_dir / "metrics.json", metrics)
    write_json(metrics_dir / "assessment.json", assessment)
    write_json(
        raw_dir / "run_provenance.json",
        {
            "run_id": config["run_id"],
            "data_sha256": data.sha256,
            "data_source": str(data.source),
            "python": sys.version,
            "platform": platform.platform(),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "brainstate_dt_ms": 0.1,
            "duration_seconds": time.time() - started,
            "deterministic_simulation": True,
        },
    )
    write_json(run_dir / "artifact_manifest.json", artifact_manifest(run_dir))
    print(
        f"RUN_DONE id={config['run_id']} duration_s={time.time() - started:.3f} "
        f"predictive={assessment['overall_predictive_result']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
