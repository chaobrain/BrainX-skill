"""Fit one experimental trace and evaluate three held-out current protocols."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import celegans_model


TRACE_CURRENTS_PA = {6: 15.0, 7: 20.0, 8: 25.0, 9: 30.0}
TRAIN_TRACE = 8
TEST_TRACES = (6, 7, 9)
SPIKE_THRESHOLD_MV = -10.0
SPIKE_REFRACTORY_MS = 5.0


def load_atf(path):
    raw = np.loadtxt(path, skiprows=11)
    time_ms = raw[:, 0] * 1000.0
    traces_mV = {trace: raw[:, trace] for trace in TRACE_CURRENTS_PA}
    expected_dt = celegans_model.DT.to_decimal(u.ms)
    if raw.shape != (10000, 11):
        raise ValueError(f"Expected a 10000x11 numeric ATF table, got {raw.shape}.")
    if not np.allclose(np.diff(time_ms), expected_dt, atol=1e-9):
        raise ValueError("The ATF time base is not uniformly sampled at 0.05 ms.")
    return time_ms, traces_mV


def training_data(path):
    time_ms, traces_mV = load_atf(path)
    return time_ms, traces_mV[TRAIN_TRACE].copy()


def detect_spikes(voltage_mV, dt_ms=0.05):
    crossings = np.flatnonzero(
        (voltage_mV[:-1] < SPIKE_THRESHOLD_MV)
        & (voltage_mV[1:] >= SPIKE_THRESHOLD_MV)
    ) + 1
    refractory_samples = int(round(SPIKE_REFRACTORY_MS / dt_ms))
    accepted = []
    for crossing in crossings:
        if not accepted or crossing - accepted[-1] >= refractory_samples:
            accepted.append(int(crossing))
    return np.asarray(accepted, dtype=int)


def trace_metrics(observed_mV, predicted_mV, dt_ms=0.05):
    residual = predicted_mV - observed_mV
    observed_spikes = detect_spikes(observed_mV, dt_ms)
    predicted_spikes = detect_spikes(predicted_mV, dt_ms)
    windows = {
        "baseline_rmse_mV": slice(0, int(round(50.0 / dt_ms))),
        "stimulus_rmse_mV": slice(
            int(round(50.0 / dt_ms)), int(round(250.0 / dt_ms))
        ),
        "recovery_rmse_mV": slice(int(round(250.0 / dt_ms)), None),
    }
    metrics = {
        "rmse_mV": float(np.sqrt(np.mean(residual**2))),
        "mae_mV": float(np.mean(np.abs(residual))),
        "correlation": float(np.corrcoef(observed_mV, predicted_mV)[0, 1]),
        "observed_spike_count": int(observed_spikes.size),
        "predicted_spike_count": int(predicted_spikes.size),
        "observed_first_spike_ms": (
            float(observed_spikes[0] * dt_ms) if observed_spikes.size else None
        ),
        "predicted_first_spike_ms": (
            float(predicted_spikes[0] * dt_ms) if predicted_spikes.size else None
        ),
        "observed_peak_mV": float(np.max(observed_mV)),
        "predicted_peak_mV": float(np.max(predicted_mV)),
    }
    for name, window in windows.items():
        metrics[name] = float(np.sqrt(np.mean(residual[window] ** 2)))
    return metrics


def _logit(value):
    value = np.clip(value, 1e-5, 1.0 - 1e-5)
    return np.log(value / (1.0 - value))


def _decode_search_coordinates(coordinates, start_nS):
    values = []
    for coordinate, start, spec in zip(
        coordinates, start_nS, celegans_model.PARAMETER_SPECS
    ):
        start_fraction = (start - spec.lower_nS) / (
            spec.upper_nS - spec.lower_nS
        )
        fraction = jax.nn.sigmoid(_logit(start_fraction) + coordinate)
        values.append(spec.lower_nS + fraction * (spec.upper_nS - spec.lower_nS))
    return jnp.stack(values)


def build_objective(target_mV, start_nS):
    cell = celegans_model.CElegansMuscle(target_mV[0] * u.mV)
    cell.init_state()
    _, current = celegans_model.current_protocol(TRACE_CURRENTS_PA[TRAIN_TRACE])
    target = jnp.asarray(target_mV) * u.mV

    def objective(*coordinates):
        parameters = _decode_search_coordinates(coordinates, start_nS)
        celegans_model.apply_parameter_vector(cell, parameters)
        celegans_model.reset_runtime_state(cell)
        prediction = celegans_model.rollout(cell, current)
        residual_mV = (prediction - target[:, None]).to_decimal(u.mV)
        loss = u.math.mean(residual_mV**2)
        return u.math.where(u.math.isfinite(loss), loss, 1.0e12)

    return brainstate.transform.jit(objective)


def fit_one_start(target_mV, start_nS, max_iterations):
    objective = build_objective(target_mV, start_nS)
    optimizer = braintools.optim.ScipyOptimizer(
        objective,
        bounds=[(-6.0, 6.0)] * len(celegans_model.PARAMETER_SPECS),
        method="Nelder-Mead",
    )
    result = optimizer.minimize(n_iter=max_iterations)
    coordinates = np.asarray([float(value) for value in result.x])
    parameters = np.asarray(_decode_search_coordinates(coordinates, start_nS))
    return {
        "parameters_nS": parameters,
        "loss_mV2": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "search_coordinates": coordinates,
    }


def fit_parameters(target_mV, max_iterations):
    nominal = np.asarray(celegans_model.initial_parameter_vector(), dtype=float)
    starts = (
        nominal,
        np.asarray([500.0, 90.0, 5.0, 2.0, 1.0, 1.2]),
        np.asarray([250.0, 45.0, 0.2, 5.0, 0.2, 0.4]),
    )
    results = [fit_one_start(target_mV, start, max_iterations) for start in starts]
    best_index = int(np.argmin([result["loss_mV2"] for result in results]))
    return results, best_index


def simulate_numpy(current_pA, initial_mV, parameters_nS, dt=celegans_model.DT):
    voltage = celegans_model.simulate(
        current_pA=current_pA,
        initial_voltage_mV=initial_mV,
        parameters_nS=jnp.asarray(parameters_nS),
        dt=dt,
    )
    return np.asarray(voltage.to_decimal(u.mV)).squeeze()


def parameter_recovery(target_mV, fitted_nS, max_iterations):
    indices = (0, 1)
    truth = fitted_nS.copy()
    truth[list(indices)] *= np.asarray([0.9, 1.1])
    truth = np.asarray(
        [
            np.clip(value, spec.lower_nS, spec.upper_nS)
            for value, spec in zip(truth, celegans_model.PARAMETER_SPECS)
        ]
    )
    synthetic = simulate_numpy(25.0, target_mV[0], truth)
    result = fit_one_start(synthetic, fitted_nS, min(max_iterations, 20))
    relative_error = {
        celegans_model.PARAMETER_SPECS[i].name: float(
            abs(result["parameters_nS"][i] - truth[i]) / truth[i]
        )
        for i in indices
    }
    return truth, result, relative_error


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(data_path, output_dir, max_iterations=40):
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_names = {
        "metrics.csv",
        "fit_starts.csv",
        "predictions.npz",
        "assessment.json",
        "fitted_parameters.json",
        "recovery.json",
        "run_config.json",
        "manifest.json",
    }
    existing_results = result_names.intersection(path.name for path in output_dir.iterdir())
    if existing_results:
        raise FileExistsError(
            f"Refusing to overwrite existing run results: {sorted(existing_results)}"
        )
    time_ms, training_target = training_data(data_path)

    initial_nS = np.asarray(celegans_model.initial_parameter_vector(), dtype=float)
    initial_prediction = simulate_numpy(25.0, training_target[0], initial_nS)
    initial_loss = float(np.mean((initial_prediction - training_target) ** 2))

    starts, best_index = fit_parameters(training_target, max_iterations)
    fitted_nS = starts[best_index]["parameters_nS"]

    _, all_traces = load_atf(data_path)
    predictions = {}
    metrics = []
    for trace in (TRAIN_TRACE,) + TEST_TRACES:
        observed = all_traces[trace]
        current_pA = TRACE_CURRENTS_PA[trace]
        prediction = simulate_numpy(current_pA, observed[0], fitted_nS)
        predictions[trace] = prediction
        row = {
            "trace": trace,
            "split": "train" if trace == TRAIN_TRACE else "test",
            "current_pA": current_pA,
        }
        row.update(trace_metrics(observed, prediction))
        metrics.append(row)

    zero_current = simulate_numpy(0.0, training_target[0], fitted_nS)
    refined = simulate_numpy(
        25.0, training_target[0], fitted_nS, dt=0.025 * u.ms
    )[::2]
    dt_refinement_rmse = float(
        np.sqrt(np.mean((predictions[TRAIN_TRACE] - refined) ** 2))
    )
    recovery_truth, recovery_result, recovery_error = parameter_recovery(
        training_target, fitted_nS, max_iterations
    )

    fit_rows = []
    for start_index, result in enumerate(starts):
        row = {
            "start": start_index,
            "selected": start_index == best_index,
            "loss_mV2": result["loss_mV2"],
            "success": result["success"],
            "iterations": result["iterations"],
            "function_evaluations": result["function_evaluations"],
            "message": result["message"],
        }
        row.update(celegans_model.parameter_dict(result["parameters_nS"]))
        fit_rows.append(row)

    metric_fields = list(metrics[0].keys())
    write_csv(output_dir / "metrics.csv", metrics, metric_fields)
    write_csv(output_dir / "fit_starts.csv", fit_rows, list(fit_rows[0].keys()))

    np.savez_compressed(
        output_dir / "predictions.npz",
        time_ms=time_ms,
        trace6_observed_mV=all_traces[6],
        trace6_predicted_mV=predictions[6],
        trace7_observed_mV=all_traces[7],
        trace7_predicted_mV=predictions[7],
        trace8_observed_mV=all_traces[8],
        trace8_predicted_mV=predictions[8],
        trace9_observed_mV=all_traces[9],
        trace9_predicted_mV=predictions[9],
        trace8_initial_prediction_mV=initial_prediction,
        zero_current_mV=zero_current,
    )

    predicted_counts = [
        row["predicted_spike_count"]
        for row in sorted(metrics, key=lambda item: item["current_pA"])
    ]
    assessment = {
        "initial_training_loss_mV2": initial_loss,
        "fitted_training_loss_mV2": starts[best_index]["loss_mV2"],
        "training_loss_improved": starts[best_index]["loss_mV2"] < initial_loss,
        "all_predictions_finite": all(
            np.isfinite(prediction).all() for prediction in predictions.values()
        ),
        "held_out_spike_counts_nondecreasing": all(
            left <= right for left, right in zip(predicted_counts, predicted_counts[1:])
        ),
        "predicted_spike_counts_15_20_25_30_pA": predicted_counts,
        "zero_current_spike_count": int(detect_spikes(zero_current).size),
        "dt_refinement_rmse_mV": dt_refinement_rmse,
        "recovery_relative_error": recovery_error,
        "parameter_interpretation": "withheld: single-trace conductances are not identifiable",
        "predictive_claim": "limited to held-out 15, 20, and 30 pA protocols",
    }
    (output_dir / "assessment.json").write_text(
        json.dumps(assessment, indent=2) + "\n", encoding="ascii"
    )
    (output_dir / "fitted_parameters.json").write_text(
        json.dumps(
            {
                "unit": "nS",
                "parameter_order": [
                    spec.name for spec in celegans_model.PARAMETER_SPECS
                ],
                "values": celegans_model.parameter_dict(fitted_nS),
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    (output_dir / "recovery.json").write_text(
        json.dumps(
            {
                "truth_nS": celegans_model.parameter_dict(recovery_truth),
                "recovered_nS": celegans_model.parameter_dict(
                    recovery_result["parameters_nS"]
                ),
                "relative_error_selected_parameters": recovery_error,
                "loss_mV2": recovery_result["loss_mV2"],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )

    config = {
        "data": str(data_path.resolve()),
        "data_sha256": sha256(data_path),
        "train_trace": TRAIN_TRACE,
        "test_traces": list(TEST_TRACES),
        "trace_currents_pA": TRACE_CURRENTS_PA,
        "stimulus_window_ms": [50.0, 250.0],
        "dt_ms": 0.05,
        "duration_ms": 500.0,
        "solver": "ind_exp_euler",
        "optimizer": "braintools.optim.ScipyOptimizer/Nelder-Mead",
        "max_iterations_per_start": max_iterations,
        "number_of_starts": 3,
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="ascii"
    )
    manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii"
    )
    return assessment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Fig4A-D.txt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=40)
    args = parser.parse_args()
    assessment = run(args.data, args.output, args.max_iterations)
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
