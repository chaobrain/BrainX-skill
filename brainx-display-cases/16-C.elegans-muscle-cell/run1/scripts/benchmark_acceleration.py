"""Benchmark native BrainCell candidate batching against serial scalar rollouts."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import brainunit as u

from cellegans_hh.data import FIT_TRACE, current_protocol, initial_voltage, load_experiment
from cellegans_hh.model import parameter_bounds, simulate


def timed(callable_):
    start = time.perf_counter()
    value = callable_()
    return value, time.perf_counter() - start


def main():
    root = Path(__file__).resolve().parents[1]
    data = load_experiment(root / "Fig4A-D.txt")
    time_axis = data.time[:1000]
    target = data.voltage_by_trace[FIT_TRACE]
    initial_v = initial_voltage(target, data.time)
    current = current_protocol(time_axis, data.current_by_trace[FIT_TRACE])

    rng = np.random.default_rng(7003)
    bounds = np.asarray(parameter_bounds())
    candidates = rng.uniform(bounds[:, 0], bounds[:, 1], size=(8, bounds.shape[0]))

    batch_cold, batch_cold_s = timed(lambda: simulate(candidates, current, initial_v))
    batch_warm, batch_warm_s = timed(lambda: simulate(candidates, current, initial_v))
    serial_cold, serial_cold_s = timed(
        lambda: u.math.stack([simulate(candidate, current, initial_v) for candidate in candidates], axis=1)
    )
    serial_warm, serial_warm_s = timed(
        lambda: u.math.stack([simulate(candidate, current, initial_v) for candidate in candidates], axis=1)
    )

    batch_states = simulate(candidates, current, initial_v, return_states=True)
    serial_states = [simulate(candidate, current, initial_v, return_states=True) for candidate in candidates]
    state_errors = {}
    for name, batch_values in batch_states.items():
        if name == "voltage":
            batch_numeric = np.asarray(batch_values.to_decimal(u.mV))
            serial_numeric = np.stack(
                [np.asarray(values[name].to_decimal(u.mV)) for values in serial_states], axis=1
            )
        elif name == "calcium_i":
            batch_numeric = np.asarray(batch_values.to_decimal(u.mM))
            serial_numeric = np.stack(
                [np.asarray(values[name].to_decimal(u.mM)) for values in serial_states], axis=1
            )
        else:
            batch_numeric = np.asarray(batch_values)
            serial_numeric = np.stack([np.asarray(values[name]) for values in serial_states], axis=1)
        state_errors[name] = float(np.max(np.abs(batch_numeric - serial_numeric)))

    batch_mV = np.asarray(batch_warm.to_decimal(u.mV))
    serial_mV = np.asarray(serial_warm.to_decimal(u.mV))
    max_abs_error = float(np.max(np.abs(batch_mV - serial_mV)))
    if not np.isfinite(batch_mV).all() or max_abs_error > 1e-5:
        raise RuntimeError(f"Batch parity failed: max error {max_abs_error} mV")
    if any(error > 1e-5 for error in state_errors.values()):
        raise RuntimeError(f"Full-State batch parity failed: {state_errors}")

    result = {
        "candidate_count": int(candidates.shape[0]),
        "time_steps": int(time_axis.shape[0]),
        "batch_shape": list(batch_cold.shape),
        "serial_shape": list(serial_cold.shape),
        "batch_cold_seconds": batch_cold_s,
        "batch_warm_seconds": batch_warm_s,
        "serial_cold_seconds": serial_cold_s,
        "serial_warm_seconds": serial_warm_s,
        "warm_speedup": serial_warm_s / batch_warm_s,
        "max_abs_error_mV": max_abs_error,
        "full_state_max_abs_errors": state_errors,
        "finite": True,
    }
    output = root / "artifacts" / "acceleration-iteration-2.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
