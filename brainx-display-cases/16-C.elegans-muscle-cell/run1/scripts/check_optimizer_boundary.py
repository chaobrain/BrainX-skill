"""Check the BrainTools black-box optimizer boundary without installing dependencies."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

import braintools
import brainunit as u

from cellegans_hh.data import FIT_TRACE, current_protocol, initial_voltage, load_experiment
from cellegans_hh.inference import InferenceProblem
from cellegans_hh.model import PARAMETER_SPECS, decode_parameters, parameter_bounds


def main():
    root = Path(__file__).resolve().parents[1]
    data = load_experiment(root / "Fig4A-D.txt")
    time = data.time[:1000]
    problem = InferenceProblem(
        time=time,
        target=data.voltage_by_trace[FIT_TRACE][:1000],
        current=current_protocol(time, data.current_by_trace[FIT_TRACE]),
        initial_v=initial_voltage(data.voltage_by_trace[FIT_TRACE], data.time),
    )

    rng = np.random.default_rng(7011)
    bounds = np.asarray(parameter_bounds(), dtype=np.float64)
    candidates = rng.uniform(bounds[:, 0], bounds[:, 1], size=(28, bounds.shape[0]))
    scipy_losses = np.asarray(problem.vectorized_objective(candidates.T))
    direct_losses = np.asarray(problem._losses(candidates))
    max_loss_error = float(np.max(np.abs(scipy_losses - direct_losses)))

    lower = decode_parameters(bounds[:, 0])
    upper = decode_parameters(bounds[:, 1])
    unit_bounds = {
        spec.name: [
            float(lower[spec.name].to_decimal(spec.unit)),
            float(upper[spec.name].to_decimal(spec.unit)),
        ]
        for spec in PARAMETER_SPECS
    }

    nevergrad_available = importlib.util.find_spec("nevergrad") is not None
    canonical_status = "not-attempted"
    canonical_error = None
    if nevergrad_available:
        canonical_status = "dependency-present-but-execution-not-required"
    else:
        try:
            braintools.optim.NevergradOptimizer(
                lambda **_parameters: direct_losses,
                n_sample=28,
                bounds={
                    spec.name: np.asarray([spec.lower, spec.upper]) * spec.unit
                    for spec in PARAMETER_SPECS
                },
                method="DE",
            )
        except (ImportError, ModuleNotFoundError) as error:
            canonical_status = "unavailable-missing-optional-nevergrad"
            canonical_error = f"{type(error).__name__}: {error}"
        else:
            canonical_status = "constructor-available-without-optional-dependency"

    result = {
        "candidate_count": 28,
        "candidate_shape": list(candidates.shape),
        "loss_shape": list(scipy_losses.shape),
        "finite_losses": bool(np.isfinite(scipy_losses).all()),
        "scipy_to_direct_max_abs_loss_error": max_loss_error,
        "parameter_order": [spec.name for spec in PARAMETER_SPECS],
        "unit_bounds_round_trip": unit_bounds,
        "nevergrad_dependency_available": nevergrad_available,
        "canonical_braintools_status": canonical_status,
        "canonical_braintools_error": canonical_error,
        "production_backend": "scipy.optimize.differential_evolution",
        "production_backend_valid": bool(
            scipy_losses.shape == (28,) and np.isfinite(scipy_losses).all() and max_loss_error == 0.0
        ),
        "limitation": (
            "The canonical BrainTools Nevergrad optimizer cannot execute without the optional "
            "nevergrad dependency. No dependency was installed; the valid SciPy boundary remains unchanged."
        ),
    }
    output = root / "artifacts" / "optimizer-boundary-iteration-2.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
