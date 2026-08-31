"""Stimulation-based parameter inference and deterministic trace metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks

import brainunit as u

from .model import PARAMETER_SPECS, initial_parameter_vector, parameter_bounds, simulate


SPIKE_HEIGHT_MV = 0.0
SPIKE_PROMINENCE_MV = 15.0
SPIKE_DISTANCE_MS = 5.0
STIMULUS_START_MS = 50.0
STIMULUS_END_MS = 300.0


def spike_times_ms(
    time_ms: np.ndarray,
    voltage_mV: np.ndarray,
    start_ms: float | None = STIMULUS_START_MS,
    end_ms: float | None = STIMULUS_END_MS,
) -> np.ndarray:
    dt_ms = float(np.median(np.diff(time_ms)))
    active = np.ones(time_ms.shape, dtype=bool)
    if start_ms is not None:
        active &= time_ms >= start_ms
    if end_ms is not None:
        active &= time_ms < end_ms
    peaks, _ = find_peaks(
        voltage_mV[active],
        height=SPIKE_HEIGHT_MV,
        prominence=SPIKE_PROMINENCE_MV,
        distance=max(1, int(round(SPIKE_DISTANCE_MS / dt_ms))),
    )
    return time_ms[active][peaks]


def trace_metrics(time, observed, predicted) -> dict[str, float | int | None]:
    time_ms = np.asarray(time.to_decimal(u.ms))
    observed_mV = np.asarray(observed.to_decimal(u.mV))
    predicted_mV = np.asarray(predicted.to_decimal(u.mV))
    residual = predicted_mV - observed_mV
    observed_spikes = spike_times_ms(time_ms, observed_mV)
    predicted_spikes = spike_times_ms(time_ms, predicted_mV)
    if np.std(observed_mV) > 0 and np.std(predicted_mV) > 0:
        correlation = float(np.corrcoef(observed_mV, predicted_mV)[0, 1])
    else:
        correlation = None
    return {
        "rmse_mV": float(np.sqrt(np.mean(residual**2))),
        "mae_mV": float(np.mean(np.abs(residual))),
        "correlation": correlation,
        "observed_spike_count": int(observed_spikes.size),
        "predicted_spike_count": int(predicted_spikes.size),
        "observed_first_spike_ms": float(observed_spikes[0]) if observed_spikes.size else None,
        "predicted_first_spike_ms": float(predicted_spikes[0]) if predicted_spikes.size else None,
        "spike_count_error": int(abs(predicted_spikes.size - observed_spikes.size)),
        "first_spike_error_ms": (
            float(abs(predicted_spikes[0] - observed_spikes[0]))
            if predicted_spikes.size and observed_spikes.size
            else None
        ),
    }


@dataclass
class CandidateBatchRecord:
    call: int
    candidates: int
    finite: int
    invalid: int
    minimum: float
    median: float
    maximum: float
    cumulative_best: float
    best_total_rmse_mV: float
    best_baseline_rmse_mV: float
    best_spike_count_penalty: float
    best_spike_timing_penalty: float


@dataclass
class InferenceProblem:
    time: u.Quantity
    target: u.Quantity
    current: u.Quantity
    initial_v: u.Quantity
    records: list[CandidateBatchRecord] = field(default_factory=list)
    _cumulative_best: float = field(default=np.inf, init=False)

    def __post_init__(self):
        self.time_ms = np.asarray(self.time.to_decimal(u.ms))
        self.target_mV = np.asarray(self.target.to_decimal(u.mV))
        self.target_spikes_ms = spike_times_ms(self.time_ms, self.target_mV)
        self.prestim = self.time_ms < STIMULUS_START_MS

    def loss_components(self, candidates: np.ndarray) -> dict[str, np.ndarray]:
        predictions = simulate(candidates, self.current, self.initial_v).to_decimal(u.mV)
        predictions = np.asarray(predictions)
        if predictions.ndim == 1:
            predictions = predictions[:, None]
        shape = predictions.shape[1]
        components = {
            "total_rmse_mV": np.full(shape, np.nan, dtype=np.float64),
            "baseline_rmse_mV": np.full(shape, np.nan, dtype=np.float64),
            "spike_count_penalty": np.full(shape, np.nan, dtype=np.float64),
            "spike_timing_penalty": np.full(shape, np.nan, dtype=np.float64),
            "total": np.full(shape, 1.0e6, dtype=np.float64),
        }
        for lane in range(predictions.shape[1]):
            predicted = predictions[:, lane]
            if not np.isfinite(predicted).all() or predicted.min() < -120.0 or predicted.max() > 100.0:
                continue
            residual = predicted - self.target_mV
            rmse = float(np.sqrt(np.mean(residual**2)))
            baseline_rmse = float(np.sqrt(np.mean(residual[self.prestim] ** 2)))
            predicted_spikes = spike_times_ms(self.time_ms, predicted)
            count_penalty = 4.0 * abs(predicted_spikes.size - self.target_spikes_ms.size)
            paired = min(predicted_spikes.size, self.target_spikes_ms.size)
            timing_penalty = 0.0
            if paired:
                timing_penalty = 0.025 * float(
                    np.mean(np.abs(predicted_spikes[:paired] - self.target_spikes_ms[:paired]))
                )
            components["total_rmse_mV"][lane] = rmse
            components["baseline_rmse_mV"][lane] = baseline_rmse
            components["spike_count_penalty"][lane] = count_penalty
            components["spike_timing_penalty"][lane] = timing_penalty
            components["total"][lane] = (
                rmse + 0.5 * baseline_rmse + count_penalty + timing_penalty
            )
        return components

    def _losses(self, candidates: np.ndarray) -> np.ndarray:
        return self.loss_components(candidates)["total"]

    def vectorized_objective(self, scipy_values: np.ndarray) -> np.ndarray | float:
        values = np.asarray(scipy_values, dtype=np.float64)
        scalar = values.ndim == 1
        candidates = values[None, :] if scalar else values.T
        components = self.loss_components(candidates)
        losses = components["total"]
        finite = np.isfinite(losses) & (losses < 1.0e6)
        best_index = int(np.argmin(losses))
        self._cumulative_best = min(self._cumulative_best, float(losses[best_index]))
        self.records.append(
            CandidateBatchRecord(
                call=len(self.records),
                candidates=int(losses.size),
                finite=int(finite.sum()),
                invalid=int((~finite).sum()),
                minimum=float(np.min(losses)),
                median=float(np.median(losses)),
                maximum=float(np.max(losses)),
                cumulative_best=self._cumulative_best,
                best_total_rmse_mV=float(components["total_rmse_mV"][best_index]),
                best_baseline_rmse_mV=float(components["baseline_rmse_mV"][best_index]),
                best_spike_count_penalty=float(components["spike_count_penalty"][best_index]),
                best_spike_timing_penalty=float(components["spike_timing_penalty"][best_index]),
            )
        )
        return float(losses[0]) if scalar else losses


@dataclass(frozen=True)
class FitResult:
    seed: int
    parameters: np.ndarray
    loss: float
    success: bool
    message: str
    nfev: int
    nit: int
    records: tuple[CandidateBatchRecord, ...]
    initial_parameters: np.ndarray
    initial_loss: float
    initial_components: dict[str, float]
    loss_closed: bool
    closure_reason: str
    candidate_evaluations: int


def fit_problem(
    problem: InferenceProblem,
    seed: int,
    maxiter: int = 18,
    popsize: int = 4,
    plateau_generations: int = 20,
    plateau_tolerance: float = 0.01,
) -> FitResult:
    initial = np.asarray(initial_parameter_vector(), dtype=np.float64)
    initial_raw = problem.loss_components(initial[None, :])
    initial_components = {name: float(values[0]) for name, values in initial_raw.items()}
    problem.records.clear()
    problem._cumulative_best = np.inf
    stopped_on_plateau = False

    def stop_on_plateau(_parameters, _convergence):
        nonlocal stopped_on_plateau
        history = [record.cumulative_best for record in problem.records]
        if len(history) <= plateau_generations:
            return False
        improvement = history[-plateau_generations - 1] - history[-1]
        stopped_on_plateau = improvement <= plateau_tolerance
        return stopped_on_plateau

    result = differential_evolution(
        problem.vectorized_objective,
        bounds=parameter_bounds(),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-3,
        atol=1e-3,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False,
        updating="deferred",
        workers=1,
        vectorized=True,
        callback=stop_on_plateau,
    )
    scipy_converged = bool(result.success) and not stopped_on_plateau
    closure_reason = (
        "predeclared-plateau"
        if stopped_on_plateau
        else "estimator-converged"
        if scipy_converged
        else "maximum-budget-without-closure"
    )
    return FitResult(
        seed=seed,
        parameters=np.asarray(result.x),
        loss=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
        nit=int(result.nit),
        records=tuple(problem.records),
        initial_parameters=initial,
        initial_loss=initial_components["total"],
        initial_components=initial_components,
        loss_closed=bool(stopped_on_plateau or scipy_converged),
        closure_reason=closure_reason,
        candidate_evaluations=sum(record.candidates for record in problem.records),
    )


def fit_passive_control(problem: InferenceProblem, seed: int = 3101) -> FitResult:
    full_bounds = parameter_bounds()

    def passive_objective(values):
        values = np.asarray(values)
        scalar = values.ndim == 1
        reduced = values[None, :] if scalar else values.T
        candidates = np.zeros((reduced.shape[0], len(PARAMETER_SPECS)))
        candidates[:, 5] = reduced[:, 0]
        candidates[:, 6] = reduced[:, 1]
        losses = problem._losses(candidates)
        return float(losses[0]) if scalar else losses

    result = differential_evolution(
        passive_objective,
        bounds=(full_bounds[5], full_bounds[6]),
        seed=seed,
        maxiter=12,
        popsize=5,
        polish=False,
        updating="deferred",
        workers=1,
        vectorized=True,
    )
    parameters = np.zeros(len(PARAMETER_SPECS))
    parameters[5:] = result.x
    return FitResult(
        seed=seed,
        parameters=parameters,
        loss=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
        nit=int(result.nit),
        records=(),
        initial_parameters=np.asarray(initial_parameter_vector(), dtype=np.float64),
        initial_loss=float("nan"),
        initial_components={},
        loss_closed=bool(result.success),
        closure_reason="control-only",
        candidate_evaluations=int(result.nfev),
    )
