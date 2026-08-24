"""Read the supplied Axon Text File without modifying the raw recording."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import brainunit as u


CURRENT_BY_TRACE_PA = {6: 15.0, 7: 20.0, 8: 25.0, 9: 30.0}
FIT_TRACE = 8
TEST_TRACES = (6, 7, 9)


@dataclass(frozen=True)
class ExperimentalData:
    time: u.Quantity
    voltage_by_trace: dict[int, u.Quantity]
    current_by_trace: dict[int, u.Quantity]
    sha256: str
    source: Path


def load_experiment(path: str | Path, stride: int = 2) -> ExperimentalData:
    """Load requested traces, preserving the declared units and split."""
    source = Path(path)
    raw = source.read_bytes()
    with source.open(newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))

    header_index = next(
        i for i, row in enumerate(rows) if row and row[0].strip('"') == "Time (s)"
    )
    numeric = np.asarray(
        [[float(value) for value in row[:11]] for row in rows[header_index + 1 :] if len(row) >= 11],
        dtype=np.float64,
    )
    if numeric.shape != (10000, 11):
        raise ValueError(f"Expected a 10000 x 11 ATF table, got {numeric.shape}.")
    if stride < 1 or numeric.shape[0] % stride:
        raise ValueError("stride must divide the 10,000-sample recording exactly.")

    time = numeric[::stride, 0] * u.second
    expected_dt = stride * 0.05 * u.ms
    actual_dt = np.median(np.diff(time.to_decimal(u.ms))) * u.ms
    if not u.math.allclose(actual_dt, expected_dt, rtol=1e-8, atol=1e-10 * u.ms):
        raise ValueError(f"Unexpected sampling interval: {actual_dt}.")

    voltages = {trace: numeric[::stride, trace] * u.mV for trace in CURRENT_BY_TRACE_PA}
    currents = {trace: value * u.pA for trace, value in CURRENT_BY_TRACE_PA.items()}
    return ExperimentalData(
        time=time,
        voltage_by_trace=voltages,
        current_by_trace=currents,
        sha256=hashlib.sha256(raw).hexdigest(),
        source=source.resolve(),
    )


def initial_voltage(voltage: u.Quantity, time: u.Quantity) -> u.Quantity:
    mask = time < 50.0 * u.ms
    return u.math.mean(voltage[mask])


def current_protocol(time: u.Quantity, amplitude: u.Quantity) -> u.Quantity:
    active = (time >= 50.0 * u.ms) & (time < 300.0 * u.ms)
    return u.math.where(active, amplitude, 0.0 * u.pA)
