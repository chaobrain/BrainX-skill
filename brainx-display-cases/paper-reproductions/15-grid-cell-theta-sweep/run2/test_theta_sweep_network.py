"""Focused checks for the generated theta-sweep display-case artifacts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "theta_sweep_network",
    HERE / "theta_sweep_network.py",
)
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)


def test_alternation_score_counts_weak_pairs_in_denominator():
    strong_alternating = np.deg2rad([20.0, -20.0, 20.0, -20.0])
    one_weak_pair = np.deg2rad([20.0, -20.0, 0.0, -20.0])

    assert MODEL.alternation_score(strong_alternating) == 1.0
    assert MODEL.alternation_score(one_weak_pair) == 1.0 / 3.0


def test_navigation_protocols_are_unit_aware_and_two_dimensional():
    for name, expected_steps in (("straight", 3000), ("speed", 4500), ("turn", 5000)):
        protocol = MODEL.make_protocol(name)
        assert protocol["index"].shape == (expected_steps,)
        assert protocol["position"].shape == (expected_steps, 2)
        assert protocol["heading"].shape == (expected_steps,)
        assert protocol["speed"].shape == (expected_steps,)
        assert protocol["time"].unit == MODEL.u.second
        assert protocol["position"].unit == MODEL.u.cm
        assert protocol["heading"].unit == MODEL.u.radian


def test_saved_results_support_reported_mechanism_claims():
    summary = json.loads((HERE / "results" / "summary.json").read_text())
    conditions = summary["conditions"]
    baseline = conditions["baseline"]

    assert baseline["alternation_score"] > baseline["shuffle_95_interval"][1]
    assert baseline["shuffle_p_upper"] < 0.001
    assert min(baseline["mean_alignment_cosine"]) > 0.98
    assert conditions["no_adaptation"]["alternation_score"] == 0.0
    assert conditions["no_theta"]["alternation_score"] == 0.0
    assert conditions["no_coupling"]["alternation_score"] > 0.9
    assert conditions["no_coupling"]["grid_alternation_score"] == [0.0, 0.0, 0.0]
    assert summary["turn"]["alternation_score"] > 0.9


def test_speed_sweep_length_is_monotonic_for_every_grid_module():
    summary = json.loads((HERE / "results" / "summary.json").read_text())
    lengths = np.asarray(summary["speed"]["mean_grid_length_cm"])

    assert np.all(np.diff(lengths, axis=0) > 0.0)
    assert np.all(np.diff(lengths, axis=1) > 0.0)


def test_vector_selection_contains_exactly_ten_distinct_cycles():
    rows = MODEL.select_vector_rows(
        {
            "cycle": np.arange(90),
            "time_s": np.linspace(1.05, 9.95, 90),
        }
    )

    assert rows.shape == (10,)
    assert np.unique(rows).size == 10


def test_saved_trajectory_vectors_have_ten_distinct_directions():
    with np.load(HERE / "results" / "theta_sweep_evidence.npz") as evidence:
        angles = evidence["turn_vector_direction_deg"]

    assert angles.shape == (10,)
    assert np.unique(np.round(angles, 2)).size == 10
