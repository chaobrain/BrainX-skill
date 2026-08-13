#!/usr/bin/env python3
"""Resource-bounded retry of the frozen run8 V1 influence experiment.

The scientific implementation is loaded from immutable run8. This retry changes
only tuning execution: the original 64 generated trials are simulated as two
sequential 32-trial chunks, matching the 32-lane influence-mapping rollout.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
FROZEN_SOURCE = RUN_DIR.parent / "run8" / "v1_influence.py"
spec = importlib.util.spec_from_file_location("frozen_v1_influence", FROZEN_SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load frozen experiment from {FROZEN_SOURCE}")
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)


def run_trials_chunked(cfg, tuning_net, mapping_net, e_pref, i_pref, targets):
    directions = np.asarray(cfg.directions_deg)
    tuning_stimuli = np.tile(directions, cfg.tuning_repeats)
    mapping_stimuli = np.tile(directions, cfg.influence_repeats)
    chunk_size = mapping_stimuli.size
    assert tuning_stimuli.size == 2 * chunk_size

    # Generate once to preserve run8's exact RNG stream, then only chunk rollout.
    tuning_rng = np.random.default_rng(cfg.seed + 500)
    tuning_e, tuning_i, tuning_gain = model.trial_currents(
        cfg, tuning_stimuli, e_pref, i_pref, tuning_rng
    )
    tuning_simulator = model.Simulator(cfg, tuning_net, batch_size=chunk_size)
    tuning_e_parts, tuning_i_parts = [], []
    for start in range(0, tuning_stimuli.size, chunk_size):
        output = tuning_simulator.run(
            tuning_e[:, start:start + chunk_size],
            tuning_i[:, start:start + chunk_size],
        )
        tuning_e_parts.append(output[0])
        tuning_i_parts.append(output[1])
    tuning_counts_e = np.concatenate(tuning_e_parts, axis=0)
    tuning_counts_i = np.concatenate(tuning_i_parts, axis=0)

    mapping_simulator = model.Simulator(cfg, mapping_net, batch_size=chunk_size)
    all_baseline_e, all_perturbed_e = [], []
    all_baseline_i, all_perturbed_i = [], []
    all_dose_base, all_dose_pert = [], []
    pairing = {
        "noise_and_stimulus_max_abs_difference_nA": 0.0,
        "pre_photostim_exc_spike_mismatches": 0,
        "pre_photostim_inh_spike_mismatches": 0,
        "pairs": 0,
    }
    mapping_gains = []
    for target_order, target in enumerate(targets):
        rng = np.random.default_rng(cfg.seed + 600 + target_order)
        e_base, i_base, gains = model.trial_currents(
            cfg, mapping_stimuli, e_pref, i_pref, rng
        )
        baseline, perturbed = mapping_simulator.run_matched_pair(
            e_base, i_base, target=int(target)
        )
        base_e, base_i, base_dose, base_pre_e, base_pre_i = baseline
        pert_e, pert_i, pert_dose, pert_pre_e, pert_pre_i = perturbed
        pairing["pre_photostim_exc_spike_mismatches"] += int(
            np.count_nonzero(base_pre_e != pert_pre_e)
        )
        pairing["pre_photostim_inh_spike_mismatches"] += int(
            np.count_nonzero(base_pre_i != pert_pre_i)
        )
        pairing["pairs"] += chunk_size
        all_baseline_e.append(base_e)
        all_perturbed_e.append(pert_e)
        all_baseline_i.append(base_i)
        all_perturbed_i.append(pert_i)
        all_dose_base.append(base_dose[:, target])
        all_dose_pert.append(pert_dose[:, target])
        mapping_gains.append(gains)

    if pairing["pre_photostim_exc_spike_mismatches"] or pairing["pre_photostim_inh_spike_mismatches"]:
        raise AssertionError("paired trajectories differ before photostimulation")
    return {
        "tuning_stimuli": tuning_stimuli,
        "tuning_counts_e": tuning_counts_e,
        "tuning_counts_i": tuning_counts_i,
        "tuning_gain": tuning_gain,
        "mapping_stimuli": mapping_stimuli,
        "mapping_gain": np.asarray(mapping_gains),
        "baseline_e": np.asarray(all_baseline_e),
        "perturbed_e": np.asarray(all_perturbed_e),
        "baseline_i": np.asarray(all_baseline_i),
        "perturbed_i": np.asarray(all_perturbed_i),
        "dose_baseline": np.asarray(all_dose_base),
        "dose_perturbed": np.asarray(all_dose_pert),
        "pairing": pairing,
    }


model.run_trials = run_trials_chunked

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--output", str(RUN_DIR)]
    model.main()
