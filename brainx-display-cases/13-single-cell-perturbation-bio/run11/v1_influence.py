#!/usr/bin/env python3
"""Single-rollout causal-pair retry of the frozen V1 experiment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
FROZEN_SOURCE = RUN_DIR.parent / "run9" / "v1_influence.py"
spec = importlib.util.spec_from_file_location("chunked_v1_influence", FROZEN_SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load frozen experiment from {FROZEN_SOURCE}")
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)
model = experiment.model


def run_trials_single_rollout_pairs(cfg, tuning_net, mapping_net, e_pref, i_pref, targets):
    directions = np.asarray(cfg.directions_deg)
    tuning_stimuli = np.tile(directions, cfg.tuning_repeats)
    mapping_stimuli = np.tile(directions, cfg.influence_repeats)
    batch_size = mapping_stimuli.size

    tuning_rng = np.random.default_rng(cfg.seed + 500)
    tuning_e, tuning_i, tuning_gain = model.trial_currents(
        cfg, tuning_stimuli, e_pref, i_pref, tuning_rng
    )
    tuning_simulator = model.Simulator(cfg, tuning_net, batch_size=batch_size)
    tuning_outputs = [
        tuning_simulator.run(tuning_e[:, start:start + batch_size], tuning_i[:, start:start + batch_size])
        for start in range(0, tuning_stimuli.size, batch_size)
    ]
    tuning_counts_e = np.concatenate([output[0] for output in tuning_outputs], axis=0)
    tuning_counts_i = np.concatenate([output[1] for output in tuning_outputs], axis=0)

    # Sixteen paired observations fit into one 32-lane rollout. The first half
    # is baseline and the second half receives the target-only photo current.
    pair_chunk = batch_size // 2
    mapping_simulator = model.Simulator(cfg, mapping_net, batch_size=batch_size)
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
        target_base_e, target_pert_e = [], []
        target_base_i, target_pert_i = [], []
        target_dose_base, target_dose_pert = [], []
        for start in range(0, batch_size, pair_chunk):
            e_chunk = e_base[:, start:start + pair_chunk]
            i_chunk = i_base[:, start:start + pair_chunk]
            e_paired = np.concatenate([e_chunk, e_chunk], axis=1)
            i_paired = np.concatenate([i_chunk, i_chunk], axis=1)
            pairing["noise_and_stimulus_max_abs_difference_nA"] = max(
                pairing["noise_and_stimulus_max_abs_difference_nA"],
                float(np.max(np.abs(e_paired[:, :pair_chunk] - e_paired[:, pair_chunk:]))),
                float(np.max(np.abs(i_paired[:, :pair_chunk] - i_paired[:, pair_chunk:]))),
            )
            output = mapping_simulator.run(
                e_paired, i_paired, target=int(target), perturb_second_half=True
            )
            e_count, i_count, dose, pre_e, pre_i = output
            pairing["pre_photostim_exc_spike_mismatches"] += int(
                np.count_nonzero(pre_e[:, :pair_chunk] != pre_e[:, pair_chunk:])
            )
            pairing["pre_photostim_inh_spike_mismatches"] += int(
                np.count_nonzero(pre_i[:, :pair_chunk] != pre_i[:, pair_chunk:])
            )
            target_base_e.append(e_count[:pair_chunk])
            target_pert_e.append(e_count[pair_chunk:])
            target_base_i.append(i_count[:pair_chunk])
            target_pert_i.append(i_count[pair_chunk:])
            target_dose_base.append(dose[:pair_chunk, target])
            target_dose_pert.append(dose[pair_chunk:, target])
            pairing["pairs"] += pair_chunk
        all_baseline_e.append(np.concatenate(target_base_e))
        all_perturbed_e.append(np.concatenate(target_pert_e))
        all_baseline_i.append(np.concatenate(target_base_i))
        all_perturbed_i.append(np.concatenate(target_pert_i))
        all_dose_base.append(np.concatenate(target_dose_base))
        all_dose_pert.append(np.concatenate(target_dose_pert))
        mapping_gains.append(gains)

    if pairing["noise_and_stimulus_max_abs_difference_nA"] != 0.0:
        raise AssertionError("paired external inputs differ")
    if pairing["pre_photostim_exc_spike_mismatches"] or pairing["pre_photostim_inh_spike_mismatches"]:
        raise AssertionError(f"paired trajectories differ before photostimulation: {pairing}")
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


model.run_trials = run_trials_single_rollout_pairs

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--output", str(RUN_DIR)]
    model.main()
