#!/usr/bin/env python3
"""Lifecycle-corrected frozen V1 single-neuron influence experiment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np


RUN_DIR = Path(__file__).resolve().parent
SOURCE = RUN_DIR.parent / "run11" / "v1_influence.py"
spec = importlib.util.spec_from_file_location("paired_v1_influence", SOURCE)
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)
model = experiment.model


def stable_run(self, e_current_nA, i_current_nA, target=None, perturb_second_half=False):
    """Reset pre-existing mapped State without recreating it after tracing."""
    assert e_current_nA.shape[1] == self.batch_size
    brainstate.nn.reset_all_states(self.net, batch_size=self.batch_size)
    target_mask = np.zeros(self.cfg.n_exc, dtype=np.float32)
    if target is not None:
        target_mask[target] = 1.0
    perturb_mask = np.zeros(self.batch_size, dtype=np.float32)
    if perturb_second_half:
        perturb_mask[self.batch_size // 2:] = 1.0
    outputs = self._rollout(
        jnp.asarray(e_current_nA) * u.nA,
        jnp.asarray(i_current_nA) * u.nA,
        jnp.asarray(model.photostim_waveform(self.cfg)) * u.nA,
        jnp.asarray(target_mask),
        jnp.asarray(perturb_mask),
    )
    return tuple(np.asarray(x) for x in outputs)


original_run_trials = model.run_trials


def lifecycle_correct_run_trials(cfg, tuning_net, mapping_net, e_pref, i_pref, targets):
    # Allocate the stable 32-lane State axes before Simulator constructs JIT.
    n_mapping = len(cfg.directions_deg) * cfg.influence_repeats
    brainstate.nn.init_all_states(tuning_net, batch_size=n_mapping)
    brainstate.nn.init_all_states(mapping_net, batch_size=n_mapping)
    return original_run_trials(cfg, tuning_net, mapping_net, e_pref, i_pref, targets)


model.Simulator.run = stable_run
model.run_trials = lifecycle_correct_run_trials

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--output", str(RUN_DIR)]
    model.main()
