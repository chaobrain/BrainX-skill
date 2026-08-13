#!/usr/bin/env python3
"""Locate any paired-lane divergence during the pre-photostimulation period."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np


RUN_DIR = Path(__file__).resolve().parent
SOURCE = RUN_DIR.parent / "run8" / "v1_influence.py"
spec = importlib.util.spec_from_file_location("frozen_v1", SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

cfg = module.Config()
brainstate.environ.set(dt=cfg.dt_ms * u.ms)
e_pos, i_pos, e_pref, i_pref = module.make_positions_and_preferences(cfg)
net = module.V1Network(cfg, e_pos, i_pos, e_pref, i_pref)
stimuli = np.tile(np.asarray(cfg.directions_deg), 2)
rng = np.random.default_rng(cfg.seed + 600)
e_current, i_current, _ = module.trial_currents(cfg, stimuli, e_pref, i_pref, rng)
e_input = np.concatenate([e_current, e_current], axis=1)
i_input = np.concatenate([i_current, i_current], axis=1)
half = stimuli.size
brainstate.nn.init_all_states(net, batch_size=2 * half)

state_checks = {}
for path, state in net.states().items():
    value = state.value
    shape = tuple(getattr(value, "shape", ()))
    if shape and shape[0] == 2 * half:
        raw = np.asarray(value.to_decimal(value.unit) if hasattr(value, "to_decimal") else value)
        state_checks[str(path)] = float(np.max(np.abs(raw[:half] - raw[half:])))

steps = int(round(cfg.pre_ms / cfg.dt_ms))
times = u.math.arange(0.0 * u.ms, cfg.pre_ms * u.ms, cfg.dt_ms * u.ms)

@brainstate.transform.jit
def rollout():
    return brainstate.transform.for_loop(
        lambda t, e, i: net.update(t, e, i),
        times,
        jnp.asarray(e_input[:steps]) * u.nA,
        jnp.asarray(i_input[:steps]) * u.nA,
    )

e_spikes, i_spikes = (np.asarray(x) for x in rollout())
e_diff = e_spikes[:, :half] != e_spikes[:, half:]
i_diff = i_spikes[:, :half] != i_spikes[:, half:]
where = np.argwhere(e_diff)
result = {
    "initialized_state_max_pair_differences": state_checks,
    "exc_spike_mismatches": int(e_diff.sum()),
    "inh_spike_mismatches": int(i_diff.sum()),
    "first_exc_mismatch_time_step_trial_neuron": where[0].tolist() if where.size else None,
    "exc_spike_shape": list(e_spikes.shape),
}
(RUN_DIR / "diagnostic.json").write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
print(json.dumps(result, indent=2))
