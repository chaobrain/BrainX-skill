#!/usr/bin/env python3
"""Test paired pre-period equality over repeated Simulator.run calls."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import brainstate
import brainunit as u
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
sim = module.Simulator(cfg, net, batch_size=32)
stimuli = np.tile(np.asarray(cfg.directions_deg), 2)
rng = np.random.default_rng(cfg.seed + 600)
e_current, i_current, _ = module.trial_currents(cfg, stimuli, e_pref, i_pref, rng)
e_input = np.concatenate([e_current, e_current], axis=1)
i_input = np.concatenate([i_current, i_current], axis=1)

calls = []
for call in range(4):
    output = sim.run(e_input, i_input, target=0, perturb_second_half=True)
    pre_e, pre_i = output[3], output[4]
    calls.append({
        "call": call,
        "exc_packed_mismatches": int(np.count_nonzero(pre_e[:, :16] != pre_e[:, 16:])),
        "inh_packed_mismatches": int(np.count_nonzero(pre_i[:, :16] != pre_i[:, 16:])),
    })
(RUN_DIR / "diagnostic.json").write_text(json.dumps(calls, indent=2) + "\n", encoding="ascii")
print(json.dumps(calls, indent=2))
