#!/usr/bin/env python3
"""Compare direct and explicitly vmapped BrainEvent CSR batches."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import brainevent
import brainstate
import brainunit as u
import jax
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
rng = np.random.default_rng(cfg.seed + 1200)
events = rng.random((16, cfg.n_exc)) < 0.1
duplicated = jnp.asarray(np.concatenate([events, events], axis=0))
conn = net.connectivity["EE"]

direct = np.asarray((brainevent.BinaryArray(duplicated) @ conn).to_decimal(u.nS))
mapped = np.asarray(
    jax.vmap(lambda row: brainevent.BinaryArray(row) @ conn)(duplicated).to_decimal(u.nS)
)
result = {
    "direct_duplicate_max_abs_difference_nS": float(np.max(np.abs(direct[:16] - direct[16:]))),
    "mapped_duplicate_max_abs_difference_nS": float(np.max(np.abs(mapped[:16] - mapped[16:]))),
    "direct_vs_mapped_mismatched_values": int(np.count_nonzero(direct != mapped)),
    "shape": list(direct.shape),
}
(RUN_DIR / "diagnostic.json").write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
print(json.dumps(result, indent=2))
