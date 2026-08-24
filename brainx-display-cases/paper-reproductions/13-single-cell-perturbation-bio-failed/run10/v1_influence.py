#!/usr/bin/env python3
"""Causal-control retry of the frozen V1 single-neuron experiment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import brainstate
import brainunit as u


RUN_DIR = Path(__file__).resolve().parent
FROZEN_SOURCE = RUN_DIR.parent / "run9" / "v1_influence.py"
spec = importlib.util.spec_from_file_location("chunked_v1_influence", FROZEN_SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load frozen experiment from {FROZEN_SOURCE}")
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)
model = experiment.model


def run_matched_pair_copied(self, e_current_nA, i_current_nA, target):
    """Run both conditions from copied, not aliased, recurrent State."""
    assert e_current_nA.shape[1] == self.batch_size
    brainstate.nn.init_all_states(self.net, batch_size=self.batch_size)
    state_snapshot = {
        path: jax.tree.map(lambda x: jnp.array(x, copy=True), state.value)
        for path, state in self.net.states().items()
    }

    target_mask = np.zeros(self.cfg.n_exc, dtype=np.float32)
    target_mask[target] = 1.0
    photo_wave = jnp.asarray(model.photostim_waveform(self.cfg)) * u.nA
    e_current = jnp.asarray(e_current_nA) * u.nA
    i_current = jnp.asarray(i_current_nA) * u.nA
    baseline = self._rollout(
        e_current,
        i_current,
        photo_wave,
        jnp.asarray(target_mask),
        jnp.zeros(self.batch_size, dtype=jnp.float32),
    )
    unexpected, missing = brainstate.nn.assign_state_values(self.net, state_snapshot)
    if unexpected or missing:
        raise ValueError(f"state restore mismatch: unexpected={unexpected}, missing={missing}")
    perturbed = self._rollout(
        e_current,
        i_current,
        photo_wave,
        jnp.asarray(target_mask),
        jnp.ones(self.batch_size, dtype=jnp.float32),
    )
    return (
        tuple(np.asarray(x) for x in baseline),
        tuple(np.asarray(x) for x in perturbed),
    )


model.Simulator.run_matched_pair = run_matched_pair_copied
experiment.model.Simulator.run_matched_pair = run_matched_pair_copied

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--output", str(RUN_DIR)]
    model.main()
