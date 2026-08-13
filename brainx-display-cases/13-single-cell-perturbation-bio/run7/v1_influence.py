#!/usr/bin/env python3
"""Frozen single-neuron influence-mapping experiment in a point-neuron V1 model.

The default command performs one preregistered model realization. It does not
search parameters or rerun failed signatures. All recurrent dynamics use the
BrainX stack; NumPy/SciPy are used only at input-generation and analysis
boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-v1-influence")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import brainevent
import brainpy
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import matplotlib
import numpy as np
from braintools.conn import DistanceDependent
from braintools.init import Constant, GaussianProfile
from scipy.signal import lfilter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Config:
    # Frozen design. Distances are um, times ms, currents nA, conductances nS.
    seed: int = 20260308
    n_exc: int = 320
    n_inh: int = 80
    field_size_um: float = 500.0
    n_targets: int = 16
    directions_deg: tuple[float, ...] = tuple(np.arange(0.0, 360.0, 45.0))
    tuning_repeats: int = 8
    influence_repeats: int = 4
    dt_ms: float = 0.5
    pre_ms: float = 100.0
    response_ms: float = 367.0
    trial_ms: float = 500.0
    photo_sweep_ms: float = 32.0
    photo_frequency_hz: float = 15.0
    photo_sweeps: int = 4
    photo_current_nA: float = 0.12
    photo_dose_window_ms: float = 250.0
    exclude_distance_um: float = 25.0
    distance_edges_um: tuple[float, ...] = (25.0, 100.0, 300.0, np.inf)
    bootstrap_samples: int = 2000
    alpha: float = 0.05

    # LIF and conductance dynamics.
    e_tau_ms: float = 20.0
    i_tau_ms: float = 10.0
    refractory_ms: float = 3.0
    resistance_Mohm: float = 100.0
    v_rest_mV: float = -65.0
    v_reset_mV: float = -60.0
    v_threshold_mV: float = -50.0
    exc_reversal_mV: float = 0.0
    inh_reversal_mV: float = -80.0
    exc_syn_tau_ms: float = 5.0
    inh_syn_tau_ms: float = 10.0

    # Feedforward drive and trial variability.
    e_background_nA: float = 0.115
    i_background_nA: float = 0.120
    e_visual_untuned_nA: float = 0.025
    e_visual_tuned_nA: float = 0.075
    i_visual_untuned_nA: float = 0.035
    i_visual_tuned_nA: float = 0.045
    e_tuning_kappa: float = 1.6
    i_tuning_kappa: float = 0.7
    gain_sd: float = 0.12
    gain_min: float = 0.65
    gain_max: float = 1.35
    noise_sd_nA: float = 0.040
    noise_tau_ms: float = 5.0
    common_noise_fraction: float = 0.15

    # Phenomenological recurrent wiring: spatial candidate scale, thinning,
    # base conductance, and functional modulation strength for EE, EI, IE, II.
    ee_sigma_um: float = 65.0
    ei_sigma_um: float = 110.0
    ie_sigma_um: float = 240.0
    ii_sigma_um: float = 100.0
    ee_max_um: float = 200.0
    ei_max_um: float = 300.0
    ie_max_um: float = 600.0
    ii_max_um: float = 300.0
    ee_keep: float = 0.45
    ei_keep: float = 0.50
    ie_keep: float = 0.55
    ii_keep: float = 0.45
    ee_weight_nS: float = 0.40
    ei_weight_nS: float = 0.50
    ie_weight_nS: float = 0.85
    ii_weight_nS: float = 0.65
    ee_feature_strength: float = 0.50
    ei_feature_strength: float = 0.75
    ie_feature_strength: float = 0.75
    ii_feature_strength: float = 0.20


PROJECTION_NAMES = ("EE", "EI", "IE", "II")


def orientation_difference_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest orientation difference in [0, 90] degrees."""
    delta = np.abs((np.asarray(a) - np.asarray(b)) % 180.0)
    return np.minimum(delta, 180.0 - delta)


def tuning_kernel(stimulus_deg: np.ndarray, preferred_deg: np.ndarray, kappa: float) -> np.ndarray:
    delta = np.deg2rad(np.asarray(stimulus_deg)[..., None] - np.asarray(preferred_deg))
    return np.exp(kappa * (np.cos(2.0 * delta) - 1.0))


def standardized(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return np.zeros_like(x) if sd == 0 else (x - x.mean()) / sd


def percentile_ci(samples: np.ndarray, alpha: float) -> list[float]:
    lo, hi = np.quantile(np.asarray(samples), [alpha / 2.0, 1.0 - alpha / 2.0])
    return [float(lo), float(hi)]


def correlation_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlations, with constant rows mapped to zero."""
    a0 = a - a.mean(axis=1, keepdims=True)
    b0 = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(a0 * a0, axis=1) * np.sum(b0 * b0, axis=1))
    return np.divide(np.sum(a0 * b0, axis=1), denom, out=np.zeros_like(denom), where=denom > 0)


def piecewise_distance_design(distance_um: np.ndarray) -> np.ndarray:
    """Continuous segmented distance basis with knots at 100 and 300 um."""
    d = np.asarray(distance_um, dtype=float)
    near = np.clip(d - 25.0, 0.0, 75.0) / 75.0
    middle = np.clip(d - 100.0, 0.0, 200.0) / 200.0
    far = np.maximum(d - 300.0, 0.0) / 200.0
    return np.column_stack([np.ones(d.size), near, middle, far])


def make_positions_and_preferences(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    e_pos = rng.uniform(0.0, cfg.field_size_um, size=(cfg.n_exc, 2))
    i_pos = rng.uniform(0.0, cfg.field_size_um, size=(cfg.n_inh, 2))
    e_pref = rng.uniform(0.0, 180.0, size=cfg.n_exc)
    i_pref = rng.uniform(0.0, 180.0, size=cfg.n_inh)
    return e_pos, i_pos, e_pref, i_pref


def _connection_csr(
    pre_pos_um: np.ndarray,
    post_pos_um: np.ndarray,
    pre_pref_deg: np.ndarray,
    post_pref_deg: np.ndarray,
    sigma_um: float,
    max_um: float,
    keep_probability: float,
    base_weight_nS: float,
    feature_strength: float,
    seed: int,
    allow_self: bool,
) -> tuple[brainevent.CSR, dict]:
    """Generate spatial candidates with BrainTools, then thin and weight them."""
    pattern = DistanceDependent(
        GaussianProfile(sigma=sigma_um * u.um, max_distance=max_um * u.um),
        weight=Constant(1.0 * u.nS),
        allow_self_connections=allow_self,
        seed=seed,
    )
    result = pattern(
        pre_pos_um.shape[0],
        post_pos_um.shape[0],
        pre_pos_um * u.um,
        post_pos_um * u.um,
    )
    pre = np.asarray(result.pre_indices, dtype=np.int32)
    post = np.asarray(result.post_indices, dtype=np.int32)
    candidates = int(pre.size)
    rng = np.random.default_rng(seed + 10_000)
    keep = rng.random(pre.size) < keep_probability
    pre, post = pre[keep], post[keep]
    diff = orientation_difference_deg(pre_pref_deg[pre], post_pref_deg[post])
    similarity = 0.5 + 0.5 * np.cos(2.0 * np.deg2rad(diff))
    weights = base_weight_nS * (1.0 + feature_strength * (similarity - 0.5))
    indptr, indices, order = brainevent.coo2csr(
        jnp.asarray(pre), jnp.asarray(post), shape=(pre_pos_um.shape[0], post_pos_um.shape[0])
    )
    conn = brainevent.CSR(
        (jnp.asarray(weights[order], dtype=jnp.float32) * u.nS, indices, indptr),
        shape=(pre_pos_um.shape[0], post_pos_um.shape[0]),
    )
    distances = np.linalg.norm(pre_pos_um[pre] - post_pos_um[post], axis=1)
    metadata = {
        "candidate_edges": candidates,
        "edges": int(pre.size),
        "density": float(pre.size / (pre_pos_um.shape[0] * post_pos_um.shape[0])),
        "mean_out_degree": float(pre.size / pre_pos_um.shape[0]),
        "mean_weight_nS": float(weights.mean()) if weights.size else 0.0,
        "mean_distance_um": float(distances.mean()) if distances.size else 0.0,
    }
    return conn, metadata


class V1Network(brainstate.nn.Module):
    """Unit-aware recurrent conductance-based E/I point-neuron network."""

    def __init__(self, cfg: Config, e_pos: np.ndarray, i_pos: np.ndarray, e_pref: np.ndarray, i_pref: np.ndarray):
        super().__init__()
        self.cfg = cfg
        v_init = braintools.init.Constant(cfg.v_rest_mV * u.mV)
        neuron_common = dict(
            R=cfg.resistance_Mohm * u.Mohm,
            V_rest=cfg.v_rest_mV * u.mV,
            V_th=cfg.v_threshold_mV * u.mV,
            V_reset=cfg.v_reset_mV * u.mV,
            tau_ref=cfg.refractory_ms * u.ms,
            V_initializer=v_init,
        )
        self.E = brainpy.state.LIFRef(cfg.n_exc, tau=cfg.e_tau_ms * u.ms, **neuron_common)
        self.I = brainpy.state.LIFRef(cfg.n_inh, tau=cfg.i_tau_ms * u.ms, **neuron_common)

        specs = {
            "EE": (e_pos, e_pos, e_pref, e_pref, cfg.ee_sigma_um, cfg.ee_max_um, cfg.ee_keep,
                   cfg.ee_weight_nS, cfg.ee_feature_strength, cfg.seed + 101, False),
            "EI": (e_pos, i_pos, e_pref, i_pref, cfg.ei_sigma_um, cfg.ei_max_um, cfg.ei_keep,
                   cfg.ei_weight_nS, cfg.ei_feature_strength, cfg.seed + 102, True),
            "IE": (i_pos, e_pos, i_pref, e_pref, cfg.ie_sigma_um, cfg.ie_max_um, cfg.ie_keep,
                   cfg.ie_weight_nS, cfg.ie_feature_strength, cfg.seed + 103, True),
            "II": (i_pos, i_pos, i_pref, i_pref, cfg.ii_sigma_um, cfg.ii_max_um, cfg.ii_keep,
                   cfg.ii_weight_nS, cfg.ii_feature_strength, cfg.seed + 104, False),
        }
        made = {name: _connection_csr(*args) for name, args in specs.items()}
        self.connectivity = {name: made[name][0] for name in PROJECTION_NAMES}
        self.connectivity_metadata = {name: made[name][1] for name in PROJECTION_NAMES}

        self.ee_syn = brainpy.state.Expon(cfg.n_exc, tau=cfg.exc_syn_tau_ms * u.ms)
        self.ei_syn = brainpy.state.Expon(cfg.n_inh, tau=cfg.exc_syn_tau_ms * u.ms)
        self.ie_syn = brainpy.state.Expon(cfg.n_exc, tau=cfg.inh_syn_tau_ms * u.ms)
        self.ii_syn = brainpy.state.Expon(cfg.n_inh, tau=cfg.inh_syn_tau_ms * u.ms)
        self.ee_out = brainpy.state.COBA(E=cfg.exc_reversal_mV * u.mV)
        self.ei_out = brainpy.state.COBA(E=cfg.exc_reversal_mV * u.mV)
        self.ie_out = brainpy.state.COBA(E=cfg.inh_reversal_mV * u.mV)
        self.ii_out = brainpy.state.COBA(E=cfg.inh_reversal_mV * u.mV)
        self.E.add_current_input("ee", self.ee_out)
        self.E.add_current_input("ie", self.ie_out)
        self.I.add_current_input("ei", self.ei_out)
        self.I.add_current_input("ii", self.ii_out)

    def update(self, t, e_current, i_current):
        with brainstate.environ.context(t=t):
            e_spike = self.E.get_spike() != 0.0
            i_spike = self.I.get_spike() != 0.0
            self.ee_out.bind_cond(self.ee_syn(brainevent.BinaryArray(e_spike) @ self.connectivity["EE"]))
            self.ei_out.bind_cond(self.ei_syn(brainevent.BinaryArray(e_spike) @ self.connectivity["EI"]))
            self.ie_out.bind_cond(self.ie_syn(brainevent.BinaryArray(i_spike) @ self.connectivity["IE"]))
            self.ii_out.bind_cond(self.ii_syn(brainevent.BinaryArray(i_spike) @ self.connectivity["II"]))
            return self.E(e_current), self.I(i_current)


def photostim_waveform(cfg: Config) -> np.ndarray:
    n_steps = int(round(cfg.trial_ms / cfg.dt_ms))
    wave = np.zeros(n_steps, dtype=np.float32)
    period_ms = 1000.0 / cfg.photo_frequency_hz
    for sweep in range(cfg.photo_sweeps):
        start_ms = cfg.pre_ms + sweep * period_ms
        start = int(round(start_ms / cfg.dt_ms))
        stop = int(round((start_ms + cfg.photo_sweep_ms) / cfg.dt_ms))
        wave[start:stop] = cfg.photo_current_nA
    return wave


def _colored_noise(rng: np.random.Generator, steps: int, trials: int, neurons: int, cfg: Config) -> np.ndarray:
    a = np.exp(-cfg.dt_ms / cfg.noise_tau_ms)
    scale = cfg.noise_sd_nA * np.sqrt(1.0 - a * a)
    individual = lfilter([scale], [1.0, -a], rng.standard_normal((steps, trials, neurons)), axis=0)
    common = lfilter([scale], [1.0, -a], rng.standard_normal((steps, trials, 1)), axis=0)
    f = cfg.common_noise_fraction
    return (np.sqrt(1.0 - f) * individual + np.sqrt(f) * common).astype(np.float32)


def trial_currents(
    cfg: Config,
    stimulus_deg: np.ndarray,
    e_pref: np.ndarray,
    i_pref: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stimulus_deg = np.asarray(stimulus_deg, dtype=float)
    trials = stimulus_deg.size
    steps = int(round(cfg.trial_ms / cfg.dt_ms))
    gain = np.clip(rng.normal(1.0, cfg.gain_sd, size=trials), cfg.gain_min, cfg.gain_max)
    e_tuning = tuning_kernel(stimulus_deg, e_pref, cfg.e_tuning_kappa)
    i_tuning = tuning_kernel(stimulus_deg, i_pref, cfg.i_tuning_kappa)
    e_visual = gain[:, None] * (cfg.e_visual_untuned_nA + cfg.e_visual_tuned_nA * e_tuning)
    i_visual = gain[:, None] * (cfg.i_visual_untuned_nA + cfg.i_visual_tuned_nA * i_tuning)
    e_current = cfg.e_background_nA + _colored_noise(rng, steps, trials, cfg.n_exc, cfg)
    i_current = cfg.i_background_nA + _colored_noise(rng, steps, trials, cfg.n_inh, cfg)
    onset = int(round(cfg.pre_ms / cfg.dt_ms))
    e_current[onset:] += e_visual[None, :, :]
    i_current[onset:] += i_visual[None, :, :]
    return e_current.astype(np.float32), i_current.astype(np.float32), gain


class Simulator:
    def __init__(self, cfg: Config, net: V1Network, batch_size: int):
        self.cfg = cfg
        self.net = net
        self.batch_size = batch_size
        self.times = u.math.arange(0.0 * u.ms, cfg.trial_ms * u.ms, cfg.dt_ms * u.ms)
        self.response_start = int(round(cfg.pre_ms / cfg.dt_ms))
        self.response_stop = int(round((cfg.pre_ms + cfg.response_ms) / cfg.dt_ms))
        self.dose_stop = int(round((cfg.pre_ms + cfg.photo_dose_window_ms) / cfg.dt_ms))
        self.pre_stop = self.response_start

        @brainstate.transform.jit
        def rollout(e_current, i_current, photo_wave, target_mask, perturb_mask):
            def step(t, e_sample, i_sample, photo_sample):
                target_current = photo_sample * perturb_mask[:, None] * target_mask[None, :]
                return net.update(t, e_sample + target_current, i_sample)

            e_spikes, i_spikes = brainstate.transform.for_loop(
                step, self.times, e_current, i_current, photo_wave
            )
            return (
                u.math.sum(e_spikes[self.response_start:self.response_stop], axis=0),
                u.math.sum(i_spikes[self.response_start:self.response_stop], axis=0),
                u.math.sum(e_spikes[self.response_start:self.dose_stop], axis=0),
                # Lossless trajectory audit with substantially smaller host transfer.
                jnp.packbits(e_spikes[:self.pre_stop] != 0, axis=-1),
                jnp.packbits(i_spikes[:self.pre_stop] != 0, axis=-1),
            )

        self._rollout = rollout

    def run(
        self,
        e_current_nA: np.ndarray,
        i_current_nA: np.ndarray,
        target: int | None = None,
        perturb_second_half: bool = False,
    ) -> tuple[np.ndarray, ...]:
        assert e_current_nA.shape[1] == self.batch_size
        brainstate.nn.init_all_states(self.net, batch_size=self.batch_size)
        target_mask = np.zeros(self.cfg.n_exc, dtype=np.float32)
        if target is not None:
            target_mask[target] = 1.0
        perturb_mask = np.zeros(self.batch_size, dtype=np.float32)
        if perturb_second_half:
            perturb_mask[self.batch_size // 2:] = 1.0
        outputs = self._rollout(
            jnp.asarray(e_current_nA) * u.nA,
            jnp.asarray(i_current_nA) * u.nA,
            jnp.asarray(photostim_waveform(self.cfg)) * u.nA,
            jnp.asarray(target_mask),
            jnp.asarray(perturb_mask),
        )
        return tuple(np.asarray(x) for x in outputs)

    def run_matched_pair(
        self,
        e_current_nA: np.ndarray,
        i_current_nA: np.ndarray,
        target: int,
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        """Run baseline and perturbation from one exact recurrent-state snapshot."""
        assert e_current_nA.shape[1] == self.batch_size
        brainstate.nn.init_all_states(self.net, batch_size=self.batch_size)
        state_snapshot = {
            path: state.value
            for path, state in self.net.states().items()
        }

        target_mask = np.zeros(self.cfg.n_exc, dtype=np.float32)
        target_mask[target] = 1.0
        photo_wave = jnp.asarray(photostim_waveform(self.cfg)) * u.nA
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
            raise ValueError(
                f"state restore mismatch: unexpected={unexpected}, missing={missing}"
            )
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


def choose_targets(cfg: Config, e_pos: np.ndarray) -> np.ndarray:
    """Outcome-blind selection: random interior excitatory cells."""
    margin = 0.15 * cfg.field_size_um
    candidates = np.flatnonzero(np.all((e_pos >= margin) & (e_pos <= cfg.field_size_um - margin), axis=1))
    if candidates.size < cfg.n_targets:
        candidates = np.arange(cfg.n_exc)
    rng = np.random.default_rng(cfg.seed + 400)
    return np.sort(rng.choice(candidates, size=cfg.n_targets, replace=False))


def run_trials(cfg: Config, net: V1Network, e_pref: np.ndarray, i_pref: np.ndarray, targets: np.ndarray) -> dict:
    directions = np.asarray(cfg.directions_deg)
    tuning_stimuli = np.tile(directions, cfg.tuning_repeats)
    mapping_stimuli = np.tile(directions, cfg.influence_repeats)
    assert tuning_stimuli.size == 2 * mapping_stimuli.size
    tuning_simulator = Simulator(cfg, net, batch_size=tuning_stimuli.size)

    tuning_rng = np.random.default_rng(cfg.seed + 500)
    tuning_e, tuning_i, tuning_gain = trial_currents(cfg, tuning_stimuli, e_pref, i_pref, tuning_rng)
    tuning_counts_e, tuning_counts_i, _, _, _ = tuning_simulator.run(tuning_e, tuning_i)

    n_mapping = mapping_stimuli.size
    mapping_simulator = tuning_simulator
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
        e_base, i_base, gains = trial_currents(cfg, mapping_stimuli, e_pref, i_pref, rng)
        pairing["noise_and_stimulus_max_abs_difference_nA"] = max(
            pairing["noise_and_stimulus_max_abs_difference_nA"],
            float(np.max(np.abs(e_base - e_base.copy()))),
            float(np.max(np.abs(i_base - i_base.copy()))),
        )
        # Keep one static JAX batch shape. The copied lanes are computational
        # padding only; all reported mapping values come from the first half.
        e_padded = np.concatenate([e_base, e_base], axis=1)
        i_padded = np.concatenate([i_base, i_base], axis=1)
        baseline, perturbed = mapping_simulator.run_matched_pair(
            e_padded, i_padded, target=int(target)
        )
        base_e, base_i, base_dose, base_pre_e, base_pre_i = baseline
        pert_e, pert_i, pert_dose, pert_pre_e, pert_pre_i = perturbed
        pairing["pre_photostim_exc_spike_mismatches"] += int(np.count_nonzero(base_pre_e[:, :n_mapping] != pert_pre_e[:, :n_mapping]))
        pairing["pre_photostim_inh_spike_mismatches"] += int(np.count_nonzero(base_pre_i[:, :n_mapping] != pert_pre_i[:, :n_mapping]))
        pairing["pairs"] += n_mapping
        all_baseline_e.append(base_e[:n_mapping])
        all_perturbed_e.append(pert_e[:n_mapping])
        all_baseline_i.append(base_i[:n_mapping])
        all_perturbed_i.append(pert_i[:n_mapping])
        all_dose_base.append(base_dose[:n_mapping, target])
        all_dose_pert.append(pert_dose[:n_mapping, target])
        mapping_gains.append(gains)

    if pairing["noise_and_stimulus_max_abs_difference_nA"] != 0.0:
        raise AssertionError("paired external inputs differ")
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


def estimate_tuning(cfg: Config, stimuli: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    directions = np.asarray(cfg.directions_deg)
    means = np.stack([counts[stimuli == direction].mean(axis=0) for direction in directions], axis=1)
    preferred = directions[np.argmax(means, axis=1)]
    return means, preferred


def build_pair_table(
    cfg: Config,
    targets: np.ndarray,
    e_pos: np.ndarray,
    tuning: np.ndarray,
    delta_e: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = {k: [] for k in ("target_order", "target", "neighbor", "distance_um", "signal_correlation", "influence_spikes")}
    for target_order, target in enumerate(targets):
        neighbors = np.arange(cfg.n_exc)
        distances = np.linalg.norm(e_pos - e_pos[target], axis=1)
        keep = (neighbors != target) & (distances >= cfg.exclude_distance_um)
        neighbors = neighbors[keep]
        corr = correlation_rows(
            np.broadcast_to(tuning[target], (neighbors.size, tuning.shape[1])), tuning[neighbors]
        )
        influence = delta_e[target_order, :, neighbors].mean(axis=1)
        values = (np.full(neighbors.size, target_order), np.full(neighbors.size, target), neighbors,
                  distances[keep], corr, influence)
        for key, value in zip(rows, values):
            rows[key].append(np.asarray(value))
    return {key: np.concatenate(value) for key, value in rows.items()}


def cluster_bootstrap(
    table: dict[str, np.ndarray],
    n_clusters: int,
    samples: int,
    rng: np.random.Generator,
    statistic: Callable[[dict[str, np.ndarray]], np.ndarray | float],
) -> np.ndarray:
    indices_by_target = [np.flatnonzero(table["target_order"] == i) for i in range(n_clusters)]
    output = []
    for _ in range(samples):
        selected = rng.integers(0, n_clusters, size=n_clusters)
        indices = np.concatenate([indices_by_target[i] for i in selected])
        output.append(statistic({key: value[indices] for key, value in table.items()}))
    return np.asarray(output)


def analyze(cfg: Config, targets: np.ndarray, e_pos: np.ndarray, trials: dict) -> tuple[dict, dict[str, np.ndarray]]:
    tuning, preferred = estimate_tuning(cfg, trials["tuning_stimuli"], trials["tuning_counts_e"])
    delta_e = trials["perturbed_e"] - trials["baseline_e"]
    delta_i = trials["perturbed_i"] - trials["baseline_i"]
    table = build_pair_table(cfg, targets, e_pos, tuning, delta_e)
    rng = np.random.default_rng(cfg.seed + 900)
    b = cfg.bootstrap_samples

    mean_influence = float(table["influence_spikes"].mean())
    mean_boot = cluster_bootstrap(table, len(targets), b, rng, lambda x: x["influence_spikes"].mean())

    edges = np.asarray(cfg.distance_edges_um)
    distance_bin = np.digitize(table["distance_um"], edges[1:-1], right=False)
    table["distance_bin"] = distance_bin

    def bin_means(data):
        return np.array([data["influence_spikes"][data["distance_bin"] == i].mean() for i in range(3)])

    distance_means = bin_means(table)
    distance_boot = cluster_bootstrap(table, len(targets), b, rng, bin_means)
    center_contrasts = np.array([distance_means[0] - distance_means[1], distance_means[2] - distance_means[1]])
    center_boot = np.column_stack([
        distance_boot[:, 0] - distance_boot[:, 1], distance_boot[:, 2] - distance_boot[:, 1]
    ])

    corr_z = standardized(table["signal_correlation"])
    design = np.column_stack([piecewise_distance_design(table["distance_um"]), corr_z])
    coefficient = float(np.linalg.lstsq(design, table["influence_spikes"], rcond=None)[0][-1])

    def signal_coefficient(data):
        x = np.column_stack([
            piecewise_distance_design(data["distance_um"]),
            standardized(data["signal_correlation"]),
        ])
        return np.linalg.lstsq(x, data["influence_spikes"], rcond=None)[0][-1]

    coefficient_boot = cluster_bootstrap(table, len(targets), b, rng, signal_coefficient)

    # Stimulus-match analysis uses target-level population changes, excluding
    # the target and <25 um neighbors exactly as in the pair table.
    directions = trials["mapping_stimuli"]
    match_rows = {k: [] for k in ("target_order", "difference_deg", "population_delta_spikes")}
    for target_order, target in enumerate(targets):
        distance = np.linalg.norm(e_pos - e_pos[target], axis=1)
        keep = (np.arange(cfg.n_exc) != target) & (distance >= cfg.exclude_distance_um)
        population_delta = delta_e[target_order][:, keep].mean(axis=1)
        diff = orientation_difference_deg(directions, preferred[target])
        match_rows["target_order"].append(np.full(directions.size, target_order))
        match_rows["difference_deg"].append(diff)
        match_rows["population_delta_spikes"].append(population_delta)
    match_table = {key: np.concatenate(value) for key, value in match_rows.items()}
    match_bins = np.array([0.0, 45.0, 90.0])

    def match_means(data):
        return np.array([
            data["population_delta_spikes"][np.isclose(data["difference_deg"], angle)].mean()
            for angle in match_bins
        ])

    match_mean = match_means(match_table)
    match_boot = cluster_bootstrap(match_table, len(targets), b, rng, match_means)
    match_contrast = float(match_mean[2] - match_mean[0])
    match_contrast_boot = match_boot[:, 2] - match_boot[:, 0]

    response_s = cfg.response_ms / 1000.0
    tuning_e_rate = trials["tuning_counts_e"].mean(axis=0) / response_s
    tuning_i_rate = trials["tuning_counts_i"].mean(axis=0) / response_s
    mapping_e_rate = trials["baseline_e"].mean(axis=(0, 1)) / response_s
    mapping_i_rate = trials["baseline_i"].mean(axis=(0, 1)) / response_s
    dose_delta = trials["dose_perturbed"] - trials["dose_baseline"]
    dose_by_target = dose_delta.mean(axis=1)
    dose_boot = np.array([
        dose_by_target[rng.integers(0, len(targets), size=len(targets))].mean() for _ in range(b)
    ])

    tests = {
        "mean_suppression": bool(percentile_ci(mean_boot, cfg.alpha)[1] < 0.0),
        "center_surround": bool(np.all(np.quantile(center_boot, cfg.alpha / 2.0, axis=0) > 0.0)),
        "negative_signal_correlation_after_distance": bool(percentile_ci(coefficient_boot, cfg.alpha)[1] < 0.0),
        "strongest_suppression_for_matching_stimulus": bool(percentile_ci(match_contrast_boot, cfg.alpha)[0] > 0.0),
    }
    experimental_dose_range = [5.37, 7.39]  # published mean +/- one reported SEM
    measured_dose = float(dose_delta.mean())
    tests["dose_within_published_mean_plus_minus_sem"] = bool(
        experimental_dose_range[0] <= measured_dose <= experimental_dose_range[1]
    )

    results = {
        "status": "completed_frozen_run",
        "tests": tests,
        "sample_counts": {
            "excitatory_neurons": cfg.n_exc,
            "inhibitory_neurons": cfg.n_inh,
            "targets": int(len(targets)),
            "tuning_trials": int(trials["tuning_stimuli"].size),
            "matched_trial_pairs": int(len(targets) * trials["mapping_stimuli"].size),
            "eligible_target_neighbor_pairs": int(table["target"].size),
            "pair_observations_before_averaging": int(table["target"].size * trials["mapping_stimuli"].size),
            "bootstrap_target_resamples": b,
        },
        "baseline_firing_rates_hz": {
            "tuning_excitatory": _distribution_summary(tuning_e_rate),
            "tuning_inhibitory": _distribution_summary(tuning_i_rate),
            "mapping_excitatory": _distribution_summary(mapping_e_rate),
            "mapping_inhibitory": _distribution_summary(mapping_i_rate),
        },
        "photostimulation": {
            "intended_model_protocol": {
                "sweeps": cfg.photo_sweeps,
                "sweep_ms": cfg.photo_sweep_ms,
                "frequency_hz": cfg.photo_frequency_hz,
                "somatic_square_current_nA": cfg.photo_current_nA,
                "onset_relative_to_visual_ms": 0.0,
            },
            "published_benchmark": {
                "added_spikes_mean": 6.38,
                "added_spikes_sem": 1.01,
                "cells": 9,
                "window_description": "four sweeps over approximately 250 ms",
            },
            "measured_added_target_spikes_250ms": {
                "mean": measured_dose,
                "ci95_target_bootstrap": percentile_ci(dose_boot, cfg.alpha),
                "sd_across_all_trials": float(dose_delta.std(ddof=1)),
                "trials": int(dose_delta.size),
            },
        },
        "effects": {
            "mean_influence_spikes_per_367ms": {
                "estimate": mean_influence,
                "ci95_target_bootstrap": percentile_ci(mean_boot, cfg.alpha),
            },
            "mean_influence_rate_change_hz": {
                "estimate": mean_influence / response_s,
                "ci95_target_bootstrap": [x / response_s for x in percentile_ci(mean_boot, cfg.alpha)],
            },
            "distance_bins": [
                {
                    "range_um": label,
                    "pairs": int(np.sum(distance_bin == i)),
                    "targets_represented": int(np.unique(table["target_order"][distance_bin == i]).size),
                    "mean_influence_spikes": float(distance_means[i]),
                    "ci95_target_bootstrap": percentile_ci(distance_boot[:, i], cfg.alpha),
                }
                for i, label in enumerate(("25-100", "100-300", ">=300"))
            ],
            "center_surround_contrasts_spikes": {
                "near_minus_middle": {
                    "estimate": float(center_contrasts[0]),
                    "ci95_target_bootstrap": percentile_ci(center_boot[:, 0], cfg.alpha),
                },
                "far_minus_middle": {
                    "estimate": float(center_contrasts[1]),
                    "ci95_target_bootstrap": percentile_ci(center_boot[:, 1], cfg.alpha),
                },
            },
            "signal_correlation_regression": {
                "coefficient_spikes_per_1sd_signal_correlation": coefficient,
                "ci95_target_bootstrap": percentile_ci(coefficient_boot, cfg.alpha),
                "distance_adjustment": "continuous segmented basis with knots at 100 and 300 um",
                "pairs": int(table["target"].size),
            },
            "stimulus_match_bins": [
                {
                    "orientation_difference_deg": float(angle),
                    "target_trial_observations": int(np.sum(np.isclose(match_table["difference_deg"], angle))),
                    "mean_population_delta_spikes_per_neuron": float(match_mean[i]),
                    "ci95_target_bootstrap": percentile_ci(match_boot[:, i], cfg.alpha),
                }
                for i, angle in enumerate(match_bins)
            ],
            "orthogonal_minus_match_spikes": {
                "estimate": match_contrast,
                "ci95_target_bootstrap": percentile_ci(match_contrast_boot, cfg.alpha),
            },
            "mean_inhibitory_population_delta_spikes_per_neuron": float(delta_i.mean()),
        },
        "pairing_checks": trials["pairing"],
        "uncertainty_scope": (
            "Percentile intervals resample photostimulated targets. They quantify target-sampling "
            "uncertainty within one frozen network realization, not model, seed, or parameter uncertainty."
        ),
    }
    arrays = {
        "tuning": tuning,
        "preferred_deg": preferred,
        "delta_e": delta_e,
        "pair_table": table,
        "match_table": match_table,
        "distance_boot": distance_boot,
        "match_boot": match_boot,
        "coefficient_boot": coefficient_boot,
    }
    return results, arrays


def _distribution_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "silent_fraction": float(np.mean(values == 0.0)),
        "neurons": int(values.size),
    }


def write_pair_csv(path: Path, table: dict[str, np.ndarray]) -> None:
    keys = ("target_order", "target", "neighbor", "distance_um", "signal_correlation", "influence_spikes", "distance_bin")
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        writer.writerows(zip(*(table[key] for key in keys)))


def plot_results(path: Path, cfg: Config, results: dict, arrays: dict[str, np.ndarray]) -> None:
    table = arrays["pair_table"]
    distance = results["effects"]["distance_bins"]
    match = results["effects"]["stimulus_match_bins"]
    coefficient = results["effects"]["signal_correlation_regression"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))

    axes[0, 0].hist(
        arrays["delta_e"].mean(axis=(0, 1)) / (cfg.response_ms / 1000.0), bins=30,
        color="#4c6a70", edgecolor="white"
    )
    axes[0, 0].axvline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set(xlabel="mean response change (Hz)", ylabel="excitatory neurons", title="Influence distribution")

    x = np.arange(3)
    y = np.array([row["mean_influence_spikes"] for row in distance])
    lo = np.array([row["ci95_target_bootstrap"][0] for row in distance])
    hi = np.array([row["ci95_target_bootstrap"][1] for row in distance])
    axes[0, 1].errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), fmt="o-", color="#b5483a", capsize=3)
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set(xticks=x, xticklabels=[row["range_um"] for row in distance], xlabel="distance (um)",
                   ylabel="change (spikes / 367 ms)", title="Distance profile")

    corr_edges = np.quantile(table["signal_correlation"], np.linspace(0, 1, 7))
    corr_bin = np.clip(np.digitize(table["signal_correlation"], corr_edges[1:-1]), 0, 5)
    corr_x = np.array([table["signal_correlation"][corr_bin == i].mean() for i in range(6)])
    corr_y = np.array([table["influence_spikes"][corr_bin == i].mean() for i in range(6)])
    axes[1, 0].plot(corr_x, corr_y, "o-", color="#4169a1")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set(xlabel="signal correlation", ylabel="change (spikes / 367 ms)",
                   title=f"Signal coefficient = {coefficient['coefficient_spikes_per_1sd_signal_correlation']:.3f}")

    mx = np.array([row["orientation_difference_deg"] for row in match])
    my = np.array([row["mean_population_delta_spikes_per_neuron"] for row in match])
    mlo = np.array([row["ci95_target_bootstrap"][0] for row in match])
    mhi = np.array([row["ci95_target_bootstrap"][1] for row in match])
    axes[1, 1].errorbar(mx, my, yerr=np.vstack([my - mlo, mhi - my]), fmt="o-", color="#4e7d4a", capsize=3)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set(xticks=mx, xlabel="stimulus-target orientation difference (deg)",
                   ylabel="population change (spikes / neuron)", title="Stimulus match")

    fig.suptitle("Frozen single-neuron influence-mapping run")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, cfg: Config, results: dict, connectivity: dict) -> None:
    tests = results["tests"]
    effect = results["effects"]
    rate = results["baseline_firing_rates_hz"]
    photo = results["photostimulation"]
    lines = [
        "# Frozen V1 single-cell influence-mapping run",
        "",
        "This is one preregistered point-neuron model realization. No parameter search or rerun was used to obtain the signs below.",
        "",
        "## Outcome",
        "",
    ]
    labels = {
        "mean_suppression": "Suppressive influence on average",
        "center_surround": "Local center-surround distance profile",
        "negative_signal_correlation_after_distance": "More negative influence with signal correlation after distance adjustment",
        "strongest_suppression_for_matching_stimulus": "Strongest suppression for target-matched stimulus",
        "dose_within_published_mean_plus_minus_sem": "Target dose within published mean +/- SEM",
    }
    lines.extend([f"- {'PASS' if tests[key] else 'FAIL'}: {labels[key]}" for key in labels])
    mean = effect["mean_influence_spikes_per_367ms"]
    coeff = effect["signal_correlation_regression"]
    contrast = effect["orthogonal_minus_match_spikes"]
    measured = photo["measured_added_target_spikes_250ms"]
    lines += [
        "",
        "## Effect sizes",
        "",
        f"Mean non-target E influence: {mean['estimate']:.4f} spikes per 367 ms "
        f"(95% target-bootstrap CI {mean['ci95_target_bootstrap'][0]:.4f}, {mean['ci95_target_bootstrap'][1]:.4f}).",
        f"Distance-adjusted signal-correlation coefficient: {coeff['coefficient_spikes_per_1sd_signal_correlation']:.4f} "
        f"spikes per 1 SD (95% CI {coeff['ci95_target_bootstrap'][0]:.4f}, {coeff['ci95_target_bootstrap'][1]:.4f}).",
        f"Orthogonal minus matched stimulus contrast: {contrast['estimate']:.4f} spikes per non-target neuron "
        f"(95% CI {contrast['ci95_target_bootstrap'][0]:.4f}, {contrast['ci95_target_bootstrap'][1]:.4f}).",
        "",
        "Distance bins:",
        "",
    ]
    for row in effect["distance_bins"]:
        lines.append(
            f"- {row['range_um']} um: {row['mean_influence_spikes']:.4f} spikes "
            f"(95% CI {row['ci95_target_bootstrap'][0]:.4f}, {row['ci95_target_bootstrap'][1]:.4f}; "
            f"{row['pairs']} pairs, {row['targets_represented']} targets)"
        )
    lines += [
        "",
        "Stimulus-match bins:",
        "",
    ]
    for row in effect["stimulus_match_bins"]:
        lines.append(
            f"- {row['orientation_difference_deg']:.0f} deg: {row['mean_population_delta_spikes_per_neuron']:.4f} "
            f"spikes (95% CI {row['ci95_target_bootstrap'][0]:.4f}, {row['ci95_target_bootstrap'][1]:.4f}; "
            f"{row['target_trial_observations']} target-trial observations)"
        )
    lines += [
        "",
        "## Protocol and controls",
        "",
        f"The model delivered four {cfg.photo_sweep_ms:.0f}-ms somatic square-current sweeps at "
        f"{cfg.photo_frequency_hz:.0f} Hz, starting at visual onset. Current amplitude was frozen at "
        f"{cfg.photo_current_nA:.3f} nA. The measured target increase was {measured['mean']:.3f} spikes in 250 ms "
        f"(95% target-bootstrap CI {measured['ci95_target_bootstrap'][0]:.3f}, "
        f"{measured['ci95_target_bootstrap'][1]:.3f}; {measured['trials']} trials). The published cell-attached "
        "benchmark was 6.38 +/- 1.01 added spikes (mean +/- SEM, n=9 cells).",
        "",
        f"All {results['pairing_checks']['pairs']} baseline-perturbation pairs used bit-identical external input arrays. "
        f"Pre-perturbation E and I spike-count mismatches were "
        f"{results['pairing_checks']['pre_photostim_exc_spike_mismatches']} and "
        f"{results['pairing_checks']['pre_photostim_inh_spike_mismatches']}.",
        "",
        "## Baseline activity",
        "",
        f"Unperturbed tuning trials: E mean {rate['tuning_excitatory']['mean']:.2f} Hz "
        f"(median {rate['tuning_excitatory']['median']:.2f}, IQR {rate['tuning_excitatory']['q25']:.2f}-"
        f"{rate['tuning_excitatory']['q75']:.2f}); I mean {rate['tuning_inhibitory']['mean']:.2f} Hz "
        f"(median {rate['tuning_inhibitory']['median']:.2f}).",
        f"Matched mapping baselines: E mean {rate['mapping_excitatory']['mean']:.2f} Hz; "
        f"I mean {rate['mapping_inhibitory']['mean']:.2f} Hz.",
        "",
        "## Model scope and assumptions",
        "",
        "The model contains conductance-based LIF point neurons, not dendrites. The experiment tests somatic "
        "single-cell perturbation and population spiking, so morphology and subcellular input location are not "
        "represented mechanisms in this first reproduction.",
        "",
        "Phenomenological assumptions are explicit: orientation-tuned feedforward current; Gaussian spatial "
        "candidate connectivity; sparse thinning; and preference-modulated synaptic weights. The broader IE than "
        "EE spatial scale and preference-modulated E-I-E pathway are hypotheses, not direct anatomical fits. "
        "Weights were frozen before the influence outcome was observed.",
        "",
        "Connectivity summary:",
        "",
    ]
    for name in PROJECTION_NAMES:
        row = connectivity[name]
        lines.append(
            f"- {name}: {row['edges']} edges, density {row['density']:.4f}, mean out-degree "
            f"{row['mean_out_degree']:.1f}, mean weight {row['mean_weight_nS']:.3f} nS, mean distance "
            f"{row['mean_distance_um']:.1f} um"
        )
    lines += [
        "",
        "The analysis excludes target-neighbor separations below 25 um. Influence is the paired perturbed-minus-baseline "
        "spike count over 367 ms. The center-surround test requires both near-minus-middle and far-minus-middle "
        "target-bootstrap CIs to be positive. Signal correlation is computed from eight repeated-trial mean direction "
        "responses and entered with a segmented continuous distance basis. Stimulus match compares 0-deg with 90-deg "
        "orientation difference. Confidence intervals resample targets only and do not cover model-seed or parameter uncertainty.",
        "",
        "Complete frozen parameters are in `parameters.json`; pair data are in `pair_influences.csv`; exact numerical "
        "outputs and test definitions are in `results.json`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def serialize_config(cfg: Config) -> dict:
    values = asdict(cfg)
    for key, value in list(values.items()):
        if isinstance(value, tuple):
            values[key] = [None if np.isinf(x) else x for x in value]
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/frozen_run"))
    parser.add_argument("--bootstrap-samples", type=int, default=None,
                        help="analysis-only override; default is the preregistered 2000")
    args = parser.parse_args()
    cfg = Config()
    if args.bootstrap_samples is not None:
        cfg = Config(**{**asdict(cfg), "bootstrap_samples": args.bootstrap_samples})
    args.output.mkdir(parents=True, exist_ok=True)

    brainstate.environ.set(dt=cfg.dt_ms * u.ms)
    e_pos, i_pos, e_pref, i_pref = make_positions_and_preferences(cfg)
    net = V1Network(cfg, e_pos, i_pos, e_pref, i_pref)
    targets = choose_targets(cfg, e_pos)
    trials = run_trials(cfg, net, e_pref, i_pref, targets)
    results, arrays = analyze(cfg, targets, e_pos, trials)
    results["connectivity"] = net.connectivity_metadata
    results["targets"] = targets.tolist()

    (args.output / "parameters.json").write_text(
        json.dumps(serialize_config(cfg), indent=2, allow_nan=False) + "\n", encoding="ascii"
    )
    (args.output / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="ascii"
    )
    write_pair_csv(args.output / "pair_influences.csv", arrays["pair_table"])
    np.savez_compressed(
        args.output / "trial_summaries.npz",
        targets=targets,
        e_positions_um=e_pos,
        i_positions_um=i_pos,
        true_e_preference_deg=e_pref,
        estimated_e_preference_deg=arrays["preferred_deg"],
        tuning_counts=arrays["tuning"],
        influence_delta_counts=arrays["delta_e"],
    )
    plot_results(args.output / "summary.png", cfg, results, arrays)
    write_report(args.output / "REPORT.md", cfg, results, net.connectivity_metadata)
    print(json.dumps({"output": str(args.output), "tests": results["tests"]}, indent=2))


if __name__ == "__main__":
    main()
