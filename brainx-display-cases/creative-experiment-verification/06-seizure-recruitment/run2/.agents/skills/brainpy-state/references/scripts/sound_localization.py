"""Localize sound direction from interaural spike timing.

Positive interaural time difference (ITD) means the right-ear spike arrives
later, so the source is on the left. Negative ITD means the source is right.
"""

from __future__ import annotations

import brainevent
import brainpy
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import numpy as np
from brainstate.transform import vmap2
from brainstate.util import filter as util_filter


DT = 0.05 * u.ms
NUM_STEPS = 80
DURATION = NUM_STEPS * DT
SOUND_ONSET = 40 * DT

PREFERRED_DELAY_STEPS = jnp.arange(-12, 13, dtype=jnp.int32)
PREFERRED_ITDS = PREFERRED_DELAY_STEPS * DT
MAX_INTERNAL_DELAY_STEPS = 12
EVALUATION_ITDS = u.math.asarray(
    [-0.53, -0.33, -0.17, -0.075, -0.05, 0.0, 0.05, 0.075, 0.17, 0.33, 0.53]
) * u.ms

V_REST = -60.0 * u.mV
V_THRESHOLD = -50.0 * u.mV
V_RESET = -60.0 * u.mV
AUDITORY_PULSE = 27.0 * u.mA
DETECTOR_WEIGHT = 24.0 * u.mA
READOUT_WEIGHT = 120.0 * u.mA

DYNAMICAL_STATE = util_filter.Any(
    util_filter.OfType(brainstate.HiddenState),
    util_filter.OfType(brainstate.ShortTermState),
)


def _lif_population(size: int, tau, tau_ref):
    return brainpy.state.LIFRef(
        size,
        R=1.0 * u.ohm,
        tau=tau,
        tau_ref=tau_ref,
        V_rest=V_REST,
        V_th=V_THRESHOLD,
        V_reset=V_RESET,
        V_initializer=braintools.init.Constant(V_REST),
    )


class EventDelayLine(brainstate.nn.Module):
    """Small integer delay bank with step 0 equal to the current event."""

    def __init__(self, max_steps: int):
        super().__init__()
        self.max_steps = max_steps

    def init_state(self):
        self.history = brainstate.HiddenState(
            jnp.zeros((self.max_steps + 1, 2), dtype=jnp.bool_)
        )

    def update(self, events):
        self.history.value = jnp.concatenate(
            [events[None, :], self.history.value[:-1]],
            axis=0,
        )
        return self.history.value


class SoundLocalizer(brainstate.nn.Module):
    """Two auditory relays, a coincidence bank, and left/right readouts."""

    def __init__(self):
        super().__init__()
        self.num_detectors = int(PREFERRED_DELAY_STEPS.shape[0])
        self.auditory = _lif_population(2, tau=0.1 * u.ms, tau_ref=0.5 * u.ms)
        self.detectors = _lif_population(
            self.num_detectors,
            tau=0.2 * u.ms,
            tau_ref=0.5 * u.ms,
        )
        self.readout = _lif_population(2, tau=0.5 * u.ms, tau_ref=0.5 * u.ms)

        self.ear_history = EventDelayLine(MAX_INTERNAL_DELAY_STEPS)
        self.left_delay_steps = jnp.maximum(PREFERRED_DELAY_STEPS, 0)
        self.right_delay_steps = jnp.maximum(-PREFERRED_DELAY_STEPS, 0)

        detector_targets = jnp.concatenate(
            [jnp.arange(self.num_detectors), jnp.arange(self.num_detectors)]
        )[:, None]
        self.detector_connection = brainevent.FixedNumPerPre(
            DETECTOR_WEIGHT,
            detector_targets,
            shape=(2 * self.num_detectors, self.num_detectors),
        )

        readout_targets = jnp.where(PREFERRED_DELAY_STEPS > 0, 0, 1)[:, None]
        readout_weights = (
            (PREFERRED_DELAY_STEPS != 0).astype(jnp.float32)[:, None]
            * READOUT_WEIGHT
        )
        self.readout_connection = brainevent.FixedNumPerPre(
            readout_weights,
            readout_targets,
            shape=(self.num_detectors, 2),
        )

    def update(self, t, itd_step):
        with brainstate.environ.context(t=t):
            arrivals = u.math.asarray(
                [SOUND_ONSET, SOUND_ONSET + itd_step * DT]
            )
            ear_drive = (
                u.math.abs(t - arrivals) < 0.25 * brainstate.environ.get_dt()
            ) * AUDITORY_PULSE
            ear_spikes = self.auditory(ear_drive) != 0.0

            history = self.ear_history.update(ear_spikes)
            left_events = history[self.left_delay_steps, 0]
            right_events = history[self.right_delay_steps, 1]
            delayed_events = jnp.concatenate([left_events, right_events])

            detector_current = (
                brainevent.BinaryArray(delayed_events) @ self.detector_connection
            )
            detector_spikes = self.detectors(detector_current) != 0.0

            readout_current = (
                brainevent.BinaryArray(detector_spikes) @ self.readout_connection
            )
            readout_spikes = self.readout(readout_current) != 0.0
            return ear_spikes, detector_spikes, readout_spikes


def simulate_itds(itds=EVALUATION_ITDS):
    """Round unit-bearing ITDs to ``dt`` and evaluate independent conditions."""
    itds = u.math.asarray(itds, unit=u.ms)
    if itds.ndim != 1:
        raise ValueError("itds must be a one-dimensional time quantity")
    itd_steps = jnp.rint(
        itds.to_decimal(u.ms) / DT.to_decimal(u.ms)
    ).astype(jnp.int32)

    with brainstate.environ.context(dt=DT):
        net = SoundLocalizer()
        brainstate.nn.vmap_init_all_states(net, axis_size=itds.shape[0])
        mapped_step = vmap2(
            net.update,
            in_axes=(None, 0),
            out_axes=0,
            state_in_axes={0: DYNAMICAL_STATE},
            state_out_axes={0: DYNAMICAL_STATE},
            unexpected_out_state_mapping="raise",
        )
        times = u.math.arange(0.0 * u.ms, DURATION, DT)

        def step(t):
            return mapped_step(t, itd_steps)

        ear_spikes, detector_spikes, readout_spikes = (
            brainstate.transform.for_loop(step, times)
        )

    return (
        jnp.sum(ear_spikes, axis=0),
        jnp.sum(detector_spikes, axis=0),
        jnp.sum(readout_spikes, axis=0),
    )


def decode_directions(detector_counts, readout_counts):
    """Decode only rows supported by coincidence-detector evidence."""
    detectors = np.asarray(detector_counts)
    counts = np.asarray(readout_counts)
    if np.any(detectors.sum(axis=1) == 0):
        raise ValueError("cannot decode a condition with no detector evidence")

    labels = np.full(counts.shape[0], "CENTER", dtype="<U6")
    labels[counts[:, 0] > counts[:, 1]] = "LEFT"
    labels[counts[:, 1] > counts[:, 0]] = "RIGHT"

    ties = counts[:, 0] == counts[:, 1]
    winning_steps = np.asarray(PREFERRED_DELAY_STEPS)[np.argmax(detectors, axis=1)]
    if np.any(ties & (winning_steps != 0)):
        raise ValueError("readout tie is not supported by the center detector")
    return labels


def assert_delay_convention():
    """Prove that tap 3 returns an impulse after three completed updates."""
    with brainstate.environ.context(dt=DT):
        delay = EventDelayLine(max_steps=3)
        brainstate.nn.init_all_states(delay)
        impulse = jnp.zeros((5, 2), dtype=jnp.bool_).at[0, 0].set(True)
        history = brainstate.transform.for_loop(delay.update, impulse)
    expected = jnp.array([False, False, False, True, False])
    assert jnp.array_equal(history[:, 3, 0], expected)


def main():
    assert_delay_convention()
    ear_counts, detector_counts, readout_counts = simulate_itds()
    directions = decode_directions(detector_counts, readout_counts)
    evaluated = np.asarray(EVALUATION_ITDS.to_decimal(u.ms))
    preferences = np.asarray(PREFERRED_ITDS.to_decimal(u.ms))
    detectors = np.asarray(detector_counts)
    readouts = np.asarray(readout_counts)

    print(" ITD (ms) | detector (ms) | L/R spikes | direction")
    print("----------+---------------+------------+----------")
    for itd, detector_row, readout_row, direction in zip(
        evaluated, detectors, readouts, directions
    ):
        detector_itd = preferences[int(np.argmax(detector_row))]
        print(
            f"{itd:9.2f} | {detector_itd:13.2f} |"
            f" {readout_row[0]:3d}/{readout_row[1]:<3d}  | {direction}"
        )

    if not np.all(np.asarray(ear_counts) == 1):
        raise RuntimeError("each auditory relay must emit exactly one spike")
    expected = np.where(
        evaluated < 0.0,
        "RIGHT",
        np.where(evaluated > 0.0, "LEFT", "CENTER"),
    )
    if not np.array_equal(directions, expected):
        raise RuntimeError("direction decoding failed")


if __name__ == "__main__":
    main()
