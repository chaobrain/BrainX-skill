"""Jeffress-style sound localization from interaural spike timing.

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


DT = 0.05 * u.ms
NUM_STEPS = 80
DURATION = NUM_STEPS * DT
SOUND_ONSET = 40 * DT

# Thirteen coincidence cells prefer ITDs from -0.6 ms to +0.6 ms.
PREFERRED_DELAY_STEPS = jnp.arange(-12, 13, 2, dtype=jnp.int32)
PREFERRED_ITDS = PREFERRED_DELAY_STEPS * DT
MAX_INTERNAL_DELAY_STEPS = 12
MAX_INTERNAL_DELAY = MAX_INTERNAL_DELAY_STEPS * DT

V_REST = -60.0 * u.mV
V_THRESHOLD = -50.0 * u.mV
V_RESET = -60.0 * u.mV
AUDITORY_PULSE = 27.0 * u.mA
DETECTOR_WEIGHT = 24.0 * u.mA
READOUT_WEIGHT = 120.0 * u.mA


def _mapped_dynamics(_path, state):
    return state.value.ndim > 0


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
    """Small mapped delay line with step 0 equal to the current event."""

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

        self.ear_history = EventDelayLine(
            max_steps=MAX_INTERNAL_DELAY_STEPS,
        )
        self.left_delay_steps = jnp.maximum(PREFERRED_DELAY_STEPS, 0)
        self.right_delay_steps = jnp.maximum(-PREFERRED_DELAY_STEPS, 0)

        # Each delayed ear channel has exactly one coincidence-cell target.
        detector_targets = jnp.concatenate(
            [jnp.arange(self.num_detectors), jnp.arange(self.num_detectors)]
        )[:, None]
        self.detector_connection = brainevent.FixedNumPerPre(
            DETECTOR_WEIGHT,
            detector_targets,
            shape=(2 * self.num_detectors, self.num_detectors),
        )

        # Positive-preference cells vote left; negative-preference cells vote
        # right. The zero-ITD cell has zero readout weight, producing a tie.
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

    def update(self, t, itd):
        """Advance the complete circuit by one integration step."""
        with brainstate.environ.context(t=t):
            arrivals = u.math.asarray([SOUND_ONSET, SOUND_ONSET + itd])
            ear_drive = (
                u.math.abs(t - arrivals) < 0.5 * brainstate.environ.get_dt()
            ) * AUDITORY_PULSE
            ear_spikes = self.auditory(ear_drive) != 0.0

            # Current sample is step 0; positive preferred ITDs delay the left
            # ear, and negative preferred ITDs delay the right ear.
            history = self.ear_history.update(ear_spikes)
            left_events = history[self.left_delay_steps, 0]
            right_events = history[self.right_delay_steps, 1]
            delayed_events = jnp.concatenate([left_events, right_events])

            detector_current = (
                brainevent.BinaryArray(delayed_events)
                @ self.detector_connection
            )
            detector_spikes = self.detectors(detector_current) != 0.0

            readout_current = (
                brainevent.BinaryArray(detector_spikes)
                @ self.readout_connection
            )
            readout_spikes = self.readout(readout_current) != 0.0
            return ear_spikes, detector_spikes, readout_spikes


def simulate_itds(itds=PREFERRED_ITDS):
    """Evaluate independent unit-bearing ITDs with one state-aware vmap."""
    itds = u.math.asarray(itds, unit=u.ms)
    if itds.ndim != 1:
        raise ValueError("itds must be a one-dimensional time quantity")

    with brainstate.environ.context(dt=DT):
        net = SoundLocalizer()
        brainstate.nn.vmap_init_all_states(net, axis_size=itds.shape[0])

        mapped_step = vmap2(
            net.update,
            in_axes=(None, 0),
            out_axes=0,
            state_in_axes={0: _mapped_dynamics},
            state_out_axes={0: _mapped_dynamics},
            unexpected_out_state_mapping="raise",
        )
        times = jnp.arange(NUM_STEPS) * DT

        def step(t):
            return mapped_step(t, itds)

        ear_spikes, detector_spikes, readout_spikes = (
            brainstate.transform.for_loop(step, times)
        )

    return (
        jnp.sum(ear_spikes, axis=0),
        jnp.sum(detector_spikes, axis=0),
        jnp.sum(readout_spikes, axis=0),
    )


def decode_directions(readout_counts):
    """Convert two-neuron spike counts into LEFT, CENTER, or RIGHT labels."""
    counts = np.asarray(readout_counts)
    labels = np.full(counts.shape[0], "CENTER", dtype="<U6")
    labels[counts[:, 0] > counts[:, 1]] = "LEFT"
    labels[counts[:, 1] > counts[:, 0]] = "RIGHT"
    return labels


def main():
    ear_counts, detector_counts, readout_counts = simulate_itds()
    directions = decode_directions(readout_counts)
    preferred = np.asarray(PREFERRED_ITDS.to_decimal(u.ms))
    detector_counts = np.asarray(detector_counts)
    readout_counts = np.asarray(readout_counts)

    print(" ITD (ms) | detector (ms) | L/R spikes | direction")
    print("----------+---------------+------------+----------")
    for itd, detector_row, readout_row, direction in zip(
        preferred,
        detector_counts,
        readout_counts,
        directions,
    ):
        detector_itd = preferred[int(np.argmax(detector_row))]
        print(
            f"{itd:9.2f} | {detector_itd:13.2f} |"
            f" {readout_row[0]:3d}/{readout_row[1]:<3d}  | {direction}"
        )

    if not np.all(np.asarray(ear_counts) == 1):
        raise RuntimeError("each auditory relay must emit exactly one spike")


if __name__ == "__main__":
    main()
