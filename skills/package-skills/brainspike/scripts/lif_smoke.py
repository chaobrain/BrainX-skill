"""Run a single constant-current LIF neuron on the CPU."""

import os


def main():
    os.environ["JAX_PLATFORMS"] = "cpu"

    import brainspike
    import brainstate
    import brainunit as u

    brainstate.environ.set(precision=64)
    sim = brainspike.Simulator(dt=0.1 * u.ms)
    neuron = sim.create(brainspike.iaf_psc_alpha, 1, I_e=500.0 * u.pA)
    recorder = sim.create(brainspike.spike_recorder)
    sim.connect(neuron, recorder)

    spikes = sim.simulate(100.0 * u.ms).spikes(recorder)
    count = int(spikes.sum())
    devices = {device.platform for device in spikes.devices()}
    if spikes.shape != (1000, 1) or count <= 0 or devices != {"cpu"}:
        raise RuntimeError(
            f"Expected shape (1000, 1) and positive firing on CPU; "
            f"got shape={spikes.shape}, count={count}, devices={devices}"
        )
    print(f"PASS: shape={spikes.shape}, spikes={count}, device=cpu")


if __name__ == "__main__":
    main()
