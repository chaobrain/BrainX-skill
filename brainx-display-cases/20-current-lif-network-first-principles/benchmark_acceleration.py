from __future__ import annotations

import json
import time
from pathlib import Path

import brainstate
import brainunit as u
import jax
import numpy as np

import lif_network


def synchronized_rollout(rollout):
    started = time.perf_counter()
    output = rollout()
    jax.block_until_ready(output[2])
    return output, time.perf_counter() - started


def main() -> None:
    config = lif_network.load_config()
    model = {
        **config["model"],
        "NE": 800,
        "NI": 200,
        "CE": 80,
        "CI": 20,
    }
    protocol = {
        **config["protocol"],
        "duration_ms": 200.0,
        "transient_ms": 50.0,
        "sample_size": 20,
    }
    benchmark_config = {**config, "model": model, "protocol": protocol}
    structures = lif_network.build_random_structures(
        11,
        ne=800,
        ni=200,
        ce=80,
        ci=20,
        sample_size=20,
        reset_mV=10.0,
        threshold_mV=20.0,
    )
    with brainstate.environ.context(dt=0.1 * u.ms, fit=False):
        network = lif_network.CurrentLIFNetwork(benchmark_config, structures)
        brainstate.nn.init_all_states(network)
        rollout = lif_network.create_rollout(network, benchmark_config)

        network.set_condition(3.0, 2.0, benchmark_config)
        network.reset()
        cold, cold_seconds = synchronized_rollout(rollout)
        cold_arrays = tuple(np.asarray(value) for value in cold)
        cold_voltage = np.asarray(network.neurons.V.value.to_decimal(u.mV))

        network.reset()
        warm, warm_seconds = synchronized_rollout(rollout)
        warm_arrays = tuple(np.asarray(value) for value in warm)
        warm_voltage = np.asarray(network.neurons.V.value.to_decimal(u.mV))

        network.set_condition(6.0, 4.0, benchmark_config)
        network.reset()
        _, changed_condition_seconds = synchronized_rollout(rollout)

    output_parity = all(
        np.array_equal(first, repeated)
        for first, repeated in zip(cold_arrays, warm_arrays, strict=True)
    )
    voltage_parity = bool(np.array_equal(cold_voltage, warm_voltage))
    result = {
        "network": {"NE": 800, "NI": 200, "CE": 80, "CI": 20},
        "duration_ms": 200.0,
        "steps": 2000,
        "backend": jax.default_backend(),
        "cold_compile_and_run_seconds": cold_seconds,
        "warm_replay_seconds": warm_seconds,
        "warm_changed_condition_seconds": changed_condition_seconds,
        "warm_speedup_over_cold": cold_seconds / warm_seconds,
        "exact_output_replay": output_parity,
        "exact_final_voltage_replay": voltage_parity,
        "output_shapes": [list(value.shape) for value in cold_arrays],
    }
    if not output_parity or not voltage_parity:
        raise RuntimeError("acceleration parity check failed")
    path = Path(__file__).with_name("acceleration_parity-iteration2.json")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
