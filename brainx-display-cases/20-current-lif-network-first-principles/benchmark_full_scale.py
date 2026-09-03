from __future__ import annotations

import json
import time
from pathlib import Path

import brainstate
import brainunit as u
import jax

import lif_network


def main() -> None:
    config = lif_network.load_config()
    benchmark_config = {
        **config,
        "protocol": {
            **config["protocol"],
            "duration_ms": 20.0,
            "transient_ms": 5.0,
        },
    }
    started = time.perf_counter()
    structures = lif_network.build_random_structures(
        11,
        ne=10_000,
        ni=2_500,
        ce=1_000,
        ci=250,
        sample_size=50,
        reset_mV=10.0,
        threshold_mV=20.0,
    )
    construction_seconds = time.perf_counter() - started
    with brainstate.environ.context(dt=0.1 * u.ms, fit=False):
        network = lif_network.CurrentLIFNetwork(benchmark_config, structures)
        brainstate.nn.init_all_states(network)
        rollout = lif_network.create_rollout(network, benchmark_config)
        network.set_condition(3.0, 2.0, benchmark_config)
        network.reset()
        started = time.perf_counter()
        cold = rollout()
        jax.block_until_ready(cold[2])
        cold_seconds = time.perf_counter() - started
        network.reset()
        started = time.perf_counter()
        warm = rollout()
        jax.block_until_ready(warm[2])
        warm_seconds = time.perf_counter() - started
    result = {
        "network": {"NE": 10000, "NI": 2500, "CE": 1000, "CI": 250},
        "benchmark_duration_ms": 20.0,
        "benchmark_steps": 200,
        "connectivity_construction_seconds": construction_seconds,
        "cold_compile_and_run_seconds": cold_seconds,
        "warm_run_seconds": warm_seconds,
        "projected_warm_seconds_per_5s_run": warm_seconds * 250.0,
        "projected_warm_seconds_for_12_runs": warm_seconds * 3000.0,
        "backend": jax.default_backend(),
    }
    path = Path(__file__).with_name("full_scale_benchmark-iteration2.json")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
