"""Measure cold/warm runtime and exact reset parity for the Brunel network."""

from __future__ import annotations

import json
import time

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import sparse_ei_network as model


def main():
    config = model.NetworkConfig(
        n_exc=800,
        n_inh=200,
        exc_indegree=80,
        inh_indegree=20,
        external_indegree=80,
        burn_ms=20.0,
        analysis_ms=80.0,
    )
    with brainstate.environ.context(dt=model.DT, precision=32):
        exc_conn, inh_conn, _, _ = model.make_connectivity(config, graph_seed=1729)
        net = model.SparseEINetwork(
            config,
            exc_conn=exc_conn,
            inh_conn=inh_conn,
            external_seed=2133,
        )
        brainstate.nn.init_all_states(net)
        runner = model.build_runner(net)

        model.reset_run(net, initial_seed=2032, external_seed=2133)
        started = time.perf_counter()
        cold = np.asarray(
            jax.block_until_ready(runner(jnp.asarray(5.0), jnp.asarray(2.0)))
        )
        cold_seconds = time.perf_counter() - started
        cold_voltage = np.asarray(net.neurons.V.value.to_decimal(u.mV))

        model.reset_run(net, initial_seed=2032, external_seed=2133)
        started = time.perf_counter()
        warm = np.asarray(
            jax.block_until_ready(runner(jnp.asarray(5.0), jnp.asarray(2.0)))
        )
        warm_seconds = time.perf_counter() - started
        warm_voltage = np.asarray(net.neurons.V.value.to_decimal(u.mV))

    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "cold_seconds": cold_seconds,
                "warm_seconds": warm_seconds,
                "output_shape": list(cold.shape),
                "spike_count": int(cold.sum()),
                "spikes_bit_identical": bool(np.array_equal(cold, warm)),
                "final_voltage_max_abs_difference_mv": float(
                    np.max(np.abs(cold_voltage - warm_voltage))
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
