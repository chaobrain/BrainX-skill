from __future__ import annotations

import json
import time
from pathlib import Path

import brainevent
import jax
import jax.numpy as jnp
import numpy as np

import lif_network


def to_csr(indices_by_post: np.ndarray, n_pre: int):
    n_post, degree = indices_by_post.shape
    sources = indices_by_post.reshape(-1)
    targets = np.repeat(np.arange(n_post, dtype=np.int32), degree)
    order = np.argsort(sources, kind="stable")
    sorted_sources = sources[order]
    sorted_targets = targets[order]
    counts = np.bincount(sorted_sources, minlength=n_pre)
    indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    return brainevent.CSR(
        (
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray(sorted_targets),
            jnp.asarray(indptr),
        ),
        shape=(n_pre, n_post),
    )


def timed(function, spikes, repeats: int) -> float:
    function(spikes).block_until_ready()
    started = time.perf_counter()
    for _ in range(repeats):
        function(spikes).block_until_ready()
    return (time.perf_counter() - started) / repeats


def main() -> None:
    ne, ni, ce, ci = 800, 200, 80, 20
    structures = lif_network.build_random_structures(
        11,
        ne=ne,
        ni=ni,
        ce=ce,
        ci=ci,
        sample_size=20,
        reset_mV=10.0,
        threshold_mV=20.0,
    )
    indices = structures["exc_indices"]
    fixed = brainevent.FixedNumPerPost(
        (jnp.asarray(1.0, dtype=jnp.float32), jnp.asarray(indices)),
        shape=(ne, ne + ni),
    )
    csr = to_csr(indices, ne)
    fixed_product = jax.jit(lambda x: brainevent.BinaryArray(x) @ fixed)
    csr_product = jax.jit(lambda x: brainevent.BinaryArray(x) @ csr)
    rng = np.random.default_rng(123)
    rows = []
    for spike_probability in (0.01, 0.05, 0.2, 0.8):
        spikes = jnp.asarray(rng.random(ne) < spike_probability)
        fixed_result = np.asarray(fixed_product(spikes))
        csr_result = np.asarray(csr_product(spikes))
        exact = bool(np.array_equal(fixed_result, csr_result))
        if not exact:
            raise RuntimeError("CSR changes fixed-fan-in communication")
        fixed_seconds = timed(fixed_product, spikes, 30)
        csr_seconds = timed(csr_product, spikes, 30)
        rows.append(
            {
                "spike_probability": spike_probability,
                "active_spikes": int(np.sum(spikes)),
                "exact_count_parity": exact,
                "fixed_num_per_post_seconds": fixed_seconds,
                "csr_seconds": csr_seconds,
                "csr_speedup": fixed_seconds / csr_seconds,
            }
        )
    output = {"shape": [ne, ne + ni], "degree": ce, "measurements": rows}
    path = Path(__file__).with_name("connectivity_benchmark.json")
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
