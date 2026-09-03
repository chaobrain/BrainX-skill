from __future__ import annotations

import json
from pathlib import Path

import brainstate
import numpy as np

import lif_network


ROOT = Path(__file__).resolve().parent


def main() -> None:
    old_path = ROOT / "runs/smoke-step2/raw/seed-11_sr.npz"
    new_path = ROOT / "runs/smoke-iteration2-delay/raw/seed-11_sr.npz"
    common_keys = (
        "exc_rate_hz",
        "inh_rate_hz",
        "global_rate_hz",
        "sample_spikes",
        "sample_ids",
        "isi_cv_valid",
        "frequencies_hz",
        "power_hz",
    )
    with np.load(old_path) as old, np.load(new_path) as new:
        parity = {key: bool(np.array_equal(old[key], new[key])) for key in common_keys}
        old_cv = old["isi_cv"]
        new_cv = new["isi_cv"]
        parity["isi_cv"] = bool(
            np.array_equal(np.isnan(old_cv), np.isnan(new_cv))
            and np.array_equal(np.nan_to_num(old_cv), np.nan_to_num(new_cv))
        )
        final_voltage = new["final_voltage_mV"]
        final_voltage_evidence = {
            "shape": list(final_voltage.shape),
            "all_finite": bool(np.all(np.isfinite(final_voltage))),
            "sha256": lif_network.array_sha256(final_voltage),
        }
    if not all(parity.values()):
        raise RuntimeError(f"native delay migration changed smoke output: {parity}")

    poisson = []
    sample_size = 250_000
    for offset, expected in enumerate((0.9, 2.0, 4.0)):
        rng = brainstate.random.RandomState(7100 + offset)
        values = np.asarray(rng.poisson(lam=expected, size=(sample_size,)))
        poisson.append(
            {
                "seed": 7100 + offset,
                "lambda": expected,
                "sample_size": sample_size,
                "sample_mean": float(np.mean(values)),
                "sample_variance": float(np.var(values)),
                "sample_sha256": lif_network.array_sha256(values),
            }
        )
    evidence = {
        "delay_migration": {
            "iteration1_raw": str(old_path.relative_to(ROOT)),
            "iteration2_raw": str(new_path.relative_to(ROOT)),
            "exact_array_parity": parity,
        },
        "iteration2_smoke_final_voltage": final_voltage_evidence,
        "poisson_fixed_snapshots": poisson,
    }
    path = ROOT / "iteration2_control_evidence.json"
    path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
