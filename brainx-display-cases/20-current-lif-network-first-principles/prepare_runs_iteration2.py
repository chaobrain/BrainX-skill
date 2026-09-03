from __future__ import annotations

import prepare_runs


SOURCE_FILES = (
    "NeuroSpecification.md",
    "BrainXStudy.md",
    "BrainXStudy-iteration2.md",
    "BrainToolsAPIGap.md",
    "config.json",
    "lif_network.py",
    "test_lif_network.py",
    "iteration2_control_evidence.json",
    "acceleration_audit-iteration2.md",
    "acceleration_parity-iteration2.json",
    "connectivity_benchmark.json",
    "full_scale_benchmark-iteration2.json",
    "reviews/iteration-1.md",
)


def prepare(run_id: str, *, mode: str, seed: int | None, cores: str | None):
    prepare_runs.prepare(
        run_id,
        mode=mode,
        seed=seed,
        cores=cores,
        iteration=2,
        estimated_seconds_per_condition=1415.2,
        source_files=SOURCE_FILES,
    )


def main() -> None:
    prepare("smoke-v2-cpu-20260903", mode="smoke", seed=None, cores=None)
    prepare(
        "production-v2-seed-11-cpu-20260903",
        mode="production",
        seed=11,
        cores="0-6",
    )
    prepare(
        "production-v2-seed-29-cpu-20260903",
        mode="production",
        seed=29,
        cores="7-13",
    )
    prepare(
        "production-v2-seed-47-cpu-20260903",
        mode="production",
        seed=47,
        cores="14-20",
    )


if __name__ == "__main__":
    main()
