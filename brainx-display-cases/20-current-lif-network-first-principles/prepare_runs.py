from __future__ import annotations

import difflib
import hashlib
import json
import subprocess
from pathlib import Path

import lif_network


ROOT = Path(__file__).resolve().parent
PYTHON = Path("/home/yixinliu/anaconda3/envs/braincell-released/bin/python")
SOURCE_FILES = (
    "NeuroSpecification.md",
    "BrainXStudy.md",
    "config.json",
    "lif_network.py",
    "test_lif_network.py",
    "acceleration_audit.md",
    "acceleration_parity.json",
    "connectivity_benchmark.json",
    "full_scale_benchmark.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def code_diff(source_files: tuple[str, ...] = SOURCE_FILES) -> str:
    chunks = []
    for relative in source_files:
        lines = (ROOT / relative).read_text(encoding="utf-8").splitlines(keepends=True)
        chunks.extend(
            difflib.unified_diff(
                [], lines, fromfile="/dev/null", tofile=relative, lineterm=""
            )
        )
    return "\n".join(chunks) + "\n"


def prepare(
    run_id: str,
    *,
    mode: str,
    seed: int | None,
    cores: str | None,
    iteration: int = 1,
    estimated_seconds_per_condition: float = 1178.6,
    source_files: tuple[str, ...] = SOURCE_FILES,
) -> None:
    run_dir = ROOT / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to replace existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    base_config = lif_network.load_config()
    active_config = lif_network.resolve_run_config(
        base_config, smoke=mode == "smoke", selected_seed=seed
    )
    environment = lif_network.environment_record()
    command_parts = []
    if cores is not None:
        command_parts.extend(["taskset", "-c", cores])
    command_parts.extend(
        [
            "env",
            "JAX_PLATFORMS=cpu",
            "MPLCONFIGDIR=/tmp/matplotlib-brainx-lif",
            "PYTHONUNBUFFERED=1",
            str(PYTHON),
            str(ROOT / "lif_network.py"),
            "--mode",
            mode,
            "--output-dir",
            str(run_dir),
        ]
    )
    if seed is not None:
        command_parts.extend(["--seed", str(seed)])
    command = " ".join(command_parts)
    source_hashes = {relative: sha256(ROOT / relative) for relative in source_files}
    config_text = json.dumps(active_config, indent=2) + "\n"
    environment_text = json.dumps(environment, indent=2) + "\n"
    (run_dir / "config.json").write_text(config_text, encoding="utf-8")
    (run_dir / "environment.json").write_text(environment_text, encoding="utf-8")
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    snapshot = code_diff(source_files)
    (run_dir / "code.diff").write_text(snapshot, encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "prepared",
                "pid": None,
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    expected = [
        "config.json",
        "environment.json",
        "command.txt",
        "code.diff",
        "run.log",
        "status.json",
        "exit_code",
        "connectivity_manifest.json",
        "run_metrics.csv",
        "condition_assessment.json",
    ]
    run_spec = f"""# Run specification

- Run ID: `{run_id}`
- Modeling-loop iteration: {iteration}
- Entry case: new
- Run level: {mode}
- Working directory: `{ROOT}`
- Interpreter: `{PYTHON}`
- Backend and device: CPU; `{environment['jax_devices']}`
- BrainState environment: `dt = {active_config['model']['dt_ms']} ms`, `fit = false`, precision contract 32-bit
- Seeds: `{active_config['protocol']['seeds']}`
- Seed policy: deterministic derived connectivity, initialization/sample, and external streams; exact replay expected
- Connectivity: exact fixed fan-in, no autapses, shared across conditions within each seed
- Checkpoint source: none
- Retry budget: zero unchanged retries; deterministic failure returns to implementation
- Expected artifacts: `{expected}` plus one connectivity NPZ and per-condition raw/metric files
- Resource estimate: full-scale warm estimate {estimated_seconds_per_condition:.1f} s per condition; production seeds run on disjoint CPU core sets; approximately 100-200 MB static arrays per process plus compiled runtime State
- Stop conditions: nonzero process exit, non-finite state/metrics, invalid connectivity, failed frozen-artifact verification, parse failure, disk exhaustion, or memory exhaustion
- Scientific acceptance: not decided by process completion; requires step-5 review

## Identity

- Git commit: `{git_value('rev-parse', 'HEAD')}`
- Git status at freeze: `{git_value('status', '--short')}`
- Locked specification SHA-256: `{sha256(ROOT / 'NeuroSpecification.md')}`
- Active config SHA-256: `{hashlib.sha256(config_text.encode('utf-8')).hexdigest()}`
- Code snapshot SHA-256: `{hashlib.sha256(snapshot.encode('utf-8')).hexdigest()}`
- Source hashes: `{json.dumps(source_hashes, sort_keys=True)}`

## Command

```text
{command}
```
"""
    (run_dir / "RUN_SPEC.md").write_text(run_spec, encoding="utf-8")


def main() -> None:
    prepare("smoke-cpu-20260903", mode="smoke", seed=None, cores=None)
    prepare("production-seed-11-cpu-20260903", mode="production", seed=11, cores="0-6")
    prepare("production-seed-29-cpu-20260903", mode="production", seed=29, cores="7-13")
    prepare("production-seed-47-cpu-20260903", mode="production", seed=47, cores="14-20")


if __name__ == "__main__":
    main()
