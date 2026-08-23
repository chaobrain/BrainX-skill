# Monitor BrainX experiments

Use this reference after `run-experiment.md` launches a BrainX run. Monitor process and artifact health, stop only under declared rules, retry only transient failures, collect inspectable outputs, and return mechanical status to `brainx-modeling-loop`; do not tune the experiment or decide scientific acceptance.

## Inspect the recorded run

Start from `RUN_SPEC.md`, `status.json`, and the recorded process, session, scheduler, or provider handle. Verify that the handle still identifies the recorded command, working directory, and start time; never select a run by broad process-name matching.

| Signal | Check | Meaning |
|---|---|---|
| Liveness | Process/job/session/provider status | Whether execution is active; absence requires exit and artifact inspection. |
| Progress | New append-only logs or declared heartbeat | Whether the run advances at its expected cadence. |
| CPU/GPU resources | Memory, disk, utilization, device errors, and recorded PID/device | Resource health and target placement, not scientific quality. |
| Checkpoint | Existence, freshness, size, completion marker, and parseability | Recovery freshness; partial writes are invalid. |
| Output | Required files, sizes, parseability, and declared finite-value checks | Mechanical completeness or deterministic invalidity. |
| Existing telemetry | Run state and predeclared metrics | Secondary evidence; immutable local artifacts remain authoritative. |

Poll at a cadence proportional to the declared heartbeat/checkpoint interval. External scheduling may wake on machine-checkable conditions such as exit code, job state, file existence, or logged step; never schedule a scientific verdict.

## Classify before acting

| Status | Evidence | Action |
|---|---|---|
| `running-healthy` | Live handle, advancing progress, resources within bounds | Wait and report the latest raw progress. |
| `running-uncertain` | Live handle but no new signal within one expected interval | Inspect logs/resources; wait unless a declared limit is crossed. |
| `done` | Exit code zero and every required artifact exists and parses | Record completion and collect artifacts. |
| `failed-deterministic` | Reproducible code/config/data, NaN/Inf, corruption, leakage, or broken locked metric | Preserve evidence; do not retry unchanged. Return to implementation. |
| `failed-transient` | Preemption, transport interruption, temporary host/device, or recoverable scheduler failure | Retry only within budget under a new linked run ID. |
| `stopped` | Declared kill condition or authorized cancellation | Preserve partial artifacts and the reason. |

An unexpected finite result, noisy metric, weak interim effect, or insufficient samples is not deterministic invalidity. Do not stop because the hypothesis appears unsupported.

## Stop and retry safely

For a declared stop condition, record the evidence, request graceful termination, allow the checkpoint/cleanup grace period, verify exit and output finalization, and force termination only when the process remains active and the scope authorizes it. Preserve logs, partial outputs, checkpoints, exit/signal status, and device diagnostics.

Retry only a transient fault within the frozen retry budget. Create a new run ID with `parent_run_id` and `retry_of`; preserve the specification, code, config, data, precision, backend, and seed policy. Apply the checkpoint gate in `run-experiment.md` before resuming. A different batch size, precision, solver, `dt`, backend, seed rule, checkpoint cadence, or model code is a new scientific run, not a retry.

Never delete a failed run or overwrite collected artifacts. Never destroy paid remote compute until copied artifacts and hashes are verified and teardown is explicitly authorized.

## Collect and report

For remote runs, copy into a new local collection path and verify the artifact manifest and hashes before marking collection complete. Report raw values before interpretation:

| Run ID | Level | Backend/device | Progress or exit | Primary declared metric | Artifacts |
|---|---|---|---|---|---|
| `<id>` | `production` | `cuda / GPU UUID` | `done / 0` | `<raw value and unit>` | `complete` |

Also report start/end time, duration, exact command path, log path, checkpoint path/status, missing or corrupt outputs, retry lineage, and paid-compute cost when applicable. Compare only runs the locked contract declares comparable and label every differing field. Return `done`, `failed`, or `stopped`; never return `accepted`.
