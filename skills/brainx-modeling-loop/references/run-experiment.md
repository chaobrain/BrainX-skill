# BrainX experiment runner

Sources:

- [BrainX installing the ecosystem](https://brainx.chaobrain.com/summ/install.html)
- [BrainX change log](https://brainx.chaobrain.com/summ/CHANGELOG.html)
- [BrainState simulation environment](https://brainx.chaobrain.com/brainstate/concepts/time_and_environment.html)
- [JAX GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)

## Purpose and boundary

Use this reference to turn a ready BrainX entry point into an inspectable launch: preflight the exact environment and compute target, verify any resume checkpoint, freeze provenance, launch one declared run level, then hand the recorded run to `monitor-experiment.md`.

Keep scientific design and iteration in `brainx-modeling-loop`. Open `brainx-install` when the selected environment is absent, incompatible, or missing the required CPU/CUDA extra; do not repair an environment inside this workflow. Open `brainx-acceleration` before production when performance, memory, batching, or multi-device work still needs design changes.

This reference may execute a run the user requested. It does not authorize provisioning paid compute, changing a scientific configuration, installing packages, destroying a remote instance, or accepting scientific claims.

## Underlying principle of BrainX experiment execution

A run snapshot represents one immutable scientific execution contract. It binds the specification, code, data, BrainX release tuple, BrainState environment, device, precision, seeds, command, and expected artifacts to one run ID.

A compute target represents the backend on which JAX and BrainX execute. Process-level JAX selection and the model's BrainState `platform` setting must agree before any BrainX or JAX import.

A run level represents execution intent. `smoke` proves mechanics at reduced scale, `production` executes the locked experiment, and `replication` repeats valid production conditions across declared seeds, subjects, protocols, or controls.

A run status represents process state, not scientific truth. `done` means the command completed and required artifacts passed mechanical checks; it never means the result is accepted.

## Execution surface overview

| Surface | Owns |
|---|---|
| `brainx-install` | Coherent BrainX release tuple, Python environment, and CPU/CUDA/TPU extra |
| `jax.devices()` and `jax.default_backend()` | Runtime-visible devices and selected JAX backend |
| `brainstate.environ` | Model run settings such as `dt`, `fit`, precision, and BrainState platform |
| `RUN_SPEC.md` and `config.json` | Immutable human-readable and machine-readable run contracts |
| Process or scheduler handle | Liveness, exit state, and cancellation |
| Logs, checkpoints, raw outputs, and metrics | Inspectable execution evidence |

## Start or resume from an explicit contract

Treat a new launch and a resumed checkpoint as separate paths that share the same preflight and provenance gates.

| API | Description |
|---|---|
| Project instructions and locked specification | Use first to recover the intended environment, entry point, run level, scientific configuration, seeds, inputs, outputs, resource estimate, and stop conditions; missing scientific choices return to the modeling workflow. |
| `python -m pip list --format=json` | Use with the exact target interpreter to record installed distributions without importing the stack; it returns package/version evidence for the snapshot. |
| `import BrainX, jax` | Use only after backend environment variables are set; it verifies imports in the selected process. |
| `jax.devices()` | Use in preflight to list devices visible to that process; it returns runtime device objects but does not prove the experiment fits or authorize a target. |
| `jax.default_backend()` | Use to compare the active JAX backend with the declared compute target; a mismatch fails preflight. |

Classify the request before launch:

| Entry case | Use when | Required next action |
|---|---|---|
| `new` | No prior run is being continued | Create a new run ID and snapshot. |
| `resume` | A compatible checkpoint must continue | Apply the checkpoint gate below, then create a new run ID linked to the source run. |
| `monitor` | A run handle or run directory already exists | Open `monitor-experiment.md`; do not relaunch. |

```bash
RUN_PYTHON=/absolute/path/to/python
JAX_PLATFORMS=cpu "$RUN_PYTHON" -c \
  "import BrainX, jax; print('BrainX OK'); print('backend:', jax.default_backend()); print('devices:', jax.devices())"
```

Replace the CPU selector with the confirmed GPU selector only after choosing the device path below.

## Select CPU or GPU before import

Choose the backend from the workload and locked experiment contract, not from device visibility alone.

| API | Description |
|---|---|
| `JAX_PLATFORMS=cpu` | Use for CPU execution; set it before the first BrainX/JAX import and require `jax.default_backend() == "cpu"`. |
| `CUDA_VISIBLE_DEVICES=<physical-id>` | Use before import to isolate one selected NVIDIA GPU; record the physical ID and the JAX-visible logical device because CUDA remaps it inside the process. |
| `JAX_PLATFORMS=cuda` | Use for NVIDIA GPU execution; fail preflight if CUDA initialization fails or JAX falls back to CPU. |
| `nvidia-smi` | Use as host-side evidence for GPU identity, free memory, utilization, driver, and active processes; it does not replace the JAX device check. |
| `brainstate.environ.set_platform("cpu" | "gpu")` | Use in the model's process-level setup when the entry point owns BrainState platform selection; it must agree with the JAX backend and must not be changed through a scoped `context()`. |

Use CPU for small smoke tests, debugging, modest models, or runs whose warm benchmark does not justify accelerator dispatch. Use GPU for compatible, sufficiently array-oriented workloads whose memory estimate fits the selected device and whose production contract declares CUDA. A CPU smoke run does not satisfy a GPU smoke gate; run a target-device smoke before GPU production.

```bash
RUN_PYTHON=/absolute/path/to/python
CUDA_VISIBLE_DEVICES=2 JAX_PLATFORMS=cuda "$RUN_PYTHON" -c \
  "import BrainX, jax; print('BrainX OK'); print('backend:', jax.default_backend()); print('devices:', jax.devices())"
```

JAX normally preallocates 75% of GPU memory. Keep that default when one process owns the GPU. Use `XLA_PYTHON_CLIENT_MEM_FRACTION=<fraction>` only for an explicit multi-process memory plan, or `XLA_PYTHON_CLIENT_PREALLOCATE=false` for debugging/coexistence while accepting greater fragmentation risk. Record every override in the run snapshot.

Launch independent runs on separate GPUs only when their State, RNG, data writes, checkpoints, and output directories are independent. Do not assume multiple visible GPUs accelerate one run; route `pmap2`, `shard_map`, and model/data sharding to `brainx-acceleration`.

For an existing remote environment, use the project's declared host, interpreter, code directory, sync method, scheduler/process manager, storage, and cost rules. Verify remote hashes and BrainX/JAX devices before launch. Do not invent a host, provision paid compute, install packages, or destroy an instance without explicit authorization.

## Verify checkpoint compatibility

A checkpoint is compatible only when its scientific contract and complete continuation State match the proposed run.

| Check | Pass condition |
|---|---|
| Identity | Source run ID, checkpoint hash, completion marker, format, and recorded step/epoch/time are present. |
| Scientific contract | Specification, config, data/preprocessing, protocol, solver, `dt`, precision, and seed policy match. |
| Environment | BrainX release tuple, JAX/JAXlib, backend, device count, and numerical settings match, or a tested migration is declared. |
| State graph | Parameter, dynamical, optimizer, RNG, online-learning, and progress State have matching names, roles, PyTree structure, shapes, dtypes, units, and transforms. |
| Witness | A read-only restore plus one representative target-backend step preserves expected shapes, units, finite values, and progress. |

Classify the result as `exact-resume`, `compatible-migration`, `weights-only-start`, or `incompatible`. A weights-only load is a new scientific run, not continuation. Never overwrite or convert the source checkpoint in place; save migrated output under the new run and preserve both hashes.

## Freeze and launch one run

Freeze the run before execution; append logs and status afterward without rewriting the contract.

| API | Description |
|---|---|
| `RUN_SPEC.md` | Use for the readable run contract; write the run identity, hashes, backend, precision, seed policy, command, expected artifacts, estimates, and stop conditions before launch, then never edit it. |
| `config.json` | Use as the exact machine-readable scientific configuration consumed by the entry point; any scientific change creates a new run ID. |
| `run.log` | Use as the append-only combined process log when separate streams are unnecessary; preserve the entry-point exit status when piping through `tee`. |
| Process manager or scheduler | Use the project's established local session, cluster scheduler, or remote process manager for long runs; record its handle and submitted command. |

Use this minimum run layout:

```text
runs/<run-id>/
|-- RUN_SPEC.md
|-- config.json
|-- environment.json
|-- command.txt
|-- code.diff
|-- status.json
|-- run.log
|-- exit_code
|-- checkpoints/
|-- raw/
`-- metrics/
```

Record the specification/config hashes; commit and dirty diff hash; data and preprocessing hashes; absolute interpreter and BrainX/JAX package tuple; physical and process-visible device identities; BrainState platform, precision, seeds, and determinism expectation; checkpoint source; exact command; expected outputs; resource estimate; timeout, retry budget, and stop conditions. Never store credentials or secret values.

Keep `RUN_SPEC.md`, `config.json`, `environment.json`, `command.txt`, and `code.diff` immutable after launch. Keep `status.json` mutable and `run.log` append-only. Any scientific, backend, precision, or seed-policy change creates a new run ID.

Follow this fixed order:

1. Confirm `smoke`, `production`, or `replication` and the `new` or `resume` entry case.
2. Read project instructions and identify the exact interpreter, environment, entry point, data, output root, backend, precision, seeds, resource budget, expected artifacts, and kill conditions.
3. Verify the BrainX meta-package environment without changing it. Route any incompatibility to `brainx-install`.
4. Preflight CPU or GPU resources and verify the selected backend in a fresh process.
5. Create a smoke run ID and immutable snapshot, run it on the declared target, then verify construction, State initialization/reset, output paths, finite representative outputs, and a checkpoint write when checkpoints are required.
6. Create a separate immutable production or replication snapshot only after the applicable readiness and acceleration gates pass.
7. Launch the exact recorded command in the exact recorded working directory and preserve its process or scheduler handle.
8. Verify liveness, log creation, target-device use, and expected early artifacts before reporting the launch.

```bash
set -o pipefail
JAX_PLATFORMS=cpu "$RUN_PYTHON" experiment.py --config "$RUN_DIR/config.json" \
  2>&1 | tee -a "$RUN_DIR/run.log"
RUN_EXIT=${PIPESTATUS[0]}
printf '%s\n' "$RUN_EXIT" > "$RUN_DIR/exit_code"
exit "$RUN_EXIT"
```

Use the GPU environment from the preceding section for a CUDA run. Do not rewrite `RUN_SPEC.md` or `config.json` after this command starts.

## Hand off to monitoring

After launch, open `monitor-experiment.md` for liveness, progress, resource health, stopping, retry, collection, and reporting. Pass the run directory and recorded process/scheduler handle. Process completion remains mechanical evidence and never implies scientific acceptance.

Open `skills/package-skills/brainstate/references/simulation-environment.md` only when exact `dt`, `fit`, precision, platform, or isolated-environment behavior needs verification. Open `brainx-install` when the environment is incompatible, and `brainx-acceleration` when OOM, precision, sharding, batching, or performance changes would alter execution semantics.

## Boundaries and common failures

- Do not design, tune, or revise the scientific experiment from observed results.
- Do not treat a visible GPU, an idle GPU, or a faster device as permission to select it.
- Do not install individual BrainX/JAX components on the run host; preserve the coherent BrainX environment or route to `brainx-install`.
- Do not use PyTorch CUDA/MPS checks for BrainX. Verify through JAX; Apple MPS is not an official BrainX install target.
- Do not silently accept CPU fallback from a failed CUDA initialization.
- Do not run multiple JAX processes on one GPU without an explicit memory-allocation plan.
- Do not assume multiple GPUs accelerate one run; use independent run processes only for independent runs, and route sharded model execution to `brainx-acceleration`.
- Do not overwrite configs, logs, checkpoints, or failed-run evidence during retry.
- Do not copy only `*.py` files to a remote host when configs, package metadata, modules, or data manifests are required by the frozen snapshot.
- Do not make W&B, SSH, `screen`, `tmux`, a cloud provider, or one operating system mandatory.
- Do not equate process completion, a decreasing loss, or a best seed with scientific acceptance.
