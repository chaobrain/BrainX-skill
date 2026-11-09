---
name: brainx-acceleration
description: Audits and refactors BrainX/BrainState simulation code for performance using state-aware BrainState transform patterns. Use for speed, acceleration, GPU performance, batching, vectorization, parameter sweeps, many neurons/trials, compile/runtime separation, memory reduction, throughput, multi-device execution, or performance-audit requests.
---

# BrainX acceleration

## Purpose and boundary

Audit or refactor BrainX simulation and training hot paths. Open `brainstate` → `SKILL.md` before a nontrivial State-aware rewrite. Keep the modeling package's orchestrator when it already expresses the scientific operation and the needed execution boundary.

## Underlying mental model

Represent populations and independent trials as array or State axes. Let BrainState control repeated execution so a compiled function can preserve mutation, randomness, and gradients. Prioritize correctness, State and RNG safety, shape stability, warm runtime, memory, then device scale.

## Diagnose and rewrite

1. Inventory the hot call sites. Record loop axes and trip counts, inputs and outputs, State reads and writes, RNG policy, transform boundaries, host transfers, and connectivity representation.
2. Rank by measured impact, semantic risk, and confidence. Identify a small deterministic baseline before changing execution. Open `references/audit-and-rewrite-patterns.md` for search patterns, bottleneck classification, transform compositions, and an audit report template.
3. Rewrite one axis or compilation boundary at a time. Keep units, update order, sparse or event connectivity, and stable shapes, dtypes, and PyTrees. Use BrainState transforms for State effects. Use raw JAX transforms for pure array functions with explicit inputs and outputs. Keep host preprocessing and I/O outside transforms.
4. Map only independent lanes. Declare which State is shared read-only, mapped, written back, or discarded. Split or advance random streams deliberately across lanes and time.
5. Compile the largest stable unit that owns the repeated work. Keep transform construction outside hot loops and benchmark already-fused or I/O-bound work before rewriting it.

| Bottleneck | Prefer |
|---|---|
| Python loop over neurons, features, or compartments | Population-shaped State and array or event operations. |
| Python time loop | `brainstate.transform.for_loop` for input-to-output steps, or `scan` when the body carries an explicit value. Keep only required history. |
| Independent trials or parameter sweeps | Native batching when provided by the model, or a State-aware mapped rollout with explicit State and RNG axes. |
| Parameter perturbation loops | State-aware `grad` for a differentiable objective. Keep per-example gradients for tasks that need them. |
| Long differentiable rollout | Checkpoint the rollout when activation storage dominates. |
| Multi-device workload | Establish an efficient single-device baseline before using `pmap2`, `shard_map`, or explicit sharding. |

## Validate and benchmark

Compare baseline and rewritten outputs, final State, units, and shapes on the same small input with stated tolerances. Check mapped State isolation, repeatable seeds, independent random draws, and gradient or optimizer updates where applicable. Confirm sparse connectivity and required outputs remain intact. Separate compile time from synchronized warm timing, and inspect memory and host transfers.

For an audit, report the bottleneck evidence, affected axis, impact, risk, proposed rewrite, and validation. For a requested rewrite, apply the smallest patch and report actual parity and timing. Keep a measured baseline when the rewrite would change scientific behavior or increase resource use without a measured benefit.

## Reference routing

| Reference | Open when |
|---|---|
| `references/audit-and-rewrite-patterns.md` | Classifying a suspected hot path, selecting among axis compositions, or drafting an acceleration audit. |
| `brainstate` → `references/brainstate/transformation-jit-expansion.md` | Choosing compilation boundaries, static arguments, timing, or diagnosing recompiles. |
| `brainstate` → `references/brainstate/transformation-vmap-expansion.md` | Mapping independent trial, ensemble, State, or RNG axes. |
| `brainstate` → `references/brainstate/transformation-grad-expansion.md` | Differentiating parameters or explicit inputs, or replacing finite differences. |
| `brainstate` → `references/brainstate/brainstate-control-flow-patterns.md` | Selecting a transformed time loop, branch, or checkpointing pattern. |
| `brainstate` → `references/brainstate/randomness-and-reproducibility.md` | Controlling stream independence, restoration, or reproducible replay. |

## Boundaries and common failures

Resolve State and RNG ownership before changing transforms. Keep a known-correct non-transformed path for tiny workloads, irreducibly dynamic shapes, or code dominated by I/O and logging when a transform offers little benefit. Address representation and memory bandwidth before batching a workload that would densify or materialize an unnecessary full history.
