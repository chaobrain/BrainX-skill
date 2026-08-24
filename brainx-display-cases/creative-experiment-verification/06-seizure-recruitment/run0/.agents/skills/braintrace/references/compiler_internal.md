# Compiler internals

Use this reference when `learner.report`, `learner.graph`, and compiler diagnostics cannot explain a missing or misplaced relation in a custom layer, ETP primitive, or online-learning algorithm. Keep normal model compilation on `braintrace.compile()`.

## Escalation path

Inspect public compilation results before calling lower-level compiler functions.

| API | Description |
|---|---|
| `braintrace.compile(model, algorithm, *example_args, ...)` | Use for normal compilation. It initializes the learner, runs the compiler pipeline once, and attaches a `CompilationReport` at `learner.report`. |
| `learner.report.show(level)` | Use first for human-readable hidden-group, included-weight, excluded-weight, and diagnostic summaries. Use levels `0` through `2` for increasing detail. |
| `learner.report.counts` | Use for programmatic checks of `hidden_groups`, `etrace_weights`, `excluded_weights`, `warnings`, and `errors`. |
| `learner.report.graph` | Use to access the compiled `ETraceGraph` after the report identifies a structural problem. |
| `graph.diagnostics` | Use to filter individual `CompilationRecord` decisions by kind or severity. |
| `graph.explain()` | Use to print the graph diagnostics grouped by decision kind. |

```python
import braintrace

learner = braintrace.compile(model, braintrace.D_RTRL, example_input)
learner.report.show(1)

assert learner.report.counts["errors"] == 0
for record in learner.report.diagnostics:
    print(record.level.name, record.kind.name, record.message)

graph = learner.report.graph
graph.explain()
```

**Invariant:** An exclusion is not automatically a compiler failure. Read the record's `kind`, `message`, `weight_path`, `hidden_paths`, and `context` before changing the model.

## Four-stage compilation pipeline

The compiler traces a `brainstate.nn.Module` to Jaxpr, discovers recurrent state structure and ETP relations, then adds the perturbations required by online algorithms.

| Stage API | Result | Use it to inspect |
|---|---|---|
| `braintrace.extract_module_info(model, *example_args)` | `ModuleInfo` | Jaxpr equations, compiled model states, hidden and parameter classifications, and path-to-variable mappings. |
| `braintrace.find_hidden_groups_from_minfo(minfo)` | `(hidden_groups, hidden_path_to_group)` | Recurrent hidden states that update together and their transition Jaxprs. |
| `braintrace.find_hidden_param_op_relations_from_minfo(minfo, hidden_path_to_group)` | Sequence of `HiddenParamOpRelation` | Which trainable inputs pass through which ETP primitive and reach which hidden group. |
| `braintrace.add_hidden_perturbation_from_minfo(minfo)` | `HiddenPerturbation` | Perturbation variables used to compute hidden-to-hidden Jacobians. |
| `braintrace.compile_etrace_graph(model, *example_args)` | `ETraceGraph` | The complete compiled graph, including module info, hidden groups, relations, perturbations, and diagnostics. |

```python
minfo = braintrace.extract_module_info(model, example_input)
groups, path_to_group = braintrace.find_hidden_groups_from_minfo(minfo)
relations = braintrace.find_hidden_param_op_relations_from_minfo(
    minfo, path_to_group
)
perturbation = braintrace.add_hidden_perturbation_from_minfo(minfo)

graph = braintrace.compile_etrace_graph(model, example_input)
assert len(graph.hidden_groups) == len(groups)
assert len(graph.hidden_param_op_relations) == len(relations)
assert graph.hidden_perturb is not None
```

Use the staged calls only to isolate a compiler stage. Use `compile_etrace_graph()` when a custom algorithm needs the finished low-level graph.

## Hidden-group discovery

Hidden groups represent recurrent states connected through the traced computation and compatible enough to share a transition relation.

The compiler performs these decisions in order:

1. Trace forward from each hidden-state input variable to reachable hidden-state output variables.
2. Merge overlapping connected components.
3. Separate shape-incompatible states.
4. Separate hidden states belonging to different sequential layers.
5. Build each group's transition Jaxpr.

Inspect `HiddenGroup.index`, `num_state`, `varshape`, `hidden_paths`, `transition_jaxpr`, and `transition_jaxpr_constvars` when grouping differs from the model's intended recurrent structure.

**Invariant:** A hidden group is a compiler-discovered recurrent component, not merely every `HiddenState` with the same shape.

## ETP relation discovery

An ETP relation means that a registered primitive consumes trainable input and its output reaches a compatible recurrent hidden group without crossing an unsupported trainable tail.

| Decision | Compiler behavior |
|---|---|
| Primitive identification | Checks primitive object identity in `ETP_PRIMITIVES`; names and wrappers do not determine participation. |
| Trainable-input discovery | Calls the primitive's registered `trainable_invars_fn` through `get_trainable_invars(primitive, eqn.params)`. |
| Parameter ownership | Traces every trainable Jaxpr variable backward to its originating `ParamState`, including through parameter transforms. |
| Hidden reachability | Runs a forward breadth-first search from the primitive output to hidden-state outputs. |
| Shape validation | Keeps only hidden outputs broadcast-compatible with the primitive output. |
| Tail validation | Excludes a weight that reaches hidden state only through another genuinely trainable ETP primitive. |

Inspect a discovered relation through `primitive`, `trainable_paths`, `x_var`, `y_var`, `hidden_groups`, `connected_hidden_paths`, and `eqn_params`.

```python
from braintrace._op import is_etp_primitive

for index, equation in enumerate(minfo.jaxpr.eqns):
    if is_etp_primitive(equation.primitive):
        in_shapes = [
            var.aval.shape if hasattr(var, "aval") else "literal"
            for var in equation.invars
        ]
        out_shapes = [var.aval.shape for var in equation.outvars]
        print(index, equation.primitive.name, in_shapes, "->", out_shapes)
```

Use this Jaxpr scan to distinguish "the primitive was not traced" from "the primitive was traced but its relation was excluded."

## Graph execution

The graph executor runs the compiled forward graph and produces the hidden-to-weight and hidden-to-hidden Jacobians consumed by an online-learning algorithm.

| API | Description |
|---|---|
| `ETraceGraphExecutor` | Use as the base executor when developing a custom algorithm that consumes an `ETraceGraph`. It runs the forward graph and exposes its Jacobian calculations. |
| `ETraceVjpGraphExecutor` | Use for algorithms in the `ETraceVjpAlgorithm` family. It provides the VJP-based executor used by the built-in estimators. |

Keep executor calls inside the algorithm lifecycle. Application code should call the compiled learner, `etrace_evolve()`, or `etrace_grad()` rather than driving the executor directly.

## Control-flow policy

`ControlFlowPolicy` controls compiler canonicalization of JAX control-flow primitives, including conditional conversion, scan unrolling, and structured scan descent.

Use it only when recurrent State must cross compiler-supported control flow. Compile a minimal example and inspect the resulting hidden groups and relations; if hidden State inside `jax.lax.scan`, `jax.lax.while_loop`, or `jax.lax.cond` remains unsupported for the chosen structure, redesign the step so the recurrent update is directly visible to the compiler.

**Invariant:** A successful JAX trace does not prove that BrainTrace discovered the intended hidden-State relation through control flow.

## Diagnostics and failure classification

`CompilationRecord` makes every inclusion, exclusion, warning, and error inspectable instead of leaving compiler decisions silent.

| Field or type | Description |
|---|---|
| `CompilationRecord.kind` | Machine-readable `DiagnosticKind`, such as relation inclusion, non-temporal exclusion, weight-to-weight exclusion, bounded transition tail, or state mismatch. |
| `CompilationRecord.level` | `DiagnosticLevel.INFO`, `WARNING`, or `ERROR`. |
| `CompilationRecord.message` | Human-readable decision summary. |
| `CompilationRecord.primitive` | ETP primitive involved, when applicable. |
| `CompilationRecord.weight_path` | Path to the originating `ParamState`. |
| `CompilationRecord.hidden_paths` | Hidden-state paths reached by the candidate relation. |
| `CompilationRecord.context` | Group indices, path classifications, and other structured details. |

Classify common problems as follows:

| Symptom | Likely cause | Action |
|---|---|---|
| No ETP equation in the Jaxpr | The model used a regular JAX operation such as `x @ weight`. | Use the appropriate registered `braintrace` ETP operation when the parameter should participate online. |
| ETP equation exists but no relation | Its output does not reach a hidden state, fails shape compatibility, or crosses another trainable ETP primitive. | Read the corresponding `CompilationRecord`; do not force a valid non-temporal or weight-to-weight exclusion into the graph. |
| Trainable input has no owner | Backward tracing cannot reach a `ParamState`. | Preserve parameter ownership and ensure `trainable_invars_fn` identifies the correct input index. |
| Hidden grouping is unexpected | Connectivity, shape, or sequential-layer filtering differs from the intended architecture. | Inspect `hidden_path_to_invar`, the hidden-group paths, and each transition Jaxpr. |
| Hidden state appears inside JAX control flow | The structure is outside the normal direct Jaxpr path. | Consult `ControlFlowPolicy` for conditional conversion, scan unrolling, or structured scan descent; redesign unsupported stateful control flow when compilation still rejects it. |

## Low-level boundaries

- Compile with representative inputs whose batch and feature shapes match execution.
- Compile once per stable model structure; do not rebuild the graph inside the training loop.
- Declare every custom primitive's trainable-input layout instead of relying on the single-weight fallback.
- Keep `gradient_enabled=True` for identity-like tail primitives only; genuinely trainable weight operations form a tail boundary.
- Treat `braintrace._op` and `braintrace._compiler` helpers as diagnostic internals, not application-level APIs.

## Sources

- [Compiler Internals](https://brainx.chaobrain.com/braintrace/advanced/compiler_internals.html)
- [Compiler, Executor & Diagnostics](https://brainx.chaobrain.com/braintrace/apis/compiler.html)
