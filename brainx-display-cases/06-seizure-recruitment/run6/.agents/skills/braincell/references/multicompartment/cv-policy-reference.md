# CV policy reference

Use this reference after opening `references/multicompartment/multicompartment-cell-workflow.md` when control-volume resolution must be chosen, varied by branch type, inspected, or checked for numerical convergence.

## Discretization model

A `CVPolicy` converts each continuous morphology branch into normalized `(prox, dist)` intervals that become isopotential control volumes; more and shorter CVs resolve spatial voltage gradients with more State and computation.

`Cell` treats the policy output as the source of truth for compartment splitting. Morphology sample points and branches do not determine the final compartment count.

## Choose a policy

| API | Use when | Important behavior and result |
|---|---|---|
| `Cell(morphology, cv_policy=None)` | Use the package default for a structural baseline. | `None` selects `CVPerBranch()` with one CV per branch. |
| `CVPerBranch(cv_per_branch=1)` | A fixed, predictable count per branch is required. | It splits every branch uniformly in normalized coordinates and returns the same count for short and long branches. |
| `MaxCVLen(max_cv_len, keep_odd=True)` | Physical branch length should cap CV size. | It computes a count independently for each branch, then splits that branch uniformly so CV lengths do not exceed the target up to tolerance. |
| `DLambda(d_lambda, frequency=..., keep_odd=True)` | Detailed cable modeling should follow electrical rather than only geometric length. | It computes electrotonic length from diameter, axial resistivity, membrane capacitance, and frequency, then sizes CVs as a fraction `d_lambda` of the AC length constant. |
| `CVPolicyByTypeRule(branch_types, policy)` | One or more morphology branch types need a distinct policy. | It pairs a tuple of type names with the policy applied to matching branches. |
| `CompositeByTypePolicy(rules, default_policy)` | Soma, dendrite, axon, or custom branches need different rules. | It checks rules in order and applies `default_policy` when no rule matches. |
| `policy.resolve_cv_bounds(morphology, paint_rules=...)` | Implementing or testing a policy directly. | It returns branch-wise tuples of normalized `(prox, dist)` intervals; normal modeling should let `Cell` call it. |

Use `CVPerBranch` for a reproducible baseline or deliberately uniform branch resolution. Use `MaxCVLen` when the modeling assumption is a physical maximum compartment length. Use `DLambda` as the detailed cable-model starting point when electrical length should control refinement.

## Inspect a policy before simulation

Construct cells with candidate policies and inspect the actual CV records; do not infer the result from morphology point count.

```python
import braincell
import brainunit as u


morphology = braincell.Morphology.from_swc("neuron.swc")

baseline = braincell.Cell(
    morphology,
    cv_policy=braincell.CVPerBranch(cv_per_branch=1),
)
length_limited = braincell.Cell(
    morphology,
    cv_policy=braincell.MaxCVLen(max_cv_len=20.0 * u.um),
)
electrical = braincell.Cell(
    morphology,
    cv_policy=braincell.DLambda(
        d_lambda=0.1,
        frequency=100.0 * u.Hz,
    ),
)

for name, cell in (
    ("baseline", baseline),
    ("length_limited", length_limited),
    ("electrical", electrical),
):
    assert cell.n_cv == len(cell.cvs)
    print(name, cell.n_cv)
    print([(cv.branch_id, cv.prox, cv.dist) for cv in cell.cvs[:4]])
```

Inspect `cv.branch_id`, `cv.branch_type`, `cv.prox`, `cv.dist`, `cv.length`, and `cv.area` when count alone cannot explain the spatial layout.

## Apply policies by branch type

Compose type rules when one global resolution rule would over-resolve some cables or under-resolve others.

```python
import braincell
import brainunit as u


policy = braincell.CompositeByTypePolicy(
    rules=(
        braincell.CVPolicyByTypeRule(
            branch_types=("soma",),
            policy=braincell.CVPerBranch(cv_per_branch=3),
        ),
        braincell.CVPolicyByTypeRule(
            branch_types=(
                "dendrite",
                "basal_dendrite",
                "apical_dendrite",
            ),
            policy=braincell.DLambda(
                d_lambda=0.1,
                frequency=100.0 * u.Hz,
            ),
        ),
    ),
    default_policy=braincell.MaxCVLen(max_cv_len=40.0 * u.um),
)

cell = braincell.Cell(morphology, cv_policy=policy)
print("total CVs:", cell.n_cv)
```

Use the exact type strings present in the loaded morphology. Inspect branch types after import before relying on a by-type rule; unmatched or misspelled types silently take the default policy.

## Respect `DLambda` cable-property constraints

`DLambda` requires uniform axial resistivity (`ra`) and membrane capacitance (`cm`) within each branch because both enter the AC length constant. Painting a different `CableProperty` onto only a sub-interval of one branch raises `ValueError`.

Resting potential and temperature may vary within a branch without violating this specific `DLambda` constraint.

If sub-branch `ra` or `cm` variation is scientifically required, choose a compatible policy such as `MaxCVLen`, or split the geometry into branches whose cable properties are internally uniform.

Always pass `d_lambda` explicitly. Also pass `frequency` explicitly when reproducibility matters; do not copy the concept-page shorthand that omits the required `d_lambda` argument.

## Check spatial convergence

Treat CV policy as a numerical assumption, not a visualization preference.

1. Choose a policy from the spatial question and inspect `cell.n_cv` and `cell.cvs`.
2. Run the scientific protocol with fixed morphology, mechanisms, stimulus, solver, and `dt`.
3. Refine only the CV policy: increase `cv_per_branch`, reduce `max_cv_len`, or reduce `d_lambda`.
4. Compare the relevant traces, spike timing, peaks, attenuation, or other spatial observable at the same probe locations.
5. Accept the coarser policy only when the observable remains within the task's stated tolerance.

Do not change `dt`, solver, and CV policy in the same convergence comparison. A stable CV count does not establish time-step or solver convergence.

Open `references/multicompartment/topology-building-and-visualization.md` when CV coverage, locset ownership, or the runtime node mapping must be inspected. Return to `references/multicompartment/multicompartment-cell-workflow.md` for mechanism declaration and simulation.

## Sources

- [Discretization](https://brainx.chaobrain.com/braincell/concepts/discretization.html)
- [Cell in BrainCell](https://brainx.chaobrain.com/braincell/tutorials/cell.html)
- [CVPolicy API](https://brainx.chaobrain.com/braincell/apis/generated/braincell.CVPolicy.html)
- [CVPerBranch API](https://brainx.chaobrain.com/braincell/apis/generated/braincell.CVPerBranch.html)
- [MaxCVLen API](https://brainx.chaobrain.com/braincell/apis/generated/braincell.MaxCVLen.html)
- [DLambda API](https://brainx.chaobrain.com/braincell/apis/generated/braincell.DLambda.html)
- [CVPolicyByTypeRule API](https://brainx.chaobrain.com/braincell/apis/generated/braincell.CVPolicyByTypeRule.html)
- [CompositeByTypePolicy API](https://brainx.chaobrain.com/braincell/apis/generated/braincell.CompositeByTypePolicy.html)
