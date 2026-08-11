# Reference Writing Style

Use this guide to turn mirrored source material into an operational reference that
helps an agent make decisions and execute correctly.

A reference should teach an agent:

- which API, format, or workflow to choose;
- how to use it correctly;
- which result, invariant, or failure condition matters.

Do not mirror the source mechanically. Inspect the mirrored sources, identify the
kind of reference they support, and reorganize the material around decisions and
execution.

## Structural Examples

Use these references as structural models:

| Reference | Type | Use it as a model when |
|---|---|---|
| `Array mechanics` (BrainUnit) | API-dominant | The material contains many parallel API families that need compact tables and occasional small workflows for non-obvious behavior. |
| `Extension mechanisms` (BrainState) | Workflow-dominant | A small number of APIs are best explained through executable patterns. |
| `Sparse formats` (BrainEvent) | Hybrid | The agent needs a compact API selection map followed by representative and end-to-end workflows. |

## 1. Select the Reference Type

Before writing, inspect the sources listed under `Sources mirrored` or `Sources`.
Determine which kind of material they primarily contain.

The source type is only an initial signal. Choose the final structure according
to the knowledge the agent needs, not the source documentation format alone.

### API-Dominant

Choose an API-dominant structure when the mirrored sources mainly contain:

- API reference pages;
- generated class or function documentation;
- long lists of parallel operations;
- signatures with similar semantics;
- method collections grouped by category.

Typical signals:

- Many APIs must be covered.
- Most APIs are used independently.
- The main difficulty is selecting the correct operation.
- Examples can be short and local.
- APIs divide into clear families.

Follow the structure of `Array mechanics`.

### Workflow-Dominant

Choose a workflow-dominant structure when the mirrored sources mainly contain:

- tutorials;
- how-to guides;
- extension guides;
- lifecycle explanations;
- examples showing several APIs working together.

Typical signals:

- There are relatively few important APIs.
- Execution order matters more than exhaustive coverage.
- Scripts carry the useful knowledge.
- Individual signatures are less important than composition.
- The source contains a strong canonical workflow.

Follow the structure of `Extension mechanisms`.

### Hybrid

Choose a hybrid structure when the mirrored sources contain both:

- several alternatives that require a selection rule; and
- one or more important end-to-end workflows.

Typical signals:

- The agent must first choose a format, orientation, backend, or mechanism.
- One representative API can establish the common path.
- A table is needed to compare variations.
- Scripts are needed to show conversion, construction, or execution order.

Follow the structure of `Sparse formats`.

## 2. Build a Decision-Oriented Structure

The reference should answer these questions in order:

1. What task does this reference cover?
2. What alternatives exist?
3. How should the agent choose among them?
4. What is the canonical usage pattern?

Do not preserve the source heading order unless it already matches this decision
flow.

### Open With a Precise Boundary

Begin with one sentence:

> Use this reference when ...

The opening should define the operational boundary by identifying:

- the task covered;
- the decisions supported;
- closely related concerns that are excluded.

Avoid broad package descriptions.

### Divide APIs Into Meaningful Families

When many APIs are present, group them by a real decision boundary. Do not group
APIs merely because they appear together in the source.

Give each API family its own heading, followed by a short selection rule:

```markdown
### Family name

Use these operations when ...

| API | Description |
|---|---|
| `Constructor(x, y, arg=...)` | One-line distinguishing behavior. |
| `operation(x, ...)` | One-line distinguishing behavior. |
```

The prose explains when to choose the family. The table explains which API
within the family to use. Put one callable or constructor in each row, with its
exact usable form in the first column.

Do not use a family as a table row and then place several APIs in another cell:

```markdown
| Family feature | APIs |
|---|---|
| General optimizers | `Adam(...)`, `AdamW(...)`, `SGD(...)` |
```

This layout makes individual APIs difficult to find and leaves no place for
their distinct selection rules.

### Add a Family Navigation Map Only When Needed

When a long reference contains multiple families, formats, or mechanisms, an
optional compact map may summarize the family-level choice before the detailed
sections.

| Family | Use when | Key constraint |
|---|---|---|
| [Family A](#family-a) | The input has ... | Preserves ... |
| [Family B](#family-b) | The workflow requires ... | Returns ... |

Keep this table compact. Its purpose is to route the agent to the correct
section, not to document APIs. Do not list callable names in this map. Omit the
map when the family headings and their selection sentences already make the
reference easy to scan.

## 3. Present APIs for Fast Lookup

### Choose the Smallest Useful Table

Do not force every API into the same table format. Choose the table based on the
decision it needs to support. In every API table, keep `API`, `Exact signature`,
`Constructor`, or another callable-form label as the first column.

#### APIs Requiring Individual Guidance

Use this format when syntax, arguments, return values, mutation behavior, units,
shapes, or other semantics require individual explanation:

| Exact signature | One-line description | Example and result |
|---|---|---|
| `reshape(x, shape)` | Changes the shape without changing the represented data. | `reshape(x, (2, 3))`<br>`# Shape: (2, 3)` |

Each row should provide:

- the exact callable form;
- the distinguishing behavior.

Add the `Example and result` column only when at least one local call makes a
non-obvious behavior, shape, type, unit, mutation, or return value clearer.
Otherwise, use the two-column form. When the column is present, place the result
on the next line as a code comment.

#### Directly Corresponding Forms

Use a comparison table when the relationship is more useful than separate
explanations:

| Functional form | In-place equivalent |
|---|---|
| `x.at[index].set(value)` | `x[index] = value` |

#### Simple API Families

Use a compact two-column table when the APIs only need lookup-level guidance:

```markdown
### API Category Name

Use these operations when ...

| API | Description |
|---|---|
| `api_name(...)` | One-line distinguishing behavior. |
```

Other table shapes are valid when they make the decision clearer. Keep the API
or callable form in the first column so the table remains scannable. Never use
a family name as a substitute for an API row.

### Keep APIs Vertically Scannable

When an agent may need to locate a specific API, list APIs vertically:

| API | Description |
|---|---|
| `reshape(...)` | Change shape without changing the represented data. |
| `transpose(...)` | Reorder axes. |
| `squeeze(...)` | Remove length-one axes. |

Use one API per row. Do not combine sibling APIs in one cell or create rows such
as `Adaptive optimizers` whose description contains `Adam(...)`, `AdamW(...)`,
and `Yogi(...)`. Do not bury distinct APIs inside prose. However, do not give
every API a full example merely because it exists.

Give every meaningful choice a visible place. Add full guidance only where
behavior or selection is non-obvious.

## 4. Teach Composition Through Workflows

Tables teach selection and lookup. Scripts teach composition.

A script should:

- demonstrate one canonical path;
- use realistic values;
- expose important intermediate objects;
- show or assert the result;
- omit unrelated setup.

Prefer one representative workflow over many nearly identical examples.

### Preserve Useful Source Workflows

When the mirrored source is tutorial-like, its scripts may contain more value
than its prose. In that case:

- keep the useful workflow;
- remove unrelated setup;
- make implicit decisions explicit;
- add expected results or assertions;
- correct naming and execution order;
- extract supporting APIs into a compact table where useful.

Do not replace a strong workflow with an artificial API catalog.

### Use At Most One Representative Path per Variation Family

For a family of similar APIs, add a representative implementation path only
when code is needed to teach composition or make a non-obvious result visible.
Document at most one path in full. Summarize the variations in a table and
explain only the differences that affect selection, behavior, or results.

This keeps code optional, prevents repetitive examples, and preserves searchable
API coverage.

## 5. Make Correctness Visible

### Place Invariants Beside the Relevant Workflow

State correctness rules immediately after the API or script they constrain.
Examples include:

- Preserve a returned permutation after index conversion.
- Distinguish wrapper-preserving from wrapper-dropping conversion.
- Functional updates return a new value.
- Index meaning depends on storage orientation.
- Traced values cannot be inspected with ordinary Python branching.
- Only certain hook phases may transform or cancel an operation.

Do not hide critical rules in a distant generic warning section. Use a final
gotchas section only for cross-cutting behavior.

### Make Results Explicit

An example should reveal what the operation changes:

```python
# Expected shape: (2, 3)
assert output.shape == (2, 3)

# The original value remains unchanged.
assert original == before_update

# Returns the structure, indices, and a value permutation.
structure, indices, value_permutation = convert(input_data)
```

Prioritize results that expose:

- shape;
- type;
- unit;
- orientation;
- ordering;
- mutation semantics;
- returned metadata.

Do not include examples that only prove the function can be called.

## 6. Use Progressive Disclosure

The main reference should contain:

- the task boundary;
- the conceptual or family map;
- selection rules;
- representative APIs;
- canonical workflows;
- critical invariants;
- links to exhaustive details.

Move repetitive or low-decision-value material into nested references.

Nested references may contain:

- every exact signature;
- official one-line descriptions;
- small examples and results;
- rare options;
- complete searchable coverage.

The main reference should remain operational, not exhaustive.

## Reference Templates

Use the template that matches the selected reference type. Add or remove
sections when the material requires it, but preserve the decision flow.

### API-Dominant Template

````markdown
# Reference Title

Use this reference when ...

## Family A

Use these APIs when ...

| API | Description |
|---|---|---|
| `Constructor(x, y, arg=...)` | One-line distinguishing behavior. |
| `operation(x, ...)` | One-line distinguishing behavior. |

```python
# Optional representative workflow when composition or results need demonstration.
```

**Invariant:** State the correctness rule beside the workflow it constrains.

## Family B

Use these APIs when ...

| API | Description | Example and result |
|---|---|---|
| `other_api(x, ...)` | One-line distinguishing behavior. | `other_api(x)`<br>`# Result: ...` |

Add an optional family navigation map before these sections only when the
reference is too long to scan directly. The map must link to family sections
without listing APIs in its cells.
````

### Workflow-Dominant Template

````markdown
# Reference Title

Use this reference when ...

## Workflow Overview

| Step | Action | Important result |
|---|---|---|

## Canonical Workflow

```python
# Executable end-to-end pattern with explicit intermediate results.
```

**Invariant:** State the correctness rule beside the relevant step.

## Supporting APIs

| API | description |
|---|---|

## Variations

Describe only variations that change selection, execution order, or results.

````

### Hybrid Template

````markdown
# Reference Title

Use this reference when ...

## Format or Mechanism A

Use this family when ...

| API | Description |
|---|---|---|
| `Constructor(x, y, arg=...)` | One-line distinguishing behavior. |

## Format or Mechanism B

Use this family when ...

| API | Description | Example and result |
|---|---|---|
| `convert(x, ...)` | One-line distinguishing behavior. | `convert(x)`<br>`# Result: ...` |

## Canonical Workflow

```python
# End-to-end construction, conversion, or execution path.
```

**Invariant:** State the correctness rule beside the workflow it constrains.

Add an optional family navigation map before these sections only when needed.
Keep each API under its family heading and in the first table column.

## Cross-Cutting Gotchas

- Include only behavior shared by multiple alternatives.
````
