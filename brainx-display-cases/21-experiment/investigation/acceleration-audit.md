# BrainX acceleration audit

| File/area | Pattern | Evidence | Axis | Impact | Risk | Rewrite | Confidence |
|---|---|---|---|---|---|---|---|
| Cell/network rollout | Already compiled | BrainState jit(for_loop), no host operations per step | T | High | State ownership | None | High |
| Neurons/fractions | Native arrays | Independent V/gates/Na/synaptic State of shape (fraction, neuron) | N, B | High | Accidental sharing | None; tested uncoupled equality | High |
| Communication | Event operator | BinaryArray @ condition-specific weight matrix; pure vmap | N | Moderate | Direction/units | None; density initializer corrected before validation | High |
| Seeds/drives | Host orchestration | Reuse compiled current runner within each seed | E | Moderate | Full-history memory | Keep serial runs and immutable artifacts | High |
| Raw recording | Full history | V/events/h/Na/gsyn needed to audit block versus small oscillations | T | Large disk cost | Omitting decisive observables | Retain full trajectories for this small-network study | High |

## Patch / rewrite plan

No approximate optimization. Use the initially compiled implementation and record correctness/timing; do not claim a speedup over an unmeasured alternative. BrainState owns mutable transforms; raw JAX vmap wraps only pure event-array communication. Initial topology is dense for 40 cells with p=0.3, not a densified large sparse graph.

## Validation plan

Completed `test_models.py`: exact repeated-reset output parity and uncoupled State independence; measured cold-plus-first and warm execution with device synchronization. `validation.json` records timing and refinement. Gradients are not applicable. Per-seed source snapshots and configs identify every run.

## Remaining risks

Full-history retention and NPZ compression dominate storage; CPU wall times vary with concurrent runs and are not benchmark comparisons. Event-threshold networks can have phase-sensitive trajectories, so network refinement compares rate, block fraction and optical-sign conclusions rather than requiring pointwise long-horizon waveform equality. Three seeds do not quantify biological uncertainty. No GPU or multi-device claim is made.
