# BrainX acceleration audit

| File/area | Pattern | Evidence | Axis | Decision |
|---|---|---|---|---|
| models.simulate | already compiled | BrainMass Simulator owns initialization, JIT and transformed time loop | time | Unchanged |
| EICircuit | native independent State | Broadcast condition-specific parameters over one dimension | conditions | Unchanged |
| experiment.metrics | host statistics | Summaries operate only after device-to-host serialization boundary | observations | Keep outside dynamics |

No acceleration rewrite was needed. `validation.json` records exact reset and stock parity and agreement of single, batched and non-JIT paths. Wall times for runs include compilation and I/O and are not warm throughput measurements. No stochastic trials or gradients are present.

## BrainTools API boundary

No external solver, optimizer, input generator, or numerical loop replaces BrainX infrastructure. BrainMass Simulator's documented callable input interface directly represents the activation ramp. Population equations extend the stock public derivative only to add the declared synaptic mechanism. NumPy measures amplitude and counts crossings as explicit host-side observation logic; it does not integrate dynamics or generate a seizure label from unrelated statistical thresholds.
