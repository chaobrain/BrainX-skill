# Couple represented scales

Use this reference when separate modeling scales exchange signals during the experiment. Each package owns its model dynamics. The exchange contract defines what crosses the boundary, when it becomes available, and which model consumes it.

| Decision | What to record |
|---|---|
| Source and target | Name the source observable, target input, physical units, and the model or view owning each State. Record the source-to-target index mapping and preserve independent trial axes. |
| Clock | Name each integration step, observation window, sampling time, and delay. On different clocks, define the exchange grid, aggregation or hold rule, and the timestamp of the first step that can consume a sample. |
| Causal order | Specify which input each model reads before it advances and when each model publishes its output. Initialize boundary values explicitly. Keep any simultaneous coupling required by the equations in a joint integration contract. |
| Execution | Use owning-package orchestration where it supports the exchange. If the two models require an explicit joint step, use BrainState time control flow and keep the boundary State and axes visible. |

For a point-neuron population driving a neural mass, count spikes per trial in a declared window and divide by the number of represented neurons and the window duration to get Hz per neuron. Apply the model's input scaling and units to that rate. If the mass feeds back, map the mass observable to the point neurons' declared drive, such as a calibrated current or event rate. A completed window can drive only updates at or after its declared availability time. Match the exchange and hold rules to the scientific delay.

Check a small, deterministic input through the full exchange. A known spike count should produce the expected rate and arrive on the first permitted target step. Check the zero-coupling limit against independent models, preserve independent trial State, and compare baseline and intervention under identical clocks and recording windows. Repeat at a finer integration or exchange interval when the inference could depend on discretization.
