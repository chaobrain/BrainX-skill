# BrainTools API gap and external boundary

Checked the owning BrainCell `references/braintools/metric.md` in full, including `braintools.metric.firing_rate`, `raster_plot`, spike synchrony metrics, `voltage_fluctuation`, and regression metrics. Checked BrainTools `Constant` in the input-current reference and used it for the main protocols.

`firing_rate(spikes, width, dt)` provides a smoothed, population-averaged time course. The required outputs here are unsmoothed **per-cell** counts in exact stimulation/late windows, joined to per-cell voltage variability, sodium availability, synaptic inward current and genotype, plus an explicitly specified synthetic optical mixture. The checked metric catalogue does not provide that compound observable or the calcium/block criteria. No fitting, generic solver, optimizer or alternative simulation infrastructure is used.

The external boundary is only offline NumPy reduction and serialization of already completed, unit-converted trajectories. Time is converted to ms, voltage to mV, sodium to mM and conductance to mS/cm2 before crossing it. Array axes remain time, condition, neuron. Rates are count divided by elapsed seconds. The custom block classifier and optical mixture are scientific definitions, not replacements for a general BrainTools algorithm.

Evidence: `validation.json` verifies finite/bounded trajectories, repeat/reset State parity, uncoupled condition independence and dt refinement. `metric-boundary-validation.json` compares the count normalization with BrainTools' population rate on a constant-rate train away from smoothing edges. `numerical-assessment.json` records the network timestep comparison after completion. NumPy never enters transformed membrane, synapse or random execution.

The revised rescue generates its unit-bearing current protocols once with `braintools.input.Constant` and passes their samples to `for_loop`. Only the channel ParamState intervention remains inside the step. Iteration 1's time-predicate current construction was not a valid API gap; it is retained solely in the original immutable run snapshot. `rescue-parity.json` verifies exact input and trajectory parity after correction.
