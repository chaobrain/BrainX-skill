# Prior Bias under Ambiguous Evidence

This experiment asks whether a small prior changes a two-choice decision mainly
when sensory evidence is ambiguous. Two noisy LIF populations compete through
recurrent excitation and mutual inhibition. A brief current pulse favors choice
A before evidence arrives; signed sensory current then favors A or B from weak
to strong levels.

The simulation is BrainX-native:

- `brainpy.state` supplies the LIF populations and conductance-based recurrent
  and inhibitory projections.
- `brainstate.transform.for_loop` advances time, state-aware `vmap2` gives every
  evidence/bias/trial lane independent dynamical state, and
  `brainstate.transform.jit` compiles the complete rollout.
- `brainunit` keeps time, voltage, current, conductance, and decision time units
  attached through the model. Raw arrays are used only for host-side analysis,
  timing, and plotting.

Run the complete experiment with:

```bash
MPLCONFIGDIR=/tmp/brainx-mpl python prior_bias_decision.py
```

The script writes the combined trajectory, psychometric, and speed figure to
`results/prior_bias_decision.png`, with exact probabilities, decision times,
and benchmark values in `results/summary.json`. The speed bar uses synchronized
wall-clock timings and reports total condition-equivalent simulated time per
wall second; state restoration is outside the timed region.

Choices use the sign at the first crossing of a fixed accumulated population
spike-difference threshold. Trials that do not cross before 300 ms use the final
sign, and exact ties contribute one half to the estimated choice probability.
