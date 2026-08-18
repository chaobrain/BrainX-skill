# Prior Bias under Ambiguous Evidence

This experiment asks whether a small prior input changes a binary decision mainly when sensory evidence is ambiguous. It simulates two noisy LIF populations with recurrent excitation and mutual inhibition, once without a prior and once with a small prior favoring choice A.

Run it with:

```bash
python prior_bias_decision.py
```

The script writes `prior_bias_results.png` and prints the probability of choosing A at each evidence level. The left panels show several zero-evidence decisions unfolding, the middle panel is the psychometric comparison, and the right panel reports measured first-call and steady compiled throughput on the current machine.

## BrainX structure

- `brainpy.state.LIFRef` supplies the two decision populations and their unit-aware membrane dynamics.
- `brainstate.transform.vmap2` gives every evidence, prior, and trial lane independent dynamical State and random draws.
- `brainstate.transform.for_loop` advances the recurrent circuit over time.
- `brainstate.transform.jit` compiles the complete reset-and-rollout operation used by the benchmark.
- `brainunit` keeps time, voltage, current, resistance, firing rate, and rate-to-current coupling dimensionally consistent.

The prior is only `+0.006 nA`, compared with evidence spanning `-0.030` to `+0.030 nA`. The quantitative check printed at the end compares the mean absolute probability shift for ambiguous (`|evidence| <= 0.006 nA`) and strong (`|evidence| >= 0.020 nA`) conditions.
