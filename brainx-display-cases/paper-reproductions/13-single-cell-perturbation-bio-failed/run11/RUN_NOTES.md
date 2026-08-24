# Run notes

## Status

This retained run failed its causal-control assertion before analysis. Across
512 matched pairs, external currents were bit-identical and inhibitory pre-onset
spikes matched, but 147 packed excitatory pre-onset bytes differed. No scientific
influence result from this run is valid. The next run must isolate whether this
arises from lane initialization or audit-axis semantics before another full run.

## Frozen retry contract

This run retains all frozen scientific and analysis parameters. Each 16-trial
mapping chunk is duplicated into one 32-lane rollout: baseline lanes first and
perturbation lanes second. The pairs share constant initial state, stimulus,
and noise within the same transformed call; only the target-current mask differs.
