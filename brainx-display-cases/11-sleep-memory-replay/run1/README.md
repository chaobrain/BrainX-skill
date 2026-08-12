# Memories Replaying During Sleep

This BrainX experiment trains four place-cell assemblies on `A -> B -> C -> D`,
runs matched networks through an uncued sleep period, suppresses excitatory
recurrent replay in one network of each pair, and compares ordered recall after
cueing only A.

```bash
MPLCONFIGDIR=/tmp/brainx-mpl JAX_PLATFORMS=cpu python sleep_replay.py
```

The script prints forward/backward replay counts and matched recall scores,
writes `sleep_replay_results.png`, and saves the underlying per-lane evidence
to `sleep_replay_evidence.npz`. The model is a compact phenomenological
demonstration, not a fit to a particular animal or recording session.

The causal comparison is narrow: sleep inputs, learned weights, neural state,
and recall protocol are matched within each pair; only excitatory recurrent
transmission during sleep is gated. Matched intrinsic fluctuations include rare
assembly-level excitability bursts at random places and times, never the learned
route order. The external place-cue tensor is identically zero during sleep.
