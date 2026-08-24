Built the runnable BrainX experiment in [spike_frequency_adaptation.py](/tmp/brainx-case-01-run1-xhigh.P4512G/workspace/spike_frequency_adaptation.py:30), with usage documented in [README.md](/tmp/brainx-case-01-run1-xhigh.P4512G/workspace/README.md).

It follows BrainCell’s dynamic-calcium `MixIons(k, ca)` pattern, uses BrainState `for_loop` and nested `vmap`, and keeps all physical parameters unit-aware. The controlled ablation sets only `g_AHP = 0`.

Verified by a full run:

- AHP removed: 45 spikes, ISI `11.24 → 11.05 ms`
- AHP intact: 27 spikes, ISI `11.88 → 20.52 ms`
- Late rate falls from `86.7 Hz` to `46.7 Hz` with adaptation

Generated figure: [spike_frequency_adaptation.png](/tmp/brainx-case-01-run1-xhigh.P4512G/workspace/spike_frequency_adaptation.png)

Run with:

```bash
python spike_frequency_adaptation.py
```