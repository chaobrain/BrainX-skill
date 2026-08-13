Implemented the complete BrainX experiment in [spike_frequency_adaptation.py](/tmp/brainx-case-01-run2-xhigh.SFBo4P/workspace/spike_frequency_adaptation.py:1), with usage documented in [README.md](/tmp/brainx-case-01-run2-xhigh.SFBo4P/workspace/README.md:1).

At `10 uA/cm^2`:

- AHP present: ISI increases `11.96 -> 20.52 ms`, 27 spikes.
- AHP removed: ISI remains near constant `11.80 -> 11.05 ms`, 46 spikes.

Dynamic calcium provides the slow signal that gates the outward AHP potassium current. Setting `g_AHP=0` prevents that signal from slowing firing.

The BrainCell-native condition grid runs all current/adaptation combinations with unit-aware state, `for_loop` evolves time, and nested BrainState `vmap` calls summarize the sweep. The experiment and assertions pass, and `py_compile` succeeds.

[Open the generated comparison figure](/tmp/brainx-case-01-run2-xhigh.SFBo4P/workspace/spike_frequency_adaptation.png)