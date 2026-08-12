# Memories Replaying During Sleep

This BrainX example teaches a recurrent spiking network the route `A -> B -> C
-> D`, lets its state continue into a zero-sensory-input sleep phase, blocks
recurrent transmission in a matched group, and tests both groups with only an
`A` cue afterward.

The model uses:

- `brainpy-state` for unit-aware LIF place cells and conductance synapses;
- `brainevent` for binary spike transmission and dense event-driven STDP;
- `brainstate` for neural/plasticity state, `for_loop` over time, and a
  state-aware `vmap2` across matched networks;
- `brainunit` for time, voltage, resistance, current, and conductance.

Run it with the BrainX environment:

```bash
python sleep_replay.py
```

The program reports the replay direction from the time-resolved order of place
events, verifies that paired weights are identical before sleep, compares
sleep-dependent weight changes and ordered recall, and writes
`sleep_replay_results.png`.

The calibrated regime is phenomenological: it demonstrates the causal
experiment and analysis, rather than fitting a particular animal or dataset.
