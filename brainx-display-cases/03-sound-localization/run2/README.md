# Sound Localization from Timing

This example models the Jeffress delay-line principle: a sound produces one
spike in each auditory relay, internal delays align the two events at one
coincidence detector, and two readout neurons report left or right.

Positive interaural time difference (ITD) means the right-ear event arrived
later and is decoded as `LEFT`; negative ITD is decoded as `RIGHT`. Exact zero
is reported as `CENTER`.

The implementation uses:

- `brainpy-state` LIF populations for the auditory relays, coincidence bank,
  and readout neurons.
- `brainevent.BinaryArray` with fixed-fan-out projections for spike delivery.
- `brainstate.transform.for_loop` over time and state-aware `vmap2` over
  independent ITDs.
- `brainunit` quantities for time, voltage, current, resistance, and all model
  constants.

Run the demonstration:

```bash
python sound_localization.py
```

Run the focused checks with Python's standard library:

```bash
python -m unittest -v
```

ITDs are rounded to the nearest `0.05 ms` simulation step and must remain
within the modeled internal delay range of `+/-0.6 ms`.
