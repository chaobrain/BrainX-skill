Implemented the BrainX-native network in [sound_localization.py](/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.6fAzYl/workspace/sound_localization.py:1).

It uses unit-aware LIF populations, BrainEvent fixed fan-out projections, coincidence-delay taps, `for_loop` over time, and state-aware `vmap2` across independent ITDs.

Run:

```bash
python sound_localization.py
```

Verified all 25 supported delays from `-0.6 ms` to `+0.6 ms`: negative decodes RIGHT, positive decodes LEFT, and zero decodes CENTER. Shape validation, unit validation, delay timing, and mapped-state isolation also pass.