Implemented the complete BrainX experiment in [seizure_recruitment.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.o1d7VN/workspace/seizure_recruitment.py:47).

It uses BrainMass regional dynamics and coupling, BrainState `for_loop`/`vmap`, and BrainUnit time quantities. Results include:

- Local: only region 0 recruited at `k=0.15`, pulse `0.35`.
- Spread: all regions recruited at `k=0.45`, pulse `0.50`, with ordered onsets at `6.4`, `9.5`, `12.8`, and `16.5 ms`.
- Sweeps cover 80 coupling, delay, and pulse combinations.

Generated [seizure_recruitment.png](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.o1d7VN/workspace/seizure_recruitment.png) and documented usage in [README.md](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.o1d7VN/workspace/README.md:1).

Verification: `python -m unittest -v` passes both delay-phase and recruitment-order tests.