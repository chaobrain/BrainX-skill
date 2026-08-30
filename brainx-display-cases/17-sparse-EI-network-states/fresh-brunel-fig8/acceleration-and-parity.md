# Acceleration and parity

- Command: `MPLCONFIGDIR=/tmp/matplotlib-case17-fresh JAX_PLATFORMS=cpu python benchmark_parity.py`
- Backend: CPU
- Workload: 1,000 neurons for 1,000 steps with the complete State, delay, sparse-event, and external-Poisson update path.
- Cold compiled execution: 13.8570 s
- Warm compiled execution: 1.1607 s
- Spike count: 8,793
- Spike output: bit-identical to eager execution
- Final voltage maximum absolute difference: 0.0 mV

The production runner keeps the whole time loop inside `brainstate.transform.for_loop` under one state-aware JIT. Sparse recurrent communication remains event-driven through `brainevent.FixedNumPerPost`; no scientific parameter, time step, delay, RNG sequence, or observable changed during acceleration.
