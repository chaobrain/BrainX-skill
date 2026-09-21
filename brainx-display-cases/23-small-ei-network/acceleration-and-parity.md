# Acceleration and parity

The model already uses the package-owned execution path: one vectorized `LIFRef` population, two event-driven projections, and one `brainstate.transform.for_loop` over time. No custom timestep loop or lower-level JAX transform was added. Runtime benchmarking and numerical parity are pending because BrainPy-State and its dependencies are not importable in the current environment.
