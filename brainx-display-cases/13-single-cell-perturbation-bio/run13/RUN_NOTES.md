# Run notes

Diagnostic run for direct matrix versus explicitly vmapped BrainEvent CSR
communication. This directory is not a scientific influence-mapping run.

Result: direct matrix input and explicit `jax.vmap` were identical, and both
preserved duplicated lanes exactly. BrainEvent batching is not the mismatch.
