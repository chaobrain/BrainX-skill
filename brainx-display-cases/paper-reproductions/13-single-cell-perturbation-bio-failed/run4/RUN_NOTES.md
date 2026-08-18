# Run notes

This frozen-parameter attempt is retained as a failed run. Unit tests passed
(`4 passed`), but the full simulation stopped before analysis with a JAX
`UnexpectedTracerError` when the same stateful network was compiled first with
batch size 64 for tuning and then with batch size 32 for influence mapping.

No scientific result can be inferred from this run. The next attempt must use a
single fixed batch shape while preserving exact baseline-perturbation state
restoration. No model or analysis parameter should be changed in that retry.
