# Independent Codex review request

Review this fresh BrainX reproduction of Brunel (2000), Fig. 8. Work read-only. Return exactly one leading verdict, `PASS` or `REFUSE`, followed by evidence-based findings ordered by severity. Do not modify files.

## Fresh-project boundary

The researcher explicitly required a new project without reading prior memory. Review only this directory. It is a new untracked addition under the required `17-sparse-EI-network-states` parent; `git status --short -- .` should show only `?? ./`. Treat any dependence on material outside this directory as a critical failure. The literal request is in `request.md`; the prospective contract is in `NeuroSpecification.md`.

## Scientific and software checks

1. Verify the implementation is BrainX-native and that the biological point-neuron owner is `brainpy.state.Neuron`, with BrainEvent fixed fan-in, BrainState State/delay/RNG/JIT, and BrainUnit quantities.
2. Check Brunel Model A equations, parameters, fixed unique fan-in/no autapses, summed external Poisson conversion, 1.5 ms physical delay, refractory/reset behavior, condition reuse/reset/reseed, analysis windows, and raw serialization.
3. Check that every categorical criterion in `assess_condition()` was declared prospectively in `NeuroSpecification.md` and that descriptive population-rate CV is not used as an undeclared gate.
4. Inspect `test_brunel_fig8.py`, `test-results.md`, `benchmark_parity.py`, and `acceleration-and-parity.md` for relevant validation and eager/JIT parity.
5. Inspect the smoke snapshot `runs/20260830T100015Z-smoke-fresh-isolated/` and the production snapshot `runs/20260830T100605Z-production-fresh-isolated/`. Recompute hashes if useful. Confirm the outer manifest binds the run contract, environment, command, addition-only `code.diff`, source hashes, and all results; mutable log/status/exit files are intentionally excluded.
6. Inspect production raw data and metrics rather than trusting prose. Confirm A/B pass and C/D failures are represented honestly. Review whether conclusions stay within the locked finite-seed claim boundary.
7. Confirm no PNG or rendered figure exists before review. Review the renderer for correctness, but do not invoke it.

The review is a gate: use `REFUSE` for any critical or major scientific, lifecycle, provenance, fresh-boundary, or artifact-integrity problem. Use `PASS` only when the evidence is sufficient to permit the existing renderer to create the final figure from the accepted immutable production raw data.
