# KCNT2 investigation

This is an illustrative literature-guided mechanism comparison, not a fitted model of a particular culture.

- `literature-review.md`: primary-source evidence and reading depth.
- `model-methods.md`: equations, parameters, observables and limitations.
- `follow-up-experiments.md`: discriminating experiments and interpretation rules.
- `report.md`: final scientific assessment and claim-evidence matrix.
- `models.py`, `experiment.py`, `rescue.py`: executable BrainX models and interventions.
- `test_models.py`, `validation.json`: focused numerical and execution checks.
- `configs/`, `runs/`: frozen configurations and raw results; exploratory and failed runs are retained and labeled.
- `analyze.py`, `assessment.json`, `fraction-summary.csv`, `artifact-manifest.json`: deterministic analysis and provenance.
- `reviews/`: independent numerical review; iteration 2 passed after an exact-parity native-input correction.

## Reproduce

The existing environment used here is `/home/yixinliu/anaconda3/envs/braincell-released/bin/python`. No packages were installed or upgraded. Run from this directory. Use a new output path for each experiment; the runner refuses to overwrite a run.

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/kcnt2-mpl XDG_CACHE_HOME=/tmp/kcnt2-cache /home/yixinliu/anaconda3/envs/braincell-released/bin/python test_models.py
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/kcnt2-mpl XDG_CACHE_HOME=/tmp/kcnt2-cache /home/yixinliu/anaconda3/envs/braincell-released/bin/python experiment.py --config configs/block_validation.json --out runs/block_replication_new
/home/yixinliu/anaconda3/envs/braincell-released/bin/python analyze.py
```

Replication requires BrainX, BrainCell, BrainPy-State, BrainState, BrainUnit, BrainEvent, BrainTools, JAX and NumPy. Exact package provenance is recorded separately. Source snapshots accompany noninitial runs. Model rates use current **density**, not pA; no cell area or glutamate-to-current calibration was assumed. Numeric model fractions and optical mixture weights must not be interpreted as estimated biological thresholds or reporter parameters.

Raw NPZ trajectories remain available locally and are indexed by SHA-256 in the artifact manifests. `artifact-manifest-iteration-2.json` extends the preserved original manifest with the corrected, bit-identical rescue run. They are ignored by Git because the full evidence set occupies several GB; source, configs, summaries and provenance are not ignored.
