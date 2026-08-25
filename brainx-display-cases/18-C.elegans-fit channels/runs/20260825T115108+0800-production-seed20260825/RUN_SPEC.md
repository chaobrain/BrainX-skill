# SHK-1 and EGL-19 channel fit production run

- Run ID: `20260825T115108+0800-production-seed20260825`
- Level: production
- Entry case: new
- Project root: `/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/18-C.elegans-fit channels`
- Specification: `NeuroSpecification.md` SHA-256 `94a5d192d8ff025571f455f2bbe00dd8c705c380c8e8b47bace17e6a206fca53`
- Entry point: `fit_channels.py` SHA-256 `8d32d10ad99d551fab4ae7f3b98a08860e6d215dddc182665cb393dad707ef3c`
- Tests: `test_fit_channels.py` SHA-256 `fb9881841cf39eb0a3f23f4c5b8f3436a0953d9a9d4632cde5762721c61feddd`
- Git commit: `e1a589815cf3d357dd8c5e65938426fe9ac7b1a6`; the case is untracked, so file hashes are authoritative.
- Data: potassium SHA-256 `f107d3dd2feadcbb48df1dea85592c783a6557d2c44b802e0510bcc61a21ba2d`; calcium SHA-256 `6c7ba942579bc976707738b6f41c28b12beae5a05d27fe8ee98be19ba7390bd4`.
- Interpreter: `/home/yixinliu/anaconda3/envs/braincell-released/bin/python`
- Backend/device: JAX CPU, `CpuDevice(id=0)`; BrainState default precision.
- Seed: `20260825`; deterministic objective and deterministic multi-start selection.
- Expected runtime: under 120 seconds after imports; no checkpoint is required for this bounded run.
- Required outputs: `raw/report.json`, `raw/fit_data.npz`, four PNG figures, `run.log`, and `exit_code`.
- Stop conditions: non-finite fit, missing required artifact, nonzero exit, or runtime above 10 minutes.
- Retry budget: zero unchanged retries for deterministic failure.

The pre-production smoke gate is the four-test suite recorded in `acceleration-and-parity.md`; it passed on the same interpreter and backend.
