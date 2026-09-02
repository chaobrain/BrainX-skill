# First-principles SHK-1 and EGL-19 production run

- Run ID: `20260902T095151+0800-production-seed20260902`
- Level/entry: production/new scientific run
- Project root: `/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/18-C.elegans-fit channels`
- Specification SHA-256: `6d27c42c903b85672791dd9b70ca2cf1b047f7bfa3ed8031ee7e9188434caa68`
- Entry-point SHA-256: `d2a8d9cfa1812f48b43d8824ceb19970374336de6e5a224f82341e42c045e927`
- Test SHA-256: `d071f925e752a22a79e3bb4d171c776d1f34e7366a4224302133e43b258d115b`
- Git commit: `f100356e453dfa98b996c801a0a269eec7c7dcca`; case diff SHA-256 before launch: `a90f5f0c7dcf582cff89f63b6072618ff3e4962e265c2d17d28237f54bb8ec3d`.
- Data SHA-256: potassium `f107d3dd2feadcbb48df1dea85592c783a6557d2c44b802e0510bcc61a21ba2d`; calcium `6c7ba942579bc976707738b6f41c28b12beae5a05d27fe8ee98be19ba7390bd4`.
- Interpreter/backend/device: `/home/yixinliu/anaconda3/envs/braincell-released/bin/python`, JAX CPU, `CpuDevice(id=0)`.
- Precision and seed: package defaults; seed `20260902`; deterministic objective, starts, selection, and synthetic recovery.
- Expected runtime: under 180 seconds; stop at non-finite fit, nonzero exit, missing artifact, or 10 minutes.
- Required outputs: `raw/report.json`, `raw/fit_data.npz`, four nonblank PNG figures, `run.log`, and `exit_code`.

Smoke evidence: five tests passed in 43.586 seconds; BrainX and JAX preflight selected CPU.
