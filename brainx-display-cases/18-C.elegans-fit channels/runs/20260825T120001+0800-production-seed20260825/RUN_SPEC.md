# SHK-1 and EGL-19 channel fit replacement production run

- Run ID: `20260825T120001+0800-production-seed20260825`
- Parent evidence: `20260825T115108+0800-production-seed20260825` (preserved; gating-summary defects corrected before review).
- Level/entry: production/new scientific run
- Project root: `/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/18-C.elegans-fit channels`
- Specification SHA-256: `94a5d192d8ff025571f455f2bbe00dd8c705c380c8e8b47bace17e6a206fca53`
- Entry-point SHA-256: `3d29bde9dc8ff419d0aea4c7137c776e0bd9c3cd6dd77c5d355dd2923862e0aa`
- Test SHA-256: `fb9881841cf39eb0a3f23f4c5b8f3436a0953d9a9d4632cde5762721c61feddd`
- Git commit: `e1a589815cf3d357dd8c5e65938426fe9ac7b1a6`; case-file hashes are authoritative because the case is untracked.
- Data SHA-256: potassium `f107d3dd2feadcbb48df1dea85592c783a6557d2c44b802e0510bcc61a21ba2d`; calcium `6c7ba942579bc976707738b6f41c28b12beae5a05d27fe8ee98be19ba7390bd4`.
- Interpreter/backend/device: `braincell-released`, JAX CPU, `CpuDevice(id=0)`.
- Seed: `20260825`; deterministic objective, starts, selection, and nominal recovery.
- Expected runtime: under 150 seconds; stop at non-finite fit, nonzero exit, missing artifact, or 10 minutes.
- Required outputs: `raw/report.json`, `raw/fit_data.npz`, four PNG figures, `run.log`, and `exit_code`.

Smoke evidence: four tests passed in 33.616 seconds; nominal exact-pipeline recovery completed with finite output.
