# Final minimal first-principles channel production run

- Run ID: `20260902T101333+0800-production-seed20260902`
- Parent evidence: `20260902T100508+0800-production-seed20260902` (preserved deterministic rejection).
- Level/entry: production/new scientific run
- Project root: `/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/18-C.elegans-fit channels`
- Specification SHA-256: `2f8e221a20c1b4e3810767920effbb9c5d888ade494d191bd79a4eb3b3bf7134`
- Entry-point SHA-256: `f4efa9b39e078e11c92f3cbd1f48d7c0252c1f6ad46444844410d2cae0b55c80`
- Test SHA-256: `e48428392a9a96c83eb165eed38aa726d3f85748cc6efd0de9cfee7435eaa5d9`
- Git commit: `f100356e453dfa98b996c801a0a269eec7c7dcca`; case diff SHA-256 before launch: `098927acf59d65e206e5da7d1c19fff4d49b70c2efd726c36995be2d46764a75`.
- Data SHA-256: potassium `f107d3dd2feadcbb48df1dea85592c783a6557d2c44b802e0510bcc61a21ba2d`; calcium `6c7ba942579bc976707738b6f41c28b12beae5a05d27fe8ee98be19ba7390bd4`.
- Interpreter/backend/device: `/home/yixinliu/anaconda3/envs/braincell-released/bin/python`, JAX CPU, `CpuDevice(id=0)`.
- Precision and seed: package defaults; seed `20260902`; deterministic objective, starts, selection, and synthetic recovery.
- Expected runtime: under 90 seconds; stop at non-finite fit, nonzero exit, missing artifact, or 10 minutes.
- Required outputs: `raw/report.json`, `raw/fit_data.npz`, four nonblank PNG figures, `run.log`, and `exit_code`.

Smoke evidence: five tests passed in 12.585 seconds; BrainX and JAX preflight selected CPU.
