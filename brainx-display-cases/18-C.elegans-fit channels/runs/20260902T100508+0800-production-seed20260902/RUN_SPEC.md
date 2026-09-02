# Minimal first-principles SHK-1 and EGL-19 production run

- Run ID: `20260902T100508+0800-production-seed20260902`
- Parent evidence: `20260902T095151+0800-production-seed20260902` (preserved deterministic rejection).
- Level/entry: production/new scientific run
- Project root: `/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/18-C.elegans-fit channels`
- Specification SHA-256: `aaa9ca355e555a027acafbb4a97048144d4d5365a573d945518fb69d7d9c0afc`
- Entry-point SHA-256: `fbf5a8168da42a40bd7f63f6b9e2a6cfac2a38a191b67d110a2602958b54fb1b`
- Test SHA-256: `e48428392a9a96c83eb165eed38aa726d3f85748cc6efd0de9cfee7435eaa5d9`
- Git commit: `f100356e453dfa98b996c801a0a269eec7c7dcca`; case diff SHA-256 before launch: `f59e395e15e2871a8bb1465f632d30678ec82cd6605c68b4fac28a6ca997e4f6`.
- Data SHA-256: potassium `f107d3dd2feadcbb48df1dea85592c783a6557d2c44b802e0510bcc61a21ba2d`; calcium `6c7ba942579bc976707738b6f41c28b12beae5a05d27fe8ee98be19ba7390bd4`.
- Interpreter/backend/device: `/home/yixinliu/anaconda3/envs/braincell-released/bin/python`, JAX CPU, `CpuDevice(id=0)`.
- Precision and seed: package defaults; seed `20260902`; deterministic objective, starts, selection, and synthetic recovery.
- Expected runtime: under 90 seconds; stop at non-finite fit, nonzero exit, missing artifact, or 10 minutes.
- Required outputs: `raw/report.json`, `raw/fit_data.npz`, four nonblank PNG figures, `run.log`, and `exit_code`.

Smoke evidence: five tests passed in 12.179 seconds; BrainX and JAX preflight selected CPU.
