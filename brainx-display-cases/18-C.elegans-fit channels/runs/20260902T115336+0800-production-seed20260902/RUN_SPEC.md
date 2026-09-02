# BrainCell-native first-principles channel production run

- Run ID: `20260902T115336+0800-production-seed20260902`
- Parent evidence: `20260902T101333+0800-production-seed20260902` (preserved; external review refused its analytic fitting path and incomplete identifiability evidence).
- Level/entry: production/new scientific run after review-driven restudy
- Project root: `/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/18-C.elegans-fit channels`
- Specification SHA-256: `d9e44f254041088cec344332c9fb6f768a6bf5a750dd63fc36a0a1fb75577120`
- Entry-point SHA-256: `0cc5694ac0ce71114b181373de4e66bd8a4d2bca01d167988adf60843282767a`
- Test SHA-256: `01d5ec7770db1d3a1ada39e4f1507d2529eb4349b80eeef8276acc01eeb5a565`
- Git commit: `f100356e453dfa98b996c801a0a269eec7c7dcca`
- Case diff SHA-256 before launch: `2ad2d5d13b1572dbc24e2b06ac1ed1b6869dbccb359df07183bf2aecde5ce596`
- Data SHA-256: potassium `f107d3dd2feadcbb48df1dea85592c783a6557d2c44b802e0510bcc61a21ba2d`; calcium `6c7ba942579bc976707738b6f41c28b12beae5a05d27fe8ee98be19ba7390bd4`
- Interpreter/backend/device: `/home/yixinliu/anaconda3/envs/braincell-released/bin/python`, JAX CPU, `CpuDevice(id=0)`
- Precision and seeds: package defaults; deterministic seeds derived from `20260902`
- Optimizer: BrainTools `ScipyOptimizer`, L-BFGS-B, bounded start-centered coordinates, BrainTools Huber objective
- Required validation: exact packed-metadata mapping, full-resolution outputs, three final centers, leave-one-voltage-out prediction, five noisy recovery truths per channel, BrainCell lifecycle/parity tests, and four requested figures

Production completed with exit code 0 in 253.153 seconds.
