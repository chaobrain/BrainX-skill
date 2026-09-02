# Direct-bound BrainCell channel production run

- Run ID: `20260902T124800+0800-production-seed20260902`
- Parent evidence: `20260902T115336+0800-production-seed20260902` (preserved; external review refused hidden effective bounds and non-identical validation optimization).
- Level/entry: production/new scientific run after iteration-5 refusal
- Project root: `/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/18-C.elegans-fit channels`
- Specification SHA-256: `e5fa36f9482eba76370a295473f13bb310bcf5e97b2ec37e15aad2413ac36dd1`
- Entry-point SHA-256: `d710c1a5cc1c4461e1eed8a4e79c7c44f61edfe2dfb1705921b566ac909a89a3`
- Test SHA-256: `33fa50f47ae5eb13dd89dfd878174b1c10b7ad5864bdc333c896ffca51321423`
- Git commit: `f100356e453dfa98b996c801a0a269eec7c7dcca`
- Case diff SHA-256 before launch: `1066040b18a183c8988a7a15aba8ed954abe0b65ab2a1afe7748424bca54e345`
- Data SHA-256: potassium `f107d3dd2feadcbb48df1dea85592c783a6557d2c44b802e0510bcc61a21ba2d`; calcium `6c7ba942579bc976707738b6f41c28b12beae5a05d27fe8ee98be19ba7390bd4`
- Interpreter/backend/device: `/home/yixinliu/anaconda3/envs/braincell-released/bin/python`, JAX CPU, `CpuDevice(id=0)`
- Optimizer: BrainTools `ScipyOptimizer`, L-BFGS-B, direct declared physical bounds, six locked seeds, 1,200-iteration budget
- Pipeline identity: observed, recovery, and voltage-holdout fits use the same BrainCell objective, seeds, bounds, budget, and successful-candidate rule
- Required outputs: report, 94-array NPZ including raw validation traces/residuals, all optimizer histories and terminations, and four requested figures

Production completed with exit code 0 in 554.046 seconds.
