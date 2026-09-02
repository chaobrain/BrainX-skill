# Closed-architecture BrainCell channel production run

- Run ID: `20260902T132147+0800-production-seed20260902`
- Parent evidence: `20260902T124800+0800-production-seed20260902` (preserved; external review required full-budget architecture closure).
- Git commit: `f100356e453dfa98b996c801a0a269eec7c7dcca`
- Specification SHA-256: `7fe24726582e9bb8e215e45a740969a2b4cf0dccf20bd9be4813d05dfe8aece5`
- Entry-point SHA-256: `9d5db2c1d52b36113a2e025f6b5e402fb75c142f32f1edde4d06982b2a725c1b`
- Test SHA-256: `33fa50f47ae5eb13dd89dfd878174b1c10b7ad5864bdc333c896ffca51321423`
- Case diff SHA-256: `c966f4ef334833202f1d32c9c1a56272bce912c69dc1ae2e09bd2619479c18a1`
- Optimizer: BrainTools L-BFGS-B, direct physical bounds, six locked seeds, 1,200 iterations for global and `m^4h` comparison fits
- Architecture closure: three successful `m^4h` candidates; robust objective improvement 3.1783%; BIC 7979.82 (`m^4`) versus 8013.42 (`m^4h`), delta +33.60 favoring `m^4`
- Pipeline identity: observed, recovery, and voltage-holdout fits use the same BrainCell objective, six seeds, physical bounds, budget, and successful-candidate rule

Production completed with exit code 0 in 629.743 seconds.
