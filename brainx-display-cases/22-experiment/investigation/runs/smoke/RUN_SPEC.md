# Run specification

Run: smoke; level: smoke; new immutable CPU run.

Command: `/home/yixinliu/anaconda3/envs/braincell-released/bin/python experiment.py --config configs/smoke.json --run-id smoke`.

Configuration SHA256: `ccd92d64e88f911b634c23b4b000bb835cb0ade553326d9b78627e3359c29c0d`.

No supplied raw data; deterministic illustrative population dynamics; no seed. Expected raw.npz, metrics.csv, summary.json, run.log, status.json and exit_code. Estimated < 2 minutes and < 1 GB RAM. Stop on exception/nonfinite output; no automatic retry.
