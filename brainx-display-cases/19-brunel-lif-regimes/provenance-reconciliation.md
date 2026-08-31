# Runtime provenance reconciliation

## Finding addressed

Iteration-3 review finding `PROV-001` identified that both production launch records name `/home/yixinliu/anaconda3/envs/braincell-released/bin/python`, while their prelaunch `environment.json` files incorrectly label that interpreter as Python 3.13.11 and the continuation process records Python 3.11.15.

## Evidence

- Both immutable `command.txt` files invoke the same absolute interpreter path.
- The completed continuation's in-process `results/provenance.json` records Python 3.11.15.
- The named interpreter currently reports Python 3.11.15; its executable timestamp is `2026-08-12 15:09:03.389389425 +0800`, before both 2026-08-29 runs.
- A process-captured runtime inventory from that interpreter reports the same BrainX dependency tuple as both prelaunch environment files: BrainX 2026.7.9, BrainEvent 0.1.2, BrainPy 2.8.1, BrainPy-State 0.1.0, BrainState 0.5.2, BrainTools 0.3.0, BrainUnit 0.5.1, JAX/JAXlib 0.10.2, NumPy 2.4.6, SaiUnit 0.5.1, and SciPy 1.17.1.

## Resolution

Treat only the Python field in the old prelaunch environment snapshots as invalid. Iteration 4 records `sys.executable`, Python implementation/version, BrainX dependency versions, backend, devices, and the source-manifest hash inside the executing process. The hardened continuation path verifies every inherited raw file against that manifest before reproducing aggregate outputs.
