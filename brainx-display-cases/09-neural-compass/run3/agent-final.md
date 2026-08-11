Implemented the BrainX spiking neural compass in [neural_compass.py](/private/tmp/neural-compass-run3-eval.eaZNFf/workspace/neural_compass.py).

Results:

- A commanded `90 deg` dark turn produced `84.65 deg` of bump motion: gain `0.941`, final error `5.35 deg`.
- All 48 intact heading controls retained active, concentrated bumps.
- The permanent `75 deg` wedge lesion yielded 22 spared and 26 failed headings.
- No disturbed heading met the sustained-recovery criteria; the classifier does not force recovery to occur.

Artifacts: [figure](/private/tmp/neural-compass-run3-eval.eaZNFf/workspace/results/neural_compass_results.png), [per-heading CSV](/private/tmp/neural-compass-run3-eval.eaZNFf/workspace/results/lesion_sweep.csv), and [summary JSON](/private/tmp/neural-compass-run3-eval.eaZNFf/workspace/results/summary.json). The BrainX-native architecture and commands are documented in [README.md](/private/tmp/neural-compass-run3-eval.eaZNFf/workspace/README.md).

Verification: full simulation completed successfully and all 4 focused tests pass.