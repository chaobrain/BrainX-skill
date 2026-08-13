# Mouse V1 single-cell influence mapping

This repository runs one frozen, mechanistic point-neuron reproduction of the
main influence-mapping tests in Chettih and Harvey (2019). It uses a recurrent
conductance-based E/I spiking network, repeated unperturbed tuning trials, and
exactly matched baseline-perturbation pairs.

Run the preregistered experiment:

```bash
python v1_influence.py
```

The command writes a human-readable report, figure, complete parameter and
result JSON, pair-level CSV, and compressed trial summaries to
`results/frozen_run/`. The default run is intentionally a single network seed
and does not tune or retry failed signatures.

Run focused analysis/protocol tests with:

```bash
pytest -q
```

The paper protocol represented here is eight low-contrast drifting directions,
photostimulation at visual onset, four 32 ms sweeps at 15 Hz, and a 367 ms
response window. Optical excitation is approximated by target-only somatic
square current. See the generated report for the measured spike dose, protocol
limitations, phenomenological connectivity assumptions, statistical tests,
and uncertainty scope.

Reference: Chettih, S. N. & Harvey, C. D. (2019), *Nature* 567:334-340,
https://doi.org/10.1038/s41586-019-0997-6
