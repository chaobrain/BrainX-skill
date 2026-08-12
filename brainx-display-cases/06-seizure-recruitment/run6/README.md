# Seizure recruitment across regions

`seizure_recruitment.py` starts a finite seizure-like perturbation in the first
of three Epileptor regions and maps when directed delayed coupling leaves it
local or recruits the two neighboring regions.

The experiment uses:

- `brainmass.EpileptorStep` for regional population dynamics and
  `brainmass.diffusive_coupling` for directed inter-region coupling;
- `brainstate.transform.for_loop` for time and
  `brainstate.transform.vmap` for the coupling, delay, and perturbation sweep;
- `brainunit` quantities for integration time, regional time constants,
  propagation delays, coupling strength, and stimulation parameters.

Run it from this directory:

```bash
python seizure_recruitment.py
```

It writes `outputs/seizure_recruitment.png` and the full, self-describing
`outputs/seizure_recruitment_results.npz`. Recruitment is classified as
`x1 > 0` continuously for 20 ms. Full propagation additionally requires onset
order `Focus < Neighbor 1 < Neighbor 2`. These thresholds and the displayed
parameter regimes are a deterministic phenomenological demonstration, not a
patient-calibrated clinical model.
