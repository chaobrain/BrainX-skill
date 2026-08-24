# Seizure recruitment across regions

`seizure_recruitment.py` perturbs one region in a directed three-region
Epileptor chain and maps when the event stays focal or recruits its neighbors.
The sweep varies dimensionless coupling and perturbation strength together with
unit-bearing propagation delay.

Run it from this directory:

```bash
python seizure_recruitment.py
```

The run uses `brainstate.transform.vmap` across independent conditions and
`brainstate.transform.for_loop` across time. It writes an auditable numeric
bundle to `results/seizure_recruitment.npz` and a summary figure to
`results/seizure_recruitment.png`. A burst is defined before the sweep as the
Epileptor LFP proxy staying below zero for 20 ms; the bundle retains every region's onset and
peak `x1`, all axis values, units, controls, protocol timing, and representative
traces.

The final two mapped conditions are mechanism controls: a strong perturbation
with coupling removed must remain in the focus, and strong coupling with the
perturbation removed must recruit no region. Parameters are phenomenological
and intended to demonstrate the recruitment boundary, not estimate a patient.
