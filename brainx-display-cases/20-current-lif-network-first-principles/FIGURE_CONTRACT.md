# Figure contract

## Evidence and destination

- Mode: final archival/report figures.
- Acceptance: iteration-2 Codex review `PASS`; scientific outcome `PARTIALLY_SUPPORTED`.
- Sources: the three immutable iteration-2 seed directories and `runs/production-v2-combined-20260903/condition_assessment.json`.
- Destination: PNG files in `figures/`, rendered at fixed physical sizes and 180-220 dpi.
- Condition order: `(3,2)`, `(6,4)`, `(5,2)`, `(4.5,0.9)` in every figure.

## Four-condition rate and raster

- Question: what event timing and instantaneous population activity occur in each condition?
- Source: accepted display seed 11; the same fixed random 50-neuron sample in every condition.
- Data: `global_rate_hz` and `sample_spikes`, time-major at the recorded 0.1 ms interval.
- Transformation: discard 0-1 s transient; retain every remaining bin without smoothing or decimation. Draw the post-transient arithmetic mean as a dashed line.
- Encoding: one rate axis directly above one raster axis per condition; common 1-5 s time base and sample-row ordering; neutral raster marks and fixed condition colors for rate traces.

## E/I rates

- Question: how do excitatory and inhibitory mean rates compare across conditions and seeds?
- Sampling unit: seed, `n=3` per population and condition.
- Transformation: arithmetic mean over the 1-5 s analysis interval, already preserved in each metric record; no uncertainty model.
- Encoding: raw seed points plus a larger cross-seed mean marker on a shared logarithmic rate axis; population identity uses both color and marker shape.

## ISI CV

- Question: how do valid per-neuron ISI-CV distributions and seed medians relate to the frozen regularity thresholds?
- Sampling structure: 12,500 neurons nested within each of three seeds; all neurons are valid in all conditions. Per-neuron distributions are pooled only for descriptive shape; seed medians remain visible as the independent replicate summaries.
- Transformation: no exclusion beyond the locked minimum of two post-transient intervals; no smoothing beyond the violin density rendering.
- Encoding: fixed condition colors, three seed-median points per condition, and horizontal boundaries at CV 0.5 and 0.8; fixed y range 0-1.8 includes every valid value.

## Global-rate spectrum

- Question: where is global-rate spectral power concentrated, and are peak frequencies consistent across seeds?
- Sampling unit: seed, `n=3` per condition.
- Transformation: the frozen Welch spectra of demeaned post-transient rates (`hann`, 10,000 samples, 5,000 overlap, constant detrend, density scaling); display 1-500 Hz on a logarithmic power axis.
- Encoding: each seed spectrum as a faint line, cross-seed arithmetic mean as a strong line, and the accepted median dominant frequency as a dashed vertical marker.

## Fixed comparison settings

- Use the same condition-to-color mapping everywhere: teal, vermilion, blue, and amber in frozen condition order.
- Use seconds for time, hertz for rate and frequency, and dimensionless ISI CV.
- Do not present a requested regime label as verified unless `condition_assessment.json` marks it verified.
