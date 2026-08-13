# A Cortical Wave Meets an Obstacle

This experiment places one excitatory and one inhibitory leaky integrate-and-fire neuron at every point of a 2D sheet. A 3 ms current spark at the left edge launches a wave. A circular patch is silenced with both a hyperpolarizing clamp and spike masking, so it neither fires nor relays events.

The implementation is BrainX-native:

- `brainpy-state` supplies the E/I `LIFRef`, `Expon`, and `COBA` dynamics.
- `brainevent` carries binary spikes through explicit local CSR projections.
- `brainstate` owns dynamical state; `vmap2` gives every sweep condition independent state and `for_loop` advances the shared time axis.
- `brainunit` keeps space, time, voltage, current, conductance, and resistance dimensionally explicit.

## Run

```bash
export MPLCONFIGDIR=/tmp/cortical-wave-mpl
python cortical_wave.py
```

The default run sweeps seven lesion radii and six inhibitory gains around the sheet's propagation threshold. For a short smoke run:

```bash
python cortical_wave.py --quick --output quick-results
```

Run the focused checks with:

```bash
pytest -q
```

## Outputs

- `results/wave_storyboard.png`: matched snapshots of the intact and obstructed sheet.
- `results/phase_map.png`: categorical outcome across lesion radius and inhibitory gain.
- `results/outcomes.csv`: one row per condition with reach and route metrics.
- `results/phase_metrics.npz`: the same metric arrays for downstream analysis.

An outcome is `dies` if less than 12% of the far-right strip is recruited. At a narrow transect through the obstacle center, a surviving wave `splits` when both bypass corridors activate and `bends` when only one corridor activates; the intact sheet `crosses`. Each phase-map cell also reports the measured right-edge reach fraction, so the category is traceable to a quantitative result rather than visual judgment alone.
