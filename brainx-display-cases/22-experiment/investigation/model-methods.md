# Model methods and assumptions

## Representation

Each circuit has two dimensionless population activations, E and I. Every condition evolves independently. The models have no individual spikes, seizure classifier trained on EEG, ion concentrations, motor system, anatomical propagation, or convulsion output. Persistent large population oscillations are a minimal network-instability proxy. A stable elevated rate is reported separately and is not silently called a seizure.

The equations use target-first weight names:

```text
tau_E dE/dt = -E + (1-E) F_E(wEE sE E - wEI sI I + P_E - dE h)
tau_I dI/dt = -I + (1-I) F_I(wIE sE E - wII sI I + P_I - dI h - L h)
F_j(x) = 1/(1+exp[-a_j(x-theta_j)]) - 1/(1+exp[a_j theta_j])
sE = 1 - release_E h; sI = 1 - release_I h
```

Here h is externally specified receptor activation, not a fitted pharmacokinetic model. Time constants are 10 and 20 ms; gains aE=1.2, aI=1; thresholds thetaE=2.8, thetaI=4; refractory factor is 1. Baseline drive PE=PI=1, wEI=15, wIE=12, wII=3. Initial E=I=0.05. All of these values are illustrative. Negative activation is possible with the stock shifted sigmoid in some unselected regimes; all reported central and neighboring conditions were checked for nonnegative trajectories.

Time starts with 1000 ms baseline, then h ramps linearly to 1 over 200 ms and remains there until 6000 ms. This compressed protocol permits baseline and asymptotic network comparison; it does not predict minutes-to-seizure, hM4Di dose-response, or ligand clearance. A baseline window at 600-950 ms is compared with late windows at 4000-5000 and 5000-6000 ms. Simulator monitors are post-update.

## Competing parameterizations

| Family | S2-like model | S1-like model | Intervention |
|---|---|---|---|
| Somatic, circuit difference | wEE=14 | wEE=12 | Syn: dE=.25, dI=.6; CaMKIIalpha: dE=.25 only; hDlx: dI=1 only |
| Somatic, targeting difference | wEE=14 | wEE=14 | Syn dI=.6 in S2 and .2 in S1, with dE=.25 in both; remaining conditions as above |
| Synaptic, circuit difference | wEE=14 | wEE=12 | Syn: 10% E-output loss and 50% I-output loss; CaMKIIalpha: E-output loss only; hDlx: I-output loss only |
| Downstream gate withdrawal | Receiving E/I circuit with wEE=14 | Same receiving circuit | S2-source withdrawal removes .6 units from downstream I drive; equally suppressed CaMKIIalpha output removes exactly the same drive; S1-source withdrawal is .2 |
| E-only counterexample | wEE=18, PE=2.5, PI=1 | Not a regional comparison | dE=.5, dI=0; baseline is a high-activity stable state |

Regional labels encode assumptions, not measured differences. In the downstream model L is prescribed loss of excitation to a receiving inhibitory population. There is no simulated S2 source, conduction delay, reciprocal loop or thalamic cell. This is the smallest model of pure output withdrawal; its failure does not reject richer distributed mechanisms.

Presynaptic factors multiply all outgoing connections of the affected cell population, including I-to-I. They change efficacy per unit presynaptic activity. Total inhibitory output sI*I can increase after recurrent recruitment even though sI is reduced. Thus late I firing or average GABA output alone cannot diagnose the direct hM4Di effect.

## Selection and validation

Settings were selected after exploratory parameter sweeps. These are constructed sufficiency examples, not fitted or blind predictions. Initial exploratory raw data and all failed settings are retained. The initial synaptic variant sweep had a weight-convention error and is excluded; `runs/variants_corrected` uses corrected equations. Its surviving settings were frozen before central and neighboring validation.

The central negative-control comparison is considered reproduced only when the S2-Syn condition has E peak-to-trough amplitude >0.1 and >=3 upward midpoint crossings in each late window, while the CaMKIIalpha-S2 and Syn-S1 controls lack that predicate. The >0.1 amplitude requirement excludes numerical fluctuations counted by midpoint crossings. Save amplitude, mean, crossing counts and full trajectories for every condition. Baseline peak-to-trough must be <0.005; all trajectories must be finite and nonnegative. These thresholds define a model proxy, not clinical diagnostic criteria.

Sensitivity varies S2 wEE over 13.5, 13.75, 14, 14.25, 14.5 and somatic I suppression over .5, .6, .7; synaptic E loss over .05, .1, .15 and I loss over .45, .5, .55; targeting-only S1 I suppression over .1, .2, .3. Every grid setting includes independent baselines and controls. Report grid fractions as sampled regimes, never probabilities for an animal.

The central run uses float64, CPU and stock RK4 at dt=.2 ms. Repeat at dt=.1 ms and with stock exponential Euler at .05 ms. Sampling intervals are 1 ms, but first post-update sample differs with dt; compare physical windows and invariant amplitudes/means, not index-aligned phase-sensitive trajectories. Focused tests independently check derivatives, input timing, reset, stock parity, and single/batched/JIT execution.

## API discrepancy handled explicitly

The rendered BrainMass documentation and public executable source use opposite cross-weight naming. Config files use biological target-first names; the constructor maps stock `wIE=config.wEI` and `wEI=config.wIE`. The independent derivative test guards that boundary. See [official public implementation](https://brainx.chaobrain.com/brainmass/_modules/brainmass/wilson_cowan.html). The model subclasses only the public derivatives and update to represent dynamic synaptic loss; BrainMass owns integration and execution. No installed package source was inspected.

## Why this can be paradoxical

Around a stable equilibrium, let gE and gI be the local slopes of the equilibrium population transfer functions. For small somatic suppressions dE and dI:

```text
Delta E = gE [gI wEI dI - (1 + gI wII) dE] / D
D = (1 - gE wEE)(1 + gI wII) + gE gI wEI wIE
```

With positive D, removing inhibitory recruitment can outweigh direct E suppression. Larger changes can cross an oscillatory instability. Equal receptor activation is not equal functional suppression. Inhibition-stabilized-network theory can explain paradoxical responses, but the mere presence of paradoxical activity does not establish an ISN, and a stable ISN need not seize. The finite-amplitude runs provide the actual numerical sufficiency evidence.
