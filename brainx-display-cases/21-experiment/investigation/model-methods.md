# Models and interpretation boundaries

## Equations

The membrane model extends the [Hodgkin-Huxley formulation](https://doi.org/10.1113/jphysiol.1952.sp004764), not a fitted human-neuron reconstruction. All currents below use an inward-positive convention. Each isopotential BrainCell neuron obeys

```
C dV/dt = Iext + gNa m^3 h (ENa - V) + gK n^4 (EK - V)
          + gL (EL - V) + gKNa a(Na) (EK - V) + gsyn (0 - V)
dx/dt = alpha_x(V)(1-x) - beta_x(V)x, x in {m,h,n}
dNa/dt = max(INa, 0)/(F depth) - (Na - Na_rest)/tau_Na
a(Na) = (Na / Khalf)^3 / [1 + (Na / Khalf)^3]
dgsyn_j/dt = -gsyn_j/tau_syn + sum_i w_ij q_i spike_i(t)
```

`m,h` use standard HH sodium rates; `n` uses BrainCell's HH potassium channel with `V_sh=-45 mV`. `F` is the Faraday constant. The sodium sensor is a phenomenological local pool; ENa remains fixed, so this is **not** a mass-conserving intracellular sodium model. Potassium/chloride accumulation, ATP regulation, pumps, calcium-dependent channels, morphology, synaptic depression, and homeostatic dynamics are omitted.

Common parameters: C=1 uF/cm2, gNa=120 mS/cm2, ENa=50 mV, EK=-77 mV, gL=0.3 mS/cm2, EL=-54.387 mV, Na_rest=5 mM, Khalf=20 mM, Hill exponent=3. These are illustrative, not measurements from the user's cells. In particular Khalf=20 mM is **not** the recombinant Slick EC50 reported in the channel literature. The effective pool and sensitivity jointly determine activation and are not separately identifiable here.

Exploration compared gK=36 and 18 mS/cm2 and sodium time constants 100 and 10 ms, with depth 0.1 and 0.01 um respectively. These matched time/depth ratios preserve a comparable mean loading scale while changing timing. The selected gK=18 regime makes sustained firing more dependent on KNa. This is explicit exploratory selection, not an independent reproduction of a human-neuron parameter set. All exploratory artifacts are retained.

## Network and interventions

Use explicit excitatory neurons with BrainPy-State exponential conductance synapses (tau=5 ms, reversal=0 mV) and BrainEvent communication. Event propagation occurs one integration step after a presynaptic upward crossing of -20 mV. This is a simplified effective-output criterion; small oscillations can fail the criterion without being true block. There is no inhibitory population in this model.

For each seed, fix random directed connectivity (probability 0.3, no autapses), current heterogeneity (10% SD), initial-voltage heterogeneity (2 mV SD), and a nested randomized order of targeted cells. Across fractions, only the targeted-cell conductance and, in the alternative model, their release factors change. Each fraction has independent dynamical State. Targeting fraction and remaining conductance per targeted cell are different parameters. Density, number of cells and network topology do not change with targeting fraction.

- Channel-loss model: q_i=1; targeted gKNa falls to zero or a specified residual fraction.
- Effective-transmission-loss model: targeted gKNa falls and q_i<1 attenuates their outgoing synapses. This is an **assumed** presynaptic/effective transmission impairment; it does not simulate or demonstrate homeostatic adaptation.
- Uncoupled control: all weights zero. At identical input and independent cells, the expected population output is `(1-f)*R_control + f*R_KD`; changing f alone cannot produce a new nonlinear dose dependence. Within-well genotype contrast remains distinct from between-well comparison.
- Channel rescue: restore conductance without changing cell identity, density or input. Transmission rescue: set q=1 with channel loss retained. Acute current reduction tests relief from an overdriven state.

## Observables

Record raw V, threshold events, sodium-channel availability h, local sodium and synaptic conductance. Report 500-ms whole-step event rates and final-100-ms voltage statistics separately. A block flag requires no late events, late mean voltage >-50 mV, SD<3 mV and mean h<0.3. It is a diagnostic operational definition; inspect raw traces before inferring biological block. A low event count alone is never sufficient.

The optical observation sensitivity model has two components:

```
CV = mean[sigmoid((V+30)/6) * max(120-V,0)/120]
CS = mean[gsyn * max(-V,0)]
C_alpha = alpha * CV/CV_control + (1-alpha) * CS/CS_control
```

CV represents noninactivating voltage-dependent calcium entry; CS represents receptor/synaptic-associated calcium. Their normalization is fixed to the all-control well **within the same seed and drive**, never separately to each genotype or fraction. Alpha is an unmeasured mixture weight, scanned from 0 to 1; it is not fitted CaMPARI2 physiology. The network does not include calcium-carrying receptors explicitly. Sustained calcium can inactivate or be extruded in real cells; this observation model omits those processes, fluorescence saturation, photoconversion kinetics and assay timing. It only tests whether plausible opposing contributions can produce opposite optical contrasts. It cannot predict a photoconversion ratio or reproduce a five-minute stimulation assay quantitatively.

## Numerical execution and validity

BrainState compiles the time loop; BrainUnit preserves physical units until offline NumPy analysis. BrainTools constructs step inputs. Use RK4, dt=0.025 ms, 100-ms baseline and 500-ms stimulation unless the saved config specifies otherwise. Reset all membrane, gate, sodium and synaptic State between drives. No parameter fitting or training is performed.

`test_models.py` checks current signs/reversals, loading/removal, finite gate rates, exact repeated-reset parity, uncoupled fraction independence, targeting and autapse invariants, the distinction between silence and block, and a twofold timestep refinement. Experiment output validation rejects nonfinite trajectories, invalid h and negative sodium. Configs, source snapshots, environment identities, raw trajectories and metrics accompany the final runs. Random-seed variability is simulation variability, not biological uncertainty or confidence intervals.
