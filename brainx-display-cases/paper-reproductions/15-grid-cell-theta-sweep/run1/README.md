# Alternating theta sweeps in a direction-grid network

This experiment implements a phenomenological firing-rate mechanism in which a
theta-modulated head-direction ring organizes spatial sweeps in three toroidal
grid-cell modules. Slow cell-wise adaptation makes the direction bump alternate
between the left and right flanks of the animal's heading. An effective
conjunctive population transforms the complete ring state into shifted toroidal
input; it never receives or imposes a decoded trajectory.

## Main result

After a 1 s warm-up, the straight-run ring alternated on 47 of 49 adjacent cycle
pairs (score 0.959). Randomly permuting the 50 cycle angles gave mean 0.469 and a
95% interval of [0.327, 0.612]; the one-sided permutation p-value was 0.00025
(4,000 shuffles). Grid alternation was 0.939 in every module. At the evaluation
phase, ring-grid direction alignment cosines were 0.991, 0.991, and 0.990; over
the active half of theta they were 0.992, 0.991, and 0.991.

The causal controls separate the mechanisms. Removing adaptation or theta
abolished direction sweeps and alternation. Removing ring-grid coupling retained
ring alternation (0.959) but abolished grid alternation and reduced decoded grid
displacements below 0.5 cm. The result therefore depends on adaptation plus
theta in the ring and on the conjunctive projection for transmission to grid.

## BrainX execution

- BrainMass owns the aggregate-population modeling scale and supplies the
  plotting interface. The specialized continuous attractor is a custom
  `DirectionGridStep`, following BrainMass's custom state-carrying rollout path.
- BrainState `HiddenState` stores ring activation, ring rate, ring adaptation,
  grid activation, grid rate, and grid adaptation. One
  `brainstate.transform.jit` wraps one `brainstate.transform.for_loop`; there is
  no Python loop over simulation time.
- BrainUnit represents integration time, theta frequency, speed, grid scale, and
  spatial position. Conversion to plain arrays occurs only for host analysis,
  plotting, and serialization.

The released environment was BrainX 2026.7.9, BrainMass 0.1.1, BrainState 0.5.2,
BrainUnit 0.5.1, and CPU JAX 0.10.2.

## Model definition

All rates and synaptic activation variables are dimensionless. Time is in ms,
position and grid scale are in cm, speed is in cm/s, angles are in radians, and
theta frequency is in Hz. `wrap` maps angles component-wise to `[-pi, pi)`.

### Direction ring

There are 72 direction units at preferred angles `theta_i` uniformly spaced on
the circle. Their rate is

```text
r_i = sigmoid(5 (u_i - 0.55)).
```

The normalized local-excitation kernel is

```text
W_ij proportional to exp[-wrap(theta_i-theta_j)^2 / (2 * 0.40^2)].
```

For theta phase `phi = 2 pi f_theta t`, define the sweep gate and normalized
speed as

```text
m(phi) = alpha_theta [1 - cos(phi)] / 2,
s_v = v / (30 cm/s).
```

The target activation and adaptation obey

```text
U_i = [2.20 + 1.10 s_v m] sum_j W_ij r_j
      - 1.10 mean_j(r_j)
      + 3.40 [1 - 0.45 s_v m]
        exp[-wrap(theta_i-h)^2 / (2 * 0.34^2)]
      - 2.00 alpha_adapt a_i,

tau_u du_i/dt = -u_i + U_i,       tau_u = 12 ms,
tau_a da_i/dt = -a_i + r_i,       tau_a = 60 ms.
```

The first term is local recurrent excitation, the population mean is global
inhibition, the Gaussian term anchors the ring to measured head direction `h`,
and `a_i` is slow firing-rate adaptation. Theta strengthens recurrence while
weakening anchoring. At excessive adaptation (relative strength 1.5), the
period-two mode collapses rather than growing without bound.

### Direction-by-grid transformation

Grid modules have scales 38, 55, and 78 cm. Each is an 18 by 18 torus with phase
coordinate `q_k`. The oblique spatial-to-phase basis is

```text
B = [[1, 0], [1/2, sqrt(3)/2]].
```

For physical position `x`, module `ell` has animal phase

```text
p_ell = wrap(2 pi B x / lambda_ell).
```

Every direction cell contributes a toroidal field shifted along its preferred
physical direction `e_i = [cos(theta_i), sin(theta_i)]`:

```text
delta = 0.19 s_v m,
c_ell_i = wrap(p_ell + 2 pi delta B e_i),
C_ell_k = sum_i w_i exp[-||wrap(q_k-c_ell_i)||^2 / (2 * 0.46^2)],
w_i = r_i^5 / sum_j r_j^5.
```

This is the effective conjunctive population. It maps the distributed ring
state into spatially shifted grid input before either population is decoded.
The decoded position is never fed back as a commanded path.

### Grid sheets

Grid rates use divisive global inhibition:

```text
g_ell_k = relu(z_ell_k)^2 /
          [1 + 0.10 sum_j relu(z_ell_j)^2].
```

The normalized toroidal recurrent kernel has Gaussian width 0.52 rad. Let
`A_ell_k` be the same 0.46-rad toroidal field centered on `p_ell`. Grid dynamics
are

```text
Z_ell_k = 2.25 sum_j K_kj g_ell_j
          - 0.85 mean_j(g_ell_j)
          + 1.90 (1 - 0.72 m) A_ell_k
          + alpha_coupling (1.25 + 2.20 m) C_ell_k
          - 0.18 b_ell_k,

tau_z dz_ell_k/dt = -z_ell_k + Z_ell_k,  tau_z = 10 ms,
tau_b db_ell_k/dt = -b_ell_k + g_ell_k,  tau_b = 240 ms.
```

Thus every sheet has local recurrent excitation, divisive and subtractive
global inhibition, sensory position anchoring, conjunctive direction-dependent
shift input, and weak adaptation.

### Integration and initialization

`dt = 2 ms`, `f_theta = 10 Hz`, and each theta cycle contains 50 integration
steps. For each state `y` with time constant `tau`, the nonlinear target is held
over one step and the leak is advanced as

```text
y(t+dt) = exp(-dt/tau) y(t) + [1-exp(-dt/tau)] target(t).
```

Ring and grid activations start from independent Gaussian perturbations with
standard deviation 0.002; adaptation and monitored rates start at zero. The
BrainState seed is 1405 for every matched condition. The shuffle seed is 8128.

## Navigation protocols

- Straight: 6 s at 30 cm/s and heading 0.
- Speed change: 9 s at 15, 30, and 45 cm/s for 3 s each, heading 0.
- Turn: 10 s at 24 cm/s; straight for 2 s, turn at 42 deg/s for 6 s, then
  continue for 2 s. Position is the cumulative velocity at `dt = 2 ms`.
- Controls: the straight run with only adaptation, theta, or conjunctive
  coupling set to zero.
- Adaptation sweep: relative strengths 0, 0.5, 1.0, and 1.5.

## Decoding and analysis definitions

The first 1 s is excluded. Ring direction is the population-vector angle
`arg(sum_i r_i exp(i theta_i))`. Grid phase is decoded by separate circular
means of each torus axis. Physical displacement from the animal is

```text
d_ell = B^-1 wrap(qhat_ell - p_ell) lambda_ell / (2 pi).
```

Cycle metrics use theta phase `pi`, 50 ms after cycle onset. Ring sweep angle is
decoded ring direction minus measured heading. Grid sweep angle is
`atan2(d_y,d_x)` minus heading; length is `||d||`.

An adjacent pair counts as alternating only when both ring angles have magnitude
at least 5 deg and their signs differ. The alternation score divides the count
by all adjacent pairs, so weak or missing sweeps reduce reliability. Grid
alternation uses the same rule. The shuffle null randomly permutes cycle order
4,000 times while preserving the measured angle distribution. The reported
one-sided p-value is `(1 + number(null >= observed)) / 4001`.

Ring-grid alignment is `cos(grid direction - ring direction)` at phase `pi`.
The phase-resolved value averages the same cosine over the active theta half
(`cos(phi) < 0`) only when grid displacement is at least 1 cm; direction is
undefined at zero displacement.

Single-cell theta skipping uses cycle-integrated direction-cell rate. With
unnormalized lag products `L1` and `L2` of mean-centered cycle rate, the index is
`(L2-L1)/(|L2|+|L1|)`. Directional tuning width is
`sqrt(-2 log R)`, where `R` is the rate-weighted resultant of measured heading.
During the 2-8 s turning interval, rate-weighted preferred theta phase is
computed in seven relative-heading bins spanning +/-70 deg; cells with at least
four populated bins receive an unwrapped linear phase slope and correlation.

## Quantitative findings

| Result | Value |
|---|---:|
| Straight ring alternation | 0.959 |
| Shuffled mean; 95% interval | 0.469; [0.327, 0.612] |
| One-sided shuffle p-value | 0.00025 |
| Mean absolute straight sweep angle | 22.93 deg |
| Grid alternation, 38/55/78 cm | 0.939 / 0.939 / 0.939 |
| Grid length, 38/55/78 cm | 4.63 / 6.86 / 9.89 cm |
| Grid-scale/length Spearman rho | 1.0 (n=3) |
| Turn ring alternation | 0.966 |
| Turn phase-resolved alignment | 0.992 / 0.991 / 0.991 |
| Speed 15/30/45 cm/s, ring angle | 0.00 / 13.60 / 29.35 deg |
| Speed 15/30/45 cm/s, reliability | 0.000 / 0.586 / 1.000 |
| Median single-cell skipping index | 0.182 |
| Tuning-width/skipping Spearman rho | 0.978 |
| Cells with turn phase code | 54 of 72 |
| Median absolute turn phase correlation | 0.820 |

Sweep length increased monotonically with both speed and grid scale. Adaptation
had a bounded operating regime: no adaptation produced no sweep, 0.5 produced
partial alternation, 1.0 produced reliable alternation, and 1.5 abolished the
ring angle while leaving a heading-directed conjunctive displacement. This is a
non-monotonic dynamical regime, not evidence that stronger adaptation always
improves sweeps.

The single-cell associations are consequences of this deterministic
phenomenological model. They are not independent biological validation. The
shuffle test addresses cycle order only, the scale correlation has three
modules, and one fixed initialization seed does not establish robustness to
parameter or noise ensembles.

## Run and outputs

Run with the released BrainX environment:

```bash
MPLCONFIGDIR=/tmp/matplotlib-theta-direction-grid \
  /home/yixinliu/anaconda3/envs/braincell-released/bin/python theta_direction_grid.py
```

Outputs under `results/`:

- `population_and_alignment.png`: population activity, phase-matched sweeps,
  shuffle control, alignment, and grid-scale relationship.
- `protocols_and_mechanisms.png`: turn trajectory, speed/adaptation effects,
  mechanism controls, and single-cell analysis.
- `trajectory_with_10_sweep_vectors.png`: focused 2D rat trajectory with exactly
  10 decoded vectors originating at selected positions.
- `summary.json`: all reported scalar and grouped summaries.
- `baseline_cycle_metrics.csv`: one row per analyzed straight-run cycle.
- `theta_direction_grid_evidence.npz`: time-resolved and cycle-resolved evidence.
