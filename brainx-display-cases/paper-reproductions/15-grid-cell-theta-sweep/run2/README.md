# Alternating theta sweeps in a direction-grid network

This display case implements a phenomenological firing-rate model in which a
theta-modulated head-direction ring drives spatially shifted input to three
two-dimensional grid attractors. The model produces left-right-alternating
sweeps during straight running and turning; it does not feed a decoded
trajectory back into either population.

## Result

After a 1 s warmup, the straight-run ring alternation score was **0.959**. The
mean score after 4,000 random permutations of cycle order was 0.469, with a 95%
interval of [0.327, 0.612] and one-sided permutation `p = 0.00025`. The three
grid modules alternated at 0.939 and remained aligned with the direction ring:
their phase-pi mean alignment cosines were 0.991, 0.991, and 0.990.

Removing either firing-rate adaptation or theta modulation abolished ring and
grid alternation. Removing the conjunctive ring-grid projection preserved ring
alternation (0.959) but abolished grid alternation and reduced mean decoded grid
displacements below 0.5 cm. These matched controls locate alternation in the
adaptation-theta interaction and its spatial transmission in the conjunctive
projection.

Turning preserved ring alternation (0.966), grid alternation (0.955), and
ring-grid alignment (mean cosine 0.990-0.991). Spatial sweep length increased
with speed and grid scale. At 15, 30, and 45 cm/s, the 38 cm module's mean
lengths were 2.37, 4.73, and 6.80 cm. Adaptation had a bounded regime: relative
strengths 0, 0.5, 1.0, and 1.5 produced alternation scores 0.000, 0.388, 0.959,
and 0.000.

## BrainX execution

- BrainMass owns the aggregate neural-population scale and supplies its
  visualization interface. The direction and grid attractors use a custom
  `DirectionGridStep` because the public model catalogue has no direction-ring
  or toroidal-grid step.
- BrainState `HiddenState` stores activation, firing rate, and adaptation for
  both populations. A single `brainstate.transform.jit` wraps each complete
  `brainstate.transform.for_loop`; there is no Python loop over simulation
  timesteps. The adaptation scan maps complete independent attractor rollouts
  with `brainstate.transform.vmap`.
- BrainUnit represents integration time, theta frequency, speed, heading angle,
  grid scale, position, and protocol duration. Conversion to raw cm, cm/s, and
  radians occurs at the explicitly normalized model boundary; conversion to
  NumPy occurs for host analysis, plotting, and serialization.

The model seed is 1405. The shuffle seed is 8128.

## Model definition

Rates and activation variables are dimensionless. Time constants are in ms,
positions and grid scales in cm, speeds in cm/s, headings in radians, and theta
frequency in Hz. `wrap(x)` maps each angle to `[-pi, pi)`.

### Head-direction ring

The ring contains 72 units with preferred directions `theta_i` uniformly spaced
on the circle. Its rate transfer is

```text
r_i = sigmoid[5 (u_i - 0.55)].
```

Local recurrent excitation uses the normalized circular kernel

```text
W_ij proportional to exp{-wrap(theta_i - theta_j)^2 / (2 * 0.40^2)}.
```

For theta phase `phi = 2 pi f_theta t`, define

```text
m(phi) = alpha_theta [1 - cos(phi)] / 2,
s_v = v / (30 cm/s).
```

The nonlinear activation target is

```text
U_i = [2.20 + 1.10 s_v m] sum_j W_ij r_j
      - 1.10 mean_j(r_j)
      + 3.40 [1 - 0.45 s_v m]
        exp{-wrap(theta_i - h)^2 / (2 * 0.34^2)}
      - 2.00 alpha_adapt a_i.
```

The first term is local excitation, the population mean is subtractive global
inhibition, the Gaussian is sensory anchoring to measured head direction `h`,
and `a_i` is slow firing-rate adaptation:

```text
tau_u du_i/dt = -u_i + U_i,       tau_u = 12 ms,
tau_a da_i/dt = -a_i + r_i,       tau_a = 60 ms.
```

Theta strengthens recurrent propagation while weakening anchoring. Adaptation
suppresses the previously active side of the ring, providing the one-cycle
memory needed for a period-two left-right state.

### Direction-by-grid projection

Grid modules have spatial scales 38, 55, and 78 cm. Each contains an 18 by 18
phase torus. The oblique spatial-to-phase basis is

```text
B = [[1, 0], [1/2, sqrt(3)/2]].
```

At animal position `x`, module `ell` has anchored phase

```text
p_ell = wrap(2 pi B x / lambda_ell).
```

Every direction unit contributes a toroidal field shifted along its own
preferred physical direction `e_i = [cos(theta_i), sin(theta_i)]`:

```text
delta = 0.19 s_v m,
c_ell_i = wrap[p_ell + 2 pi delta B e_i],
w_i = r_i^5 / sum_j r_j^5,
C_ell_k = sum_i w_i
          exp{-||wrap(q_k - c_ell_i)||^2 / (2 * 0.46^2)}.
```

`C_ell_k` is the effective conjunctive population. It transforms the
distributed direction-ring state into a spatially shifted input field before
decoding. No decoded ring angle or decoded position is used to command a path.

### Grid attractors

Grid rates use divisive global inhibition:

```text
g_ell_k = relu(z_ell_k)^2 /
          [1 + 0.10 sum_j relu(z_ell_j)^2].
```

The normalized toroidal recurrent Gaussian has width 0.52 rad. Let `A_ell_k`
be a 0.46-rad toroidal Gaussian centered on the animal phase `p_ell`. Then

```text
Z_ell_k = 2.25 sum_j K_kj g_ell_j
          - 0.85 mean_j(g_ell_j)
          + 1.90 (1 - 0.72 m) A_ell_k
          + alpha_coupling (1.25 + 2.20 m) C_ell_k
          - 0.18 b_ell_k,

tau_z dz_ell_k/dt = -z_ell_k + Z_ell_k,  tau_z = 10 ms,
tau_b db_ell_k/dt = -b_ell_k + g_ell_k,  tau_b = 240 ms.
```

Thus each grid sheet has local recurrent excitation, subtractive and divisive
global inhibition, direct position anchoring, direction-dependent shifted
input, and weak adaptation.

### Integration and initialization

`dt = 2 ms` and `f_theta = 10 Hz`, giving 50 steps per theta cycle. The leak
term for each state `y` is integrated exactly while its nonlinear target is
held fixed over one step:

```text
y(t + dt) = exp(-dt/tau) y(t) + [1 - exp(-dt/tau)] target(t).
```

Ring and grid activation start from independent Gaussian perturbations with
standard deviation 0.002. Rates and adaptation start at zero. Every matched
condition starts from the same seeded perturbations.

## Navigation protocols

- Straight: 6 s at 30 cm/s with heading 0.
- Speed change: 9 s at 15, 30, and 45 cm/s for 3 s each with heading 0.
- Turn: 10 s at 24 cm/s; straight for 2 s, turn at 42 deg/s for 6 s, then
  continue for 2 s.
- Controls: straight runs with adaptation, theta modulation, or conjunctive
  coupling individually set to zero.
- Adaptation scan: relative strengths 0, 0.5, 1.0, and 1.5, mapped through one
  BrainState `vmap` call with identical initial conditions.

Positions are the cumulative integral of commanded velocity at `dt = 2 ms`.

## Decoding and analysis definitions

The first 1 s of each run is excluded. Ring direction is the population-vector
angle

```text
direction_hat = arg sum_i r_i exp(i theta_i).
```

Each grid phase coordinate is decoded by a separate circular population mean.
The physical displacement of module `ell` from the animal is

```text
d_ell = B^-1 wrap(qhat_ell - p_ell) lambda_ell / (2 pi).
```

Cycle metrics sample theta phase `pi`, 50 ms after cycle onset. Ring sweep
angle is `direction_hat - heading`. Grid sweep angle is `atan2(d_y, d_x) -
heading`, and sweep length is `||d||`.

An adjacent pair counts as alternating only if both ring angles have magnitude
at least 5 deg and their signs differ. The alternation score divides the number
of such pairs by all adjacent pairs, so absent or weak sweeps reduce the score.
Grid alternation uses the same rule. The shuffle null randomly permutes the
observed cycle order 4,000 times while preserving the angle distribution. The
one-sided p-value is

```text
(1 + number of shuffled scores >= observed score) / 4001.
```

Ring-grid alignment is `cos(grid direction - ring direction)`. The phase-pi
summary averages this value over analyzed cycles. The phase-resolved summary
uses the active half of theta, `cos(phi) < 0`, and excludes grid displacements
shorter than 1 cm, where direction is poorly defined.

Speed summaries omit the first 0.5 s after each speed transition. The 10 arrows
in the trajectory figure are the first grid module's decoded displacement
vectors at 10 evenly spaced analyzed cycles within the 2-8 s turning interval.
Each arrow originates at the animal's position at that cycle; red and blue
identify left and right ring sweeps. Positive angular offset is counterclockwise
from heading and is labeled left.

## Interpretation and limits

The simulations establish a mechanism within this deterministic,
phenomenological model. They do not fit either cited dataset. The cycle-order
shuffle tests temporal organization, not uncertainty over model seeds or
parameters. The adaptation scan demonstrates a non-monotonic regime rather
than a general claim that more adaptation improves alternation. Alignment in
no-theta and no-coupling controls is not interpreted when displacement is below
the 1 cm direction threshold.

The model is motivated by Vollan et al. (Nature, 2025) and Ji et al. (Current
Biology, 2025).

## Run and outputs

```bash
python theta_sweep_network.py
```

Use `--quick` to run only the straight baseline.

Outputs under `results/`:

- `population_dynamics.png`: ring activity, within-cycle decoding, grid bumps,
  cycle alternation, shuffle null, and ring-grid alignment.
- `navigation_and_controls.png`: turn trajectory, speed effect, adaptation
  scan, and matched mechanism controls.
- `trajectory_with_10_sweep_vectors.png`: focused 2D trajectory with exactly 10
  decoded theta-sweep vectors.
- `summary.json`: scalar and grouped quantitative results.
- `straight_cycle_metrics.csv`: one row per analyzed straight-run cycle.
- `theta_sweep_evidence.npz`: time-resolved and cycle-resolved evidence arrays.
