"""Reference script mirrored from:
https://brainx.chaobrain.com/brainmass/gallery/model_zoo/kuramoto.html

Converted from the official notebook at upstream commit
293eaeb8948474c645208b68c531b7cb5482d4fb. Notebook outputs are omitted and
``%matplotlib inline`` remains commented; code and Markdown cells are preserved.
"""

# %% tags=["remove-cell"]
# %matplotlib inline
import brainmass
import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
brainstate.random.seed(0)
brainstate.environ.set(dt=0.1 * u.ms)

# %% [markdown]
# # Kuramoto Phase Oscillators
#
# The **Kuramoto model** is the canonical model of synchronization in a population of coupled phase oscillators. Each oscillator $i$ has a natural frequency $\omega_i$ and is pulled toward its neighbours' phases:
#
# $$\dot\theta_i = \omega_i + \frac{K}{N}\sum_j W_{ij}\sin(\theta_j - \theta_i).$$
#
# Below a critical coupling $K_c$ the oscillators drift incoherently; above it they synchronize. Global coherence is measured by the Kuramoto order parameter $R = |\frac{1}{N}\sum_j e^{i\theta_j}|$, which rises from ~0 (incoherent) to ~1 (fully synchronized).
#
# **Reference:** Kuramoto (1975), *Self-entrainment of a population of coupled nonlinear oscillators*, in International Symposium on Mathematical Problems in Theoretical Physics.

# %% [markdown]
# ## Build the model
#
# We build a population of `N = 50` all-to-all coupled oscillators with heterogeneous natural frequencies.

# %%
N = 50
omega = np.asarray(brainstate.random.normal(1.0, 0.3, N))
node = brainmass.KuramotoNetwork(in_size=N, omega=omega, K=2.0,
                                 conn=np.ones((N, N)))
node

# %% [markdown]
# ## Run a simulation

# %%
sim = brainmass.Simulator(node, dt=0.1 * u.ms)
res = sim.run(150. * u.ms, monitors=['theta'])
theta = u.get_magnitude(res['theta'])
R = np.abs(np.mean(np.exp(1j * theta), axis=1))
R[0], R[-1]

# %% [markdown]
# ## Visualize
#
# Left: the order parameter `R` climbing toward synchronization. Right: individual phases (mod $2\pi$) bunching together over time.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(u.get_magnitude(res['ts']), R)
axes[0].set_xlabel('time (ms)'); axes[0].set_ylabel('order parameter R')
axes[0].set_ylim(0, 1); axes[0].set_title('Synchronization (K = 2.0)')
axes[1].plot(u.get_magnitude(res['ts']), np.mod(theta[:, ::5], 2 * np.pi),
             lw=0.6, alpha=0.7)
axes[1].set_xlabel('time (ms)'); axes[1].set_ylabel('phase (rad)')
axes[1].set_title('Individual phases')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Try it: vary the coupling strength `K`
#
# Sweep `K` across the synchronization transition: weak coupling stays incoherent (`R` low), strong coupling locks the population (`R` -> 1).

# %%
for K in [0.0, 1.0, 3.0]:
    m = brainmass.KuramotoNetwork(in_size=N, omega=omega, K=K, conn=np.ones((N, N)))
    r = brainmass.Simulator(m, dt=0.1 * u.ms).run(200. * u.ms, monitors=['theta'])
    th = u.get_magnitude(r['theta'])
    R_final = float(np.abs(np.mean(np.exp(1j * th[-50:]), axis=1)).mean())
    print(f'K = {K:.1f}  ->  steady order parameter R = {R_final:.3f}')
