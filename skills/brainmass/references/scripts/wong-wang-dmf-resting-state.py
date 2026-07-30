"""Reference script mirrored from:
https://brainx.chaobrain.com/brainmass/gallery/model_zoo/wong_wang_exc_inh.html

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
# # Wong-Wang Excitatory-Inhibitory (Dynamic Mean Field)
#
# The **two-population (excitatory-inhibitory) reduced Wong-Wang model**, often called the *dynamic mean field* (DMF), is the resting-state workhorse for whole-brain modeling. Each region has an excitatory NMDA gating variable $S_E$ and an inhibitory GABA gating variable $S_I$; a feedback-inhibition weight $J_i$ regulates the local excitation-inhibition balance so that the excitatory population settles near a biologically plausible ~3 Hz resting rate. It is the node model behind much resting-state BOLD / functional-connectivity work.
#
# *(Distinct from the reduced two-choice* `WongWangStep` *decision model.)*
#
# **Reference:** Deco, Ponce-Alvarez, Hagmann, Romani, Mantini & Corbetta (2014), *How local excitation-inhibition ratio impacts the whole brain dynamics*, Journal of Neuroscience 34(23):7886-7898.

# %% [markdown]
# ## Build the model

# %%
node = brainmass.WongWangExcInhStep(in_size=1, J_i=1.0, G=2.0)
node

# %% [markdown]
# ## Run a simulation
#
# We monitor the excitatory gating variable `S_e` and the derived excitatory firing rate via the `H_e()` observable.

# %%
sim = brainmass.Simulator(node, dt=0.1 * u.ms)
res = sim.run(3000. * u.ms,
              monitors={'S_e': 'S_e', 'rate_e': lambda m: m.H_e()})
float(u.get_magnitude(res['rate_e'])[-1, 0])  # resting excitatory rate (Hz)

# %% [markdown]
# ## Visualize
#
# The gating variable settles to a steady value and the excitatory rate stabilizes near the ~3 Hz resting fixed point.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
brainmass.viz.plot_timeseries(res['S_e'], ts=res['ts'], ax=axes[0])
axes[0].set_title('Excitatory gating S_e')
brainmass.viz.plot_timeseries(res['rate_e'], ts=res['ts'], ax=axes[1])
axes[1].set_title('Excitatory rate H_e (Hz)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Try it: vary the feedback inhibition `J_i`
#
# The feedback-inhibition weight `J_i` sets the E-I balance. Stronger inhibition clamps the excitatory rate lower; weaker inhibition lets it climb. We report the settled excitatory rate.

# %%
for J_i in [0.5, 1.0, 1.5]:
    m = brainmass.WongWangExcInhStep(in_size=1, J_i=J_i, G=2.0)
    r = brainmass.Simulator(m, dt=0.1 * u.ms).run(
        3000. * u.ms, monitors={'rate_e': lambda m: m.H_e()})
    rate = float(u.get_magnitude(r['rate_e'])[-1, 0])
    print(f'J_i = {J_i:.1f}  ->  resting excitatory rate = {rate:.2f} Hz')
