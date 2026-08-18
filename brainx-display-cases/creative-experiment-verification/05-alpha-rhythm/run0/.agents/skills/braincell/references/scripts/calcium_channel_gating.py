"""
Reference script mirrored from:
https://brainx.chaobrain.com/braincell/examples/calcium_channel_gating.html

Purpose:
Channel gating diagnostic script. Use when the task asks for voltage-dependent activation/inactivation curves, low-threshold vs high-threshold calcium channel comparison, steady-state gating, or direct channel methods such as `f_p_inf` / `f_q_inf`.

Use this when:
- The task asks for calcium channel gating curves or steady-state activation/inactivation.
- The agent needs direct channel-method inspection rather than membrane-potential simulation.

Do not use this when:
- The task needs the default current-clamp simulation pattern for membrane traces.
"""

import braintools
import brainunit as u
import matplotlib.pyplot as plt
import braincell


# Channel-level diagnostic: instantiate channel classes directly, no cell update loop.
# Create Low-Threshold T-Type Calcium Channel (ICaT)
cat = braincell.channel.CaT_HP1992(1)

# Create High-Threshold Calcium Channel (ICaHT)
caht = braincell.channel.CaHT_HM1992(1)

# Generate Voltage Sequence
vs = u.math.arange(-100 * u.mV, 0 * u.mV, 0.1 * u.mV)

# These channels' steady-state gates depend only on voltage, but the API
# requires a calcium IonInfo argument; supply a resting calcium state.
ca = braincell.IonInfo(Ci=5e-5 * u.mM, Co=2. * u.mM, E=120. * u.mV, valence=2)

# Create Figure and Subplot Layout
fig, gs = braintools.visualize.get_figure(1, 2, 3., 4.5)

# Compute steady-state values of activation and inactivation gates for low-threshold channel
q_inf = cat.f_q_inf(vs, ca)
p_inf = cat.f_p_inf(vs, ca)

# Add subplot 1
fig.add_subplot(gs[0, 0])
plt.plot(vs / u.mV, q_inf, label='q_inf (activation gate)')  # Convert x-axis to mV for readability
plt.plot(vs / u.mV, p_inf, label='p_inf (inactivation gate)')
plt.legend()  # Show legend
plt.fill_between([-80, -60], 1., alpha=0.2)  # Highlight typical activation range for low-threshold channel
plt.title('Low-Threshold Calcium Channel (ICaT)')
plt.xlabel('Membrane Potential (mV)')

# Compute q_inf and p_inf for high-threshold channel
q_inf = caht.f_q_inf(vs, ca)
p_inf = caht.f_p_inf(vs, ca)

# Add subplot 2
fig.add_subplot(gs[0, 1])
plt.plot(vs / u.mV, q_inf, label='q_inf (activation gate)')
plt.plot(vs / u.mV, p_inf, label='p_inf (inactivation gate)')
plt.fill_between([-60, -40], 1., alpha=0.2)  # Highlight typical activation range for high-threshold channel
plt.legend()
plt.xlabel('Membrane Potential (mV)')
plt.title('High-Threshold Calcium Channel (ICaHT)')

# Display figure
plt.show()
