"""
Reference script mirrored from:
https://brainx.chaobrain.com/brainmass/tutorials/07_gradient_free_fitting.html

Purpose:
Carry the complete gradient-free counterpart to the BrainMass skill's inline
gradient-fitting example. The objective is written once and fitted with
Nevergrad differential evolution and SciPy Nelder-Mead.

Use this when:
- The objective is non-differentiable, discrete, jagged, or black-box.
- A usable gradient is unavailable or unreliable.
- The task needs a complete `Fitter` example for `nevergrad` or derivative-free
  `scipy` rather than the canonical `grad` path in the skill body.

Do not use this when:
- The objective is differentiable; use the skill body's gradient workflow.
- The parameters do not have finite transform-derived bounds and no explicit
  `search_space` has been supplied.

Operational boundary:
Gradient-free `Fitter` backends keep the same objective, but they take an
options dict (or method-name string) instead of a `braintools.optim` instance.
They also search a bounded box. Here `SigmoidT(0.05, 3.0)` supplies that box.
"""

import brainmass
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
from brainstate.nn import Param, SigmoidT


brainstate.environ.set(dt=0.1 * u.ms)
brainstate.random.seed(0)


def make_hopf(a, trainable=False):
    """Build the tutorial Hopf node, optionally with bounded fitted `a`."""
    a_param = Param(a, t=SigmoidT(0.05, 3.0), fit=True) if trainable else a
    return brainmass.HopfStep(
        in_size=1,
        a=a_param,
        w=0.3,
        init_x=braintools.init.Constant(0.5),
        init_y=braintools.init.Constant(0.0),
    )


def settled_amplitude(node):
    """Measure the settled Hopf limit-cycle amplitude."""
    res = brainmass.Simulator(node, dt=0.1 * u.ms).run(
        300.0 * u.ms,
        monitors=['x'],
        transient=150.0 * u.ms,
    )
    x = u.get_magnitude(res['x'])
    return jnp.sqrt(jnp.mean(x ** 2)) * jnp.sqrt(2.0)


# Generate the same known target used by the official gradient-fitting tutorial.
target_amplitude = float(settled_amplitude(make_hopf(1.5)))


def loss_fn(model):
    """Use the same objective for every optimizer backend."""
    amp = settled_amplitude(model)
    return (amp - target_amplitude) ** 2, amp


print(f"target amplitude (true a* = 1.5): {target_amplitude:.4f}")


# Nevergrad: `n_steps` is the number of generations. With `n_sample=6`,
# 12 generations require roughly 72 forward simulations.
node = make_hopf(0.1, trainable=True)
ng_result = brainmass.Fitter(
    node,
    {'method': 'DE', 'n_sample': 6},
    loss_fn=loss_fn,
    backend='nevergrad',
).fit(n_steps=12)

print(
    f"nevergrad: fitted a = {float(ng_result.best_params['a']):.4f}"
    "   (true a* = 1.5)"
)
print(f"           best loss = {ng_result.best_loss:.2e}")
print(f"           simulations = {len(ng_result.history)}")


# SciPy: for this backend `n_steps` is the number of random restarts.
node = make_hopf(0.1, trainable=True)
sp_result = brainmass.Fitter(
    node,
    {'method': 'Nelder-Mead'},
    loss_fn=loss_fn,
    backend='scipy',
).fit(n_steps=3)

print(
    f"scipy: fitted a = {float(sp_result.best_params['a']):.4f}"
    "   (true a* = 1.5)"
)
print(f"       best loss = {sp_result.best_loss:.2e}")
