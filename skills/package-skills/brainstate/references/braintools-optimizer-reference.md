# Optimizers for BrainState training

Use this reference after choosing the trainable State collection and loss. The optimizer owns the update and its auxiliary State while the model owns the parameter values.

| Need | Use |
|---|---|
| A simple gradient-step baseline | Use `braintools.optim.SGD(lr=...)`. Register the `ParamState` collection once, then call `update(grads)` with the gradient PyTree for the same collection. |
| Adaptive minibatch training | Use `braintools.optim.Adam(lr=...)`. Its moments are separate from model State. Include optimizer State when saving or restoring an exact training continuation. |
| A scheduled learning rate | Choose a scheduler from the optimization API and advance it at the documented step or epoch boundary. Check the rate sequence in a short run before training. |
| An existing Optax transformation | Use the documented BrainTools bridge and keep the same registered gradient and parameter PyTrees. |
| Standalone SciPy or Nevergrad fitting | Use the active model package's fitting workflow. These wrappers own a search over explicit candidates and use a different lifecycle from `register_trainable_weights()` and `update()`. |

Open `brainstate` → `SKILL.md` for the compiled training step. Register after constructing the model and before tracing the step. Preserve both the State paths and the nested value structure. Keep optimizer moments across training steps and start with fresh optimizer State for independent candidates.

- [BrainTools optimization API](https://brainx.chaobrain.com/braintools/apis/optim.html)
- [BrainState training tutorial](https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html)
