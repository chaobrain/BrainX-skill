# Brain Dynamics Delay Protocol Blueprint

## Purpose

Explain BrainState Brain Dynamics delay APIs and delay-buffer behavior after `brainstate` or the BrainState dynamics parent routes here.

Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/02_synaptic_delays.html

## Used by

- `references/brainstate-dynamics/dynamics-and-integration.md`

This nested child is never a first-hop skill or index route.

## Open when

- The user asks about `output_delay(...)` or `prefetch_delay(...)`.
- The user needs synaptic, axonal, feedback, heterogeneous, step-based, or time-based delays.
- The user asks about delay buffers, interpolation, or delayed state access.

## Should eventually cover

- `brainstate.nn.Delay`.
- `brainstate.nn.DelayAccess`.
- `brainstate.nn.StateWithDelay`.
- Rotation versus concatenation buffers.
- Step-based retrieval.
- Time-based retrieval.
- Interpolation.
- Heterogeneous delays.
- Synaptic delays.
- Axonal delays.
- Feedback delays.
- Automatic synchronization after `update()`.

## Common mistakes to document

- Treating output delay and state/prefetch delay as the same mechanism.
- Forgetting that delay buffers synchronize after update-cycle hooks.
- Using one fixed delay assumption when heterogeneous delays are needed.
- Confusing simulation time units with delay-buffer step indices.

## Placeholder examples

- Minimal `output_delay(...)`.
- Minimal `prefetch_delay(...)`.
- Step-based retrieval.
- Time-based retrieval.
- Heterogeneous synaptic delay.
