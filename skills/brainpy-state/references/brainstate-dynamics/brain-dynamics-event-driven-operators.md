# Brain Dynamics Event-Driven Operators Blueprint

## Purpose

Route and catalog event-driven BrainState/brainpy.state operators for sparse spiking connectivity.

Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/03_event_driven_operators.html

## Used by

- `references/brainstate-dynamics/dynamics-and-integration.md`
- `references/brainstate-dynamics/brain-dynamics-snn-workflows.md`

The dynamics parent is the only first-hop selector. The SNN sibling may cross-route here only after the parent has established the dynamics tree.

## Open when

- The user has spike-train input.
- The user has sparse spiking activity.
- The user has large sparse connectivity.
- The user asks about `EventLinear`, `EventFixedProb`, `FixedNumConn`, or `EventFixedNumConn`.
- The user wants scalable SNN connectivity.

## Should eventually cover

- Why event-driven operators exploit sparse spike trains.
- Cost scaling with active inputs rather than dense population size.
- `EventLinear`.
- `EventFixedProb`.
- `FixedNumConn` / `EventFixedNumConn`.
- When to switch from dense `Linear` to `EventLinear`.
- Preserving behavior while changing connectivity implementation for scale.

## Common mistakes to document

- Using dense operators when spike activity and connectivity are sparse.
- Replacing dense operators without checking behavior equivalence.
- Treating event-driven operators as neuron dynamics rather than communication/projection operators.
- Hiding connectivity inside a local Dynamics update.

## Placeholder examples

- Dense-to-event operator route.
- Sparse spike-train input checklist.
- Event connectivity scaling check.
