# Share nonlinear synaptic State conditionally

Share a source trace only when the represented outgoing receptor dynamics have the same kinetic parameters, initial State, and input history at the times being compared. Nonlinearity does not forbid source sharing, but it generally forbids aggregating independent source drives before applying the nonlinear dynamics.

## Compare against an edge reference

| API | Use |
|---|---|
| `brainspike.Synapse` | Define a native receptor state with its own update equation and explicit shape. |
| `SaturatingSynapse` in `scripts/advanced_native_patterns.py` | Use the verified exact constant-drive step for `dr/dt = alpha*s*(1-r)-beta*r`. |
| Source-shaped State `(N_pre,)` | Use when the outgoing trace-sharing conditions hold; apply connectivity after evaluating receptor State. |
| Edge-shaped State `(N_pre,N_post)` | Keep distinct kinetics or input histories when the source-sharing conditions fail. |

The fixture assumes nonnegative input/alpha, positive beta, and initial receptor occupancy in `[0,1]`. It solves one step analytically with physical time; it is a mechanics model, not a claim to reproduce every AMPA/NMDA implementation.

For homogeneous outgoing kinetics, the Docker check computes source traces and a separate per-edge reference, then applies the same conductance weights. Their target conductances agree exactly in the tested sequence. A second check changes outgoing edge kinetics and demonstrates that reusing the original source trace changes the target response. Another counterexample sums two source drives before saturation and produces a different result from summing independently saturated traces.

Open `scripts/advanced_native_patterns.py` for the canonical class. Instantiate a source-shaped and an edge-shaped filter, initialize their State, advance both with the same source events broadcast appropriately, and compare weighted target outputs before optimizing storage.

Do not reuse a current source trace for edge-specific arrival histories without representing those delays. Do not describe target-aligned nonlinear aggregation as an exact replacement for the original network. Keep separate source, receptor, or edge State where the model requires it; preserve all scientifically distinct parameters.

Open `references/native-projections.md` for the complete chemical-current order and linear target-alignment conditions. Open `references/native-delays.md` before adding signal history. The native implementation replaces the modeling capability without restoring retired AlignPre/AlignPost wrapper APIs.

## Source

[BrainSpike Synapse contract](https://brainx.chaobrain.com/brainspike/apis/generated/brainspike.Synapse.html). Equivalence and counterexamples are checked in the pinned BrainSpikes 0.2.0 Docker runtime.
