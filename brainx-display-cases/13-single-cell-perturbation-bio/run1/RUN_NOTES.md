# Run notes

## Status

Use `run1` as the interpretable prespecified run. `run0` is retained but invalid:
its matching projection descriptors could merge ordinary and forced synaptic
state, and its paired rollouts did not restore an exact common initial state.

`run1` uses independent concrete synapse/output instances and restores a snapshot
of every model State before each baseline and perturbation branch. An exact
baseline-replay assertion passes for every connectivity regime.

## Outcome

The frozen parameter set does **not** reproduce feature-specific suppression and
high-similarity amplification. Inhibition dominance and broad inhibition have
near-zero influence throughout, as expected for the controls. The strong,
specific E/I regime also has no negative intermediate-similarity influence and
no positive influence in the highest-similarity bin. Its small positive values
in lower bins are comparable to their standard errors.

Do not interpret this negative result as refuting the mechanism. This compact
network has 100 neurons, one connectivity realization, three trials per 16
stimuli, high baseline firing near 58 Hz, and a weakly resolved six-event effect.
No parameters were adjusted after perturbation outcomes were observed.

## Protocol note

The prompt combines four 15-Hz photostimulation sweeps with six added action
potentials over approximately 250 ms. This run implements the explicit causal
dose: six additional recurrent output events at 100, 150, 200, 250, 300, and
350 ms, normalized by six delivered events.
