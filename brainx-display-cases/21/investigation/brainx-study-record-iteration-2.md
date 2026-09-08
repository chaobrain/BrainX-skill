# Iteration 2 restudy

## Reviewer finding map

BX-001 is owned by BrainCell's current-clamp workflow and BrainTools input construction, with BrainState for time-major iteration and BrainUnit for current-density arrays. Network/channel equations and scientific conclusions were supported and are unchanged.

## Sources restudied

- `brainx-general-guard/SKILL.md`, first: prefer the owning input API and preserve State/units.
- Complete `braincell/SKILL.md`: use `braintools.input.Constant` for timed baseline and stimulation sections.
- Complete BrainCell `references/braintools/input-current.md`: construct/generate under rollout dt, then slice time and current together with `for_loop`; do not reconstruct sections with time predicates in step.
- Complete BrainCell `references/scripts/hh_neuron_basics.py`: keep cell construction, initialization and update ownership unchanged; apply the root's current input API where the older script uses a constant scalar.
- Complete `brainstate/SKILL.md`: registered State, environment dt and multi-input for_loop.
- Complete `brainunit/SKILL.md` and `references/array-mechanics.md`: construct quantities at input creation, join with `u.math.stack`, and convert only when saving raw numeric evidence.

## Resulting design

Create the reduced-drive protocol (0 for 100 ms, 120 for 300 ms, 20 for 400 ms) and the shared sustained-drive protocol (0 for 100 ms, 120 for 700 ms) with `braintools.input.Constant`, all in uA/cm2. Stack their generated samples into three condition columns. Pass each current vector directly to the cell. Retain only the channel ParamState time intervention inside step.

Save to a new `runs/acute_rescue_native` directory. Compare all samples against the original declared waveform and all recorded trajectories against `runs/acute_rescue/raw.npz`; require exact equality. Keep source snapshots and append the new artifacts to the manifest. No other experiment needs rerunning because its source and inputs do not change.
