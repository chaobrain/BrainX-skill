# Online Working Memory

## Prompt

Train a recurrent spiking network online to remember a brief cue through a silent delay and report whether the next cue matches it.

## Expected BrainX Packages

- `brainpy-state`: define the recurrent spiking network, cue inputs, and match-or-nonmatch readout.
- `brainevent`: implement efficient spike-driven recurrent communication.
- `braintrace`: train the temporal task online with eligibility traces instead of sequence-length-dependent backpropagation through time.
- `brainstate`: manage recurrent and optimizer state, use `for_loop` for streaming updates, and batch independent examples with `vmap`.
- `brainunit`: enforce consistent neural and temporal quantities throughout the model.
