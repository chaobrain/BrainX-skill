# Prior Bias under Ambiguous Evidence

## Prompt

Does a small prior bias change a decision mainly when the evidence is ambiguous? Build a noisy two-choice brain circuit, compare unbiased and slightly biased decisions from weak to strong evidence, show several choices unfolding, and plot the resulting choice probabilities and measured simulation speed.

## Expected BrainX Packages

- `brainpy-state`: define the noisy competing decision populations, recurrent excitation, mutual inhibition, and prior input.
- `brainstate`: use `for_loop` for evidence accumulation, `vmap` for parallel bias and evidence conditions, and `jit` for compiled execution whose simulation speed can be measured.
- `brainunit`: keep evidence input, prior input, membrane dynamics, synaptic coupling, and decision times physically consistent.
markdown