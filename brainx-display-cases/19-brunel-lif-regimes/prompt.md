# Brunel-lif-regimes

## Prompt
Simulate a current-based LIF network: NE=10,000, NI=2,500; fixed CE/CI/Cext=1000/250/1000; tau_m=20 ms, t_ref=2 ms,threshold/reset=20/10 m V; delta jumps JE=0.1 mV and JI=-gJE with 1.5 ms delay; independent Poisson drive nu_ext=eta*theta/(JE*CE*tau_m).Run (g,eta)=(3,2),(6,4),(5,2),(4.5,0.9) over a fixed seed set. Render and save a 4-condition figure: for each condition, place instantaneous global rate (0.1 ms bins; mean dashed) above a raster for a fixed random 50-neuron sample. Also save E/I rates, ISI CV and global-rate spectrum; report dominant frequency. Verify the respective regimes: synchronous regular, fast synchronous irregular,asynchronous irregular, and slow synchronous irregular. Do not stop at arrays: write PNG files and list paths.

## Expected BrainX Packages

- `brainpy-state`: build the LIF populations, delayed synapses, and Poisson input.
- `brainevent`: implement sparse event-driven recurrent communication.
- `brainunit`: enforce consistent voltage and time units.
- `brainstate`: manage and compile the network simulation.