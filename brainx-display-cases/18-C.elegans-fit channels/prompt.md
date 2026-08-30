# C.elegans fix channel


## Prompt

Model the dynamics of SHK-1 channel based on the provided experimental data of the C.elegans's muscle cell. The data is in '/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/18-C.elegans-fit channels/Fig1C D I-V K currents.pxp'. This is the voltage clamp data of WT, slo-2 mutant and shk-1 mutant. Use wild-type minus SHK-1 mutants as the SHK-1 voltage clamp data, and fix the channel SHK-1 with HH function. The protocol involve voltage steps from 0 mV to +100 mV in 20 mV increments, a holding potential of Vh = –60 mV, and step duration of 100 ms. Draw currents-time figure to show the fitting result and show the steady-state activation curve and the activation time constant function alongside the experimental data. After that, fix the EGL-19 channel using the same method as SHK-1, the data is in '/home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/18-C.elegans-fit channels/Fig.3A I-V Ca currents.pxp'(the voltages are -20mV to +40mV, with 10mV increment).

## Expected BrainX Packages

- `braincell`: build models for the two channels.
- `brainunit`: enforce consistent units for current, voltage,etc.
- `brainstate`: manage channel state.


## Reference
Du X, Crodelle J, Barranca VJ, Li S, Shi Y, Gao S, et al. (2025) Biophysical modeling and experimental analysis of the dynamics of C. elegans body-wall muscle cells. PLoS Comput Biol 21(1): e1012318. https://doi.org/10.1371/journal.pcbi.1012318