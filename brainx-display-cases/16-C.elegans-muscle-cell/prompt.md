# C.elegans muscle cell


## Prompt

Build an HH model(including 7 channels: SHK-1,EGL-19,SLO-2,Kr,Na and Leak) of the C.elegans's muscle cell and use stimulation-based inference to train the model using real experimental data and estimate the parameters. Then, give the model varying stimulation currents, and see whether the model's behavior consistent with real experimental data. You can find the experimental data in /home/yixinliu/gitcode/BrainX-skill/brainx-display-cases/16-C.elegans-muscle-cell/Fig4A-D.txt, Trace #6 to Trace #9 represent giving 15pA, 20pA, 25pA and 30pA currents stimulations. Use one trace for training and the other three traces for testing.

## Expected BrainX Packages

- `braincell`: build a HH cell model with various ion channels.
- `brainunit`: enforce consistent units for current, voltage,etc.
- `brainstate`: manage the cell state, such as the cell's voltage that evolved with time.


## Reference
Du X, Crodelle J, Barranca VJ, Li S, Shi Y, Gao S, et al. (2025) Biophysical modeling and experimental analysis of the dynamics of C. elegans body-wall muscle cells. PLoS Comput Biol 21(1): e1012318. https://doi.org/10.1371/journal.pcbi.1012318