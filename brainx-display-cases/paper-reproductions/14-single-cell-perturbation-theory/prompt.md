# Reproduce the V1 neuronal perturbome model

Reproduce the central computational result of Sadeh and Clopath (2020) in BrainX, using the authors' ModelDB implementation (accession 262045) as the source of equations, default parameters, receptive fields, stimuli, connectivity regimes, perturbation protocol, and analysis.

Determine which combinations of excitatory and inhibitory connection strength and functional specificity produce feature-specific suppression between excitatory neurons. Include the inhibition-dominant and broadly tuned inhibition controls used to distinguish the proposed mechanism. Report the original parameter values, any changes required by the BrainX implementation, baseline stability and activity, and the parameter region rather than only one successful setting.

Compare the reproduced influence-versus-response-similarity curves with the paper's qualitative result. Preserve negative results and do not tune against the final curve without reporting the search procedure and evaluating the selected regime on held-out seeds or nearby parameter values. If the original source or a required definition is unavailable, stop short of calling the result a paper reproduction and state exactly what remains unverified.

References:

- Sadeh, S. & Clopath, C. (2020), PNAS 117:26966-26976. https://doi.org/10.1073/pnas.2004568117
- ModelDB 262045. https://github.com/ModelDBRepository/262045
