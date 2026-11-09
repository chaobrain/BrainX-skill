# From experimental observations to mechanism comparisons

## Purpose and boundary

Open this reference during step 0 when an observation has competing explanations or the researcher asks whether modeling can distinguish them. Use it to choose the scientific comparison before package selection and implementation. For a fixed model implementation or reproduction, use the specified design. If that design cannot answer part of the question, report the unsupported inference and propose the smallest adaptation under step 0's authorization rule; do not substitute another model or retune fixed parameters.

## Separate the observation from its interpretation

Write the reported contrast as a measured quantity, comparison unit, intervention, and observation window. Keep within-sample contrasts distinct from comparisons across independent samples. Distinguish latent neural activity from the instrument's output and the biological outcome the researcher wants explained.

Ask for missing experimental facts that could change that interpretation, using the user's terminology. For example, ask whether “more active” means more recorded spikes or a larger reporter signal, and whether the comparison uses neighbors in one sample or means across samples. Readouts in a related paper do not establish the user's assay. If unanswered, retain the alternatives and state which results apply to each. When the request also supplies a computational design, apply step 0's technical-clarification and delegation rules to its consequential missing settings.

## Choose the comparison before the model

Express each credible explanation as `intervention -> affected process -> circuit or cellular consequence -> measured readout`. Mark each unsupported link as an assumption. Use the literature gate for evidence that could change these links; do not simulate every imaginable mechanism.

Record the comparison under `Acceptance boundary` in `NeuroSpecification.md`; use a linked `mechanism-comparison.md` artifact when the table would make the specification too long:

| Explanation and causal chain | Distinguishing measurement or intervention | Smallest model that preserves the distinction | Decisive unmeasured assumption and check |
|---|---|---|---|
| Reduced presynaptic recruitment -> fewer spikes -> weaker postsynaptic response | Restore the presynaptic spike train and measure transmission | Separate spike generation from efficacy per spike | Recovery of transmission when spikes are restored; test a direct efficacy loss alternative |
| Reduced transmission per spike -> weaker postsynaptic response | Impose the same presynaptic spike train and measure transmission | Represent efficacy independently from spike generation | Preserved imposed spikes with reduced transmission; allow mixed mechanisms |

The example establishes a decision boundary, not a required synaptic model. Choose states, inputs, coupling, and observation mapping that preserve the differences needed for this question. Add biological detail only when omitting it collapses a relevant prediction or invalidates the readout. Keep multiple mechanisms compatible when the evidence permits their coexistence.

If no available measurement separates the candidates, state the non-identifiability and identify a new measurement that could. Modeling may still establish conditional sufficiency or a falsifiable prediction; do not promise mechanism identification from indistinguishable observations.

## Check structural distinguishability before large simulations

Compare the effective equations, initial State, inputs, parameter changes, and observation mappings for the conditions the experiment distinguishes. If two interventions become identical in all those respects, their predicted distributions are identical by construction. Independent seeds do not restore biological distinguishability.

Decide whether that equivalence is the hypothesis being tested or an artifact of the abstraction. If it is intentional, state the limited hypothesis and its conditional prediction; a small witness or analytic argument can suffice. If the original comparison depends on a difference discarded by the agent's design, restore that difference before production. If the difference is excluded by a user-fixed design, report the resulting limit and the adaptation needed to test it. Do not reject a broader mechanism family because one collapsed representation fails a negative control.

Use separate regions, cell types, pathways, or kinetics only when they carry a required difference. A prescribed external input can test input withdrawal, but cannot establish the omitted source's dynamics, feedback, or event initiation.

## Test the assumptions that can change the answer

Rank unmeasured assumptions by whether a plausible alternative could reverse the headline result or change which mechanism appears supported. Prioritize observation mapping, operating regime, effective intervention strength, and omitted causal routes when they control the comparison; do not give every parameter an equal sweep by default.

Use free parameters and controls permitted by the request for these checks. If a decisive assumption is fixed by the user and variation is not authorized, preserve it and state the conditional inference; proposing a sensitivity comparison does not authorize changing the requested run.

For each decisive assumption:

1. State the conclusion it enables and a defensible alternative or range. Ground the range in supplied data or literature when available; label an unsupported range exploratory.

2. Choose the smallest informative comparison: change the observation mapping, match effective perturbation, remove a pathway, restore a mediator, or vary the relevant parameter while preserving matched controls.

3. Determine whether the conclusion survives, reverses, or becomes undecidable. Keep failed and unfavorable settings with the tested boundaries.

4. Translate the dependence into a conditional claim and a measurement the researcher could make to resolve it.

Use additional seeds for stochastic repeatability and solver checks for numerical reliability. Neither establishes robustness to an untested measurement model or biological assumption. An effect found only near complete functional loss supports a conditional existence claim; it does not establish that a modest experimental perturbation has that effect.

For regional or cell-type comparisons, separate differences in circuit response from differences in delivered perturbation. Match measured functional effects when available; otherwise compare both explanations conditionally. Equal nominal dose, construct, or fluorescence does not by itself fix the effective intervention.

## Carry the answer boundary through review and delivery

Map each part of the original question to its modeled observable, decisive evidence, substitutions, and unresolved assumptions in the result assessment. Preserve a proxy's name throughout: a reporter change is not automatically a firing change, and population oscillations do not establish a behavioral phenotype.

For the leading remaining alternatives, propose a feasible discriminating experiment and state what each plausible outcome would support, weaken, or leave unresolved, including mixed or null results. Prefer a measurement that separates mechanisms over one that repeats their shared prediction. A rescue shared by several causal routes is not, by itself, proof of one route.

Keep the comparison, required controls, assumption checks, and conditional claim boundaries in or linked from `NeuroSpecification.md`. Use step 1 for BrainX package/API choices and steps 2-4 for implementation and evidence; include the comparison and its result assessment in step 5.
