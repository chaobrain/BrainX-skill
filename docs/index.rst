BrainX Skill
============

BrainX Skill is a portable instruction and workflow layer for general coding agents. Install it once, describe a neuroscience question in natural language, and let the agent route the task to the BrainX packages that own its modeling scale, state, units, events, and execution.

The agent translates the research question. Explicit BrainX code controls the dynamics and produces the evidence.

.. raw:: html

   <figure class="quickstart-media">
     <video controls autoplay muted loop playsinline preload="metadata" aria-label="BrainX Skill turning a natural-language research request into a simulation and result">
       <source src="_static/video/brainx-skill.mp4" type="video/mp4">
       Your browser does not support embedded MP4 video.
     </video>
     <figcaption>A research prompt becomes an executable BrainX experiment.</figcaption>
   </figure>

Quick start
-----------

Install the portable skill bundle, then give your agent a concrete modeling question.

.. code-block:: bash

   npx brainx-skill install

.. code-block:: text

   Show why a neuron fires quickly at first but gradually slows during a
   steady input, then remove its adaptation current and reveal what changes.

What the skill changes
----------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Intuitive and automatic
      :shadow: md
      :class-card: sd-border-0

      State the scientific intent in natural language. The skill identifies the modeling scale, selects the owning BrainX packages, and shapes a clear experiment.

   .. grid-item-card:: Accurate
      :shadow: md
      :class-card: sd-border-0

      The agent writes the model; BrainX executes it. Units, state transitions, solvers, connectivity, and dynamics remain explicit in code rather than being approximated by language generation.

   .. grid-item-card:: Creative
      :shadow: md
      :class-card: sd-border-0

      Turn a hypothesis into a perturbation, control, sweep, and observable result. The cases below show ideas tested across cells, circuits, populations, regions, and training workflows.

   .. grid-item-card:: Efficient
      :shadow: md
      :class-card: sd-border-0

      Use state-aware transforms such as ``jit``, ``vmap``, and ``for_loop`` to compile time evolution and batch conditions, trials, observers, or network realizations.

Creative experiment verification
--------------------------------

Use these cases when the central claim is a neural mechanism, perturbation, or emergent behavior.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 01 — Spike-frequency adaptation
      :link: cases/01-spike-frequency-adaptation
      :link-type: doc

      Remove one adaptation current and expose the causal change in firing.

   .. grid-item-card:: 05 — Alpha rhythm
      :link: cases/05-alpha-rhythm
      :link-type: doc

      Weaken inhibition and compare the resulting population signal.

   .. grid-item-card:: 06 — Seizure recruitment
      :link: cases/06-seizure-recruitment
      :link-type: doc

      Find when a focal burst stays local or recruits neighboring regions.

   .. grid-item-card:: 08 — Binocular rivalry
      :link: cases/08-binocular-rivalry
      :link-type: doc

      Explain alternating percepts through adaptation, noise, and competition.

   .. grid-item-card:: 09 — Neural compass
      :link: cases/09-neural-compass
      :link-type: doc

      Rotate and lesion a ring attractor to map recovery and failure.

Speed and performance
---------------------

Use these cases when batching, long simulation, or compiled execution is part of the scientific requirement.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 03 — Sound localization
      :link: cases/03-sound-localization
      :link-type: doc

      Batch precise auditory delays through an event-driven spiking circuit.

   .. grid-item-card:: 07 — Cortical wave obstacle
      :link: cases/07-cortical-wave-obstacle
      :link-type: doc

      Sweep lesion size and inhibition across a spatial spiking network.

   .. grid-item-card:: 10 — Prior bias
      :link: cases/10-prior-bias
      :link-type: doc

      Measure a psychometric effect and compiled decision throughput.

   .. grid-item-card:: 12 — Edge of criticality
      :link: cases/12-edge-of-criticality
      :link-type: doc

      Map coupling and network realizations to locate a stable transition.

Use the skill for training
--------------------------

Use these cases when state must persist across learning, reversal, sleep, or recall.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 02 — Learning temporal order
      :link: cases/02-learning-temporal-order
      :link-type: doc

      Acquire a temporal rule, reverse it, and show relearning in the same circuit.

   .. grid-item-card:: 11 — Sleep memory replay
      :link: cases/11-sleep-memory-replay
      :link-type: doc

      Compare matched networks with replay enabled or suppressed during sleep.

Paper reproduction
------------------

Use this category to reconstruct published models and verify reported results with explicit BrainX experiments.

:doc:`Open paper reproduction <paper-reproduction>`

How to read a case
------------------

Each case keeps the evidence chain visible: the exact prompt, the agent's modeling decisions, and the generated result. The comparison area is intentionally reserved for future matched runs with and without the skill; it does not imply benchmark data that has not yet been collected.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Experiments

   creative-experiment-verification
   speed-and-performance
   use-the-skill-for-training
   paper-reproduction
