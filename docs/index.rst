BrainX Skill documentation
==========================

`BrainX Skill <https://github.com/chaobrain/BrainX-skill>`_ is a portable instruction and workflow layer that installs on general coding agents. Describe a neuroscience question in natural language; the skill routes it to the BrainX packages that own the model scale, state, units, events, and execution.

The agent translates the research question. Explicit BrainX code controls the dynamics and produces the evidence.

----

Features
^^^^^^^^

.. grid::

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Intuitive and automatic
         :class-title: sd-fs-6

         .. div:: sd-font-normal

            Turn a natural-language research question into a clear experiment at the correct modeling scale.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Accurate
         :class-title: sd-fs-6

         .. div:: sd-font-normal

            Use explicit BrainX models, units, state transitions, solvers, and connectivity to control the dynamics in code.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Creative
         :class-title: sd-fs-6

         .. div:: sd-font-normal

            Convert a hypothesis into perturbations, controls, sweeps, and observable results with one prompt.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Efficient
         :class-title: sd-fs-6

         .. div:: sd-font-normal

            Apply state-aware ``jit``, ``vmap``, and ``for_loop`` transforms to compile simulation and batch experiments.

----

Installation
^^^^^^^^^^^^

.. code-block:: bash

   npx brainx-skill install

----

Learn more
^^^^^^^^^^

.. grid::

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`download;2em` Installation
         :class-card: sd-text-black sd-bg-light
         :link: installation
         :link-type: doc

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`rocket_launch;2em` Quickstart
         :class-card: sd-text-black sd-bg-light
         :link: quickstart
         :link-type: doc

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`science;2em` Creative experiment verification
         :class-card: sd-text-black sd-bg-light
         :link: creative-experiment-verification
         :link-type: doc

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`speed;2em` Speed and performance
         :class-card: sd-text-black sd-bg-light
         :link: speed-and-performance
         :link-type: doc

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`model_training;2em` Use the skill for training
         :class-card: sd-text-black sd-bg-light
         :link: use-the-skill-for-training
         :link-type: doc

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`library_books;2em` Paper reproductions
         :class-card: sd-text-black sd-bg-light
         :link: paper-reproduction
         :link-type: doc

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`account_tree;2em` Skill reference
         :class-card: sd-text-black sd-bg-light
         :link: skill-reference
         :link-type: doc

----

See also the ecosystem
^^^^^^^^^^^^^^^^^^^^^^

BrainX Skill is part of the `BrainX brain simulation ecosystem <https://brainx.chaobrain.com/>`_.

----

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Quickstart

   installation
   Quickstart <quickstart>
   supported-agents

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Examples

   creative-experiment-verification
   speed-and-performance
   use-the-skill-for-training
   paper-reproduction

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Skill reference

   skill-reference
