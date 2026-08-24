Sound localization
==================

This experiment uses only the submillisecond difference between left- and right-ear arrival times to drive a spiking direction readout. Exact delay taps, event communication, and batched independent state make timing semantics the core of the implementation.

Prompt
------

.. container:: prompt-bubble

   Build a spiking network that tells whether a sound came from the left or right using only the tiny difference in when it reaches the two ears.

Agent decision path
-------------------

.. container:: agent-trace

   1. Classify the system as a point-neuron auditory circuit.
   2. Route neurons to ``brainpy-state``, spike delivery to ``brainevent``, and physical timing to ``brainunit``.
   3. Select the Jeffress delay-line principle with two auditory relays and a coincidence bank.
   4. Round physical interaural delays once to an explicit ``0.05 ms`` time grid.
   5. Deliver binary spikes through fixed-fan-out projections to left and right readouts.
   6. Give every interaural delay independent mapped membrane, refractory, and delay state.
   7. Advance all lanes through one state-aware ``vmap2`` step inside ``for_loop``.
   8. Verify delay impulse convention, dimensional bounds, and direction labels with three tests.

Result
------

.. container:: result-lede

   Every tested negative interaural time difference is classified as ``RIGHT``, every positive difference as ``LEFT``, and simultaneous arrival as ``CENTER``. All three focused tests pass.

.. raw:: html

   <div class="sound-evidence" role="img" aria-label="Direction classification evidence: negative interaural delays are right, zero is center, and positive delays are left.">
     <div><strong>−0.60 ms</strong><span>RIGHT</span></div>
     <div><strong>−0.20 ms</strong><span>RIGHT</span></div>
     <div><strong>0.00 ms</strong><span>CENTER</span></div>
     <div><strong>+0.20 ms</strong><span>LEFT</span></div>
     <div><strong>+0.60 ms</strong><span>LEFT</span></div>
   </div>

The selected run did not generate a plot. This structured result preserves the committed classification evidence without inventing a graph.

With-BrainX skill/Without skill comparison
------------------------------------------

.. grid:: 1 2 2 2
   :gutter: 2
   :class-container: comparison-grid

   .. grid-item::

      **With BrainX skill**

      .. container:: pending-label

         Source captured · benchmark pending

      Add measured rollout throughput across the same delay grid, including compile policy and all timing-semantic tests.

   .. grid-item::

      **Without BrainX skill**

      .. container:: pending-label

         Matched run not collected

      Use the same ``0.05 ms`` grid, ``+/-0.6 ms`` bounds, and direction convention before comparing speed or correctness.
