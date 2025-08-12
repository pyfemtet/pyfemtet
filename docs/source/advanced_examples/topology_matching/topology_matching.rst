Topology Matching
==========================================

This examples demonstrates how to keep 
the consistency of boundary condition assignments
when the topology of the model changes.


Sample Files
--------------------------------------------------

.. note::
   Keep the :download:`sample project<topology_matching.femprj>`
   and :download:`sample script<topology_matching.py>`
   in the same folder.


Explamation of the Topology Matching
--------------------------------------------------

In optimization, the model geometry is updated
by changing the design parameters.

At this time, depending on how the model is created,
there is a known issue where the topology numbers
inside the CAD can change, resulting in unintended
assignments of boundary conditions and mesh sizes.

We have experimentally implemented a Topology Matching
feature based on the technique described in [1].

This sample demonstrates optimization using Topology Matching
to address the problem where boundary conditions would otherwise
be lost with conventional methods.

.. note::

    **Limitations:**
    This feature currently supports only models with a single body.


.. note::

   For further concept about the effect of the
   topology matching feature, please refer to
   :download:`effect_of_topology_matching.pdf` .


Design Variables
--------------------------------------------------

.. figure:: topology_matching_model.png
   
   Appearance of the Model


============== ========================
Variable Name  Description  
============== ========================
hoge           hoge
hoge           hoge
============== ========================


Objective Function
-----------------------------

- hoge (to minimize)


Sample Code
--------------------------------------------------

.. literalinclude:: ../_temporary_sample_files/topology_matching.py
    :language: python
    :linenos:
    :caption: topology_matching.py


Execution Result of the Sample Code
--------------------------------------------------
.. figure:: topology_matching_result.png
   :width: 300

   Execution result of topology_matching.py.
   


Reference
--------------------------------------------------

[1]
Benjamin Jones, James Noeckel, Milin Kodnongbua, Ilya Baran, and Adriana Schulz. 2023.
B-rep Matching for Collaborating Across CAD Systems.
ACM Trans. Graph. 42, 4, Article 104 (August 2023), 13 pages.
https://doi.org/10.1145/3592125
