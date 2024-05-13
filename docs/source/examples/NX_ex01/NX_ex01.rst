External CAD (NX) Integration
==============================

With PyFemtet, you can optimize the collaboration between CAD and Femtet even when importing models created in an external CAD (NX) into Femtet for analysis.


Sample File
------------------------------
.. note::

   Place the 
   :download:`sample model<../../../../pyfemtet/FemtetPJTSample/NX_ex01/NX_ex01.prt>`
   and
   :download:`sample project<../../../../pyfemtet/FemtetPJTSample/NX_ex01/NX_ex01.femprj>`
   in the same folder.
   Keep the project open in Femtet, then double-click on the
   :download:`sample code<../../../../pyfemtet/FemtetPJTSample/NX_ex01/NX_ex01.py>`
   to execute it.


Details as a FEM Problem
------------------------------

.. figure:: NX_ex01_analysis.png
   
   Appearance of the Model (and Analysis Conditions)

- fix ... Fully Fixed
- load ... Load in the -Z direction (1N)
- mirror ... Symmetrical to the XZ plane


Design Variables
------------------------------

.. figure:: NX_ex01_model_dsgn.png
   
   Appearance of the Model Section (and Design Variables)

============== ============
Variable Name  Description
============== ============
A              Web Tickness
B              Flange Tickness
C              Flange Bending
============== ============


Objective Function
------------------------------

- Maximum Displacement in the Z direction (set to 0)
- Volume (minimize)


Sample Code
------------------------------

.. literalinclude:: ../../../../pyfemtet/FemtetPJTSample/NX_ex01/NX_ex01.py
   :language: python
   :linenos:
   :caption: NX_ex01.py


Execution Result of the Sample Code
----------------------------------------

.. figure:: NX_ex01_result.png
   :width: 300

   Execution result of NX_ex01.py. The horizontal axis is displacement, and the vertical axis is volume.

After the 20 trials, a Pareto set of displacement and volume is obtained.


.. note::

   Results may vary slightly depending on the versions of Femtet, PyFemtet, and the optimization engine it depends on.
