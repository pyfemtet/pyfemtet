External CAD (Solidworks) Integration
========================================

PyFemtet allows parametric optimization even for analysis models created with external CAD (Solidworks) and imported into Femtet.

An example will be explained using an H-shaped steel that was parametrically modeled in an external CAD (Solidworks) and analyzed using Femtet's stress analysis solver to minimize volume while minimizing displacement.

.. note::

   Other than the sample code and execution results, the items are the same as in :doc:`../NX_ex01/NX_ex01`.



Sample File
------------------------------
.. note::

   Place the :download:`sample model<../../../../pyfemtet/FemtetPJTSample/Sldworks_ex01/Sldworks_ex01.SLDPRT>`
   and :download:`sample project<../../../../pyfemtet/FemtetPJTSample/Sldworks_ex01/Sldworks_ex01.femprj>` in the same folder, keep the project open in Femtet,
   and double-click on :download:`sample code<../../../../pyfemtet/FemtetPJTSample/Sldworks_ex01/Sldworks_ex01.py>` to execute.



Sample Code
------------------------------

.. literalinclude:: ../../../../pyfemtet/FemtetPJTSample/Sldworks_ex01/Sldworks_ex01.py
   :language: python
   :linenos:
   :caption: Sldworks_ex01.py


Execution Result of the Sample Code
----------------------------------------

.. figure:: Sldworks_ex01_result.png
   :width: 300

   Execution result of Sldworks_ex01.py. The horizontal axis is displacement, and the vertical axis is volume.

After 20 trials, a Pareto set of displacement and volume is obtained.


.. note::

   Results may vary slightly depending on the versions of Femtet, PyFemtet, and the optimization engine it depends on.
