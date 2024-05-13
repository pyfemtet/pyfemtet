Heat-generating elements on the substrate (parallel computation)
==========================================================================================


Parallelize wat_ex14_parametric with 3 Femtet instances. Other items, except for sample code and execution results, are the same as :doc:`../wat_ex14/wat_ex14`.


Sample File
------------------------------
.. note::

   Keep the :download:`sample project <../../../../pyfemtet/FemtetPJTSample/wat_ex14_parametric.femprj>`
   open in Femtet, then double-click on the :download:`sample code <../../../../pyfemtet/FemtetPJTSample/wat_ex14_parallel_parametric.py>`
   to execute it.

.. note::

   For details on the FEM problem, please refer to FemtetHelp / Examples / Heat Conduction Analysis / Example 14.


設計変数
------------------------------

.. figure:: wat_ex14_model.png
   
   Appearance of the Model

============== ========================
Variable Name  Description
============== ========================
substrate_w    Width of the substrate
substrate_d    Depth of the substrate
============== ========================

Objective Function
-----------------------------

- Maximum temperature of the main chip (to minimize)
- Maximum temperature of the sub chip (to minimize)
- Occupied area on the substrate plane (to minimize)


Sample Code
---------------

.. literalinclude:: ../../../../pyfemtet/FemtetPJTSample/wat_ex14_parallel_parametric.py
   :language: python
   :linenos:
   :caption: wat_ex14_parallel_parametric.py


.. note::

   To parallelize, simply pass the desired number of parallelizations to the ``n_parallel`` argument of the ``optimize()`` method.


Execution Result of the Sample Code
----------------------------------------

Execution Environment

+--------+--------------------------------------------+
| OS     | windows 10                                 |
+--------+--------------------------------------------+
| CPU    | Intel Core-i7 12700 (12 cores, 20 threads) |
+--------+--------------------------------------------+
| Memory | 32 GB                                      |
+--------+--------------------------------------------+


Execution Results

================================  =========================
Without Parallelization           With 3 Parallelizations
117 sec                           74 sec
================================  =========================

In this demo, calculations were performed with 3 Femtet instances. For the problem in :doc:`../wat_ex14/wat_ex14`, without using parallelization in the above execution environment, it took 117 seconds for 20 trials. However, in this demo, 21 trials were completed in 74 seconds, reducing the execution time by 37%.

.. note::

   Generally, when parallelizing numerical calculations by N, the execution time does not simply become 1/N.

.. warning::

   The acceleration effect of parallelization varies depending on the execution environment and analysis model.

.. note::

   Results may vary slightly depending on the versions of Femtet, PyFemtet, and the optimization engine it depends on.
