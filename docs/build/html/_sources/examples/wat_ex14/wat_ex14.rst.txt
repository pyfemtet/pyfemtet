Heating element on the substrate
===================================

Using Femtet's heat conduction analysis solver, we explain an example of searching for the substrate dimensions that minimize the size of the substrate while keeping the maximum temperature of an IC chip on the substrate to a minimum.


Sample File
--------------------
.. note::

   Keep the :download:`sample project<../../../../pyfemtet/FemtetPJTSample/wat_ex14_parametric.femprj>`
   open in Femtet, and double-click on the :download:`sample code<../../../../pyfemtet/FemtetPJTSample/wat_ex14_parametric.py>`
   to execute it.

.. note::

   For details on the FEM problem, please refer to FemtetHelp / Examples / Heat Conduction Analysis / Example 14.


Design Variables
------------------

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

.. literalinclude:: ../../../../pyfemtet/FemtetPJTSample/wat_ex14_parametric.py
   :language: python
   :linenos:
   :caption: wat_ex14_parametric.py


Execution Result of the Sample Code
--------------------------------------------------

.. figure:: wat_ex14_result.png
   :width: 300

   Execution result of wat_ex14_parametric.py. This is a pair plot with the combination of each objective function on the vertical axis and horizontal axis.


From the results of the 20 trials, the following can be observed.

- The temperature of the main chip and the temperature of the sub chip can both be reduced by decreasing one of them.
- Reducing the substrate size increases the temperature of the main chip.
- Reducing the substrate size increases the temperature of the sub chip.

From this, it can be seen that it is possible to design both the main chip and the sub chip to minimize temperature, but there is a trade-off relationship between the temperature of each chip and the substrate size, and it is understood that these minimizations are not compatible.

.. tip::

   In multi-objective optimization, it is possible that the optimization of objective functions may not be compatible. In such cases, the designer needs to select the appropriate design from among the trade-off solutions.

.. note::

   Since the physical reasons for these trade-offs cannot be derived from optimization algorithms, designers need to interpret the analysis results of Femtet.

In this problem, it can be inferred that the reduced substrate size has decreased the heat dissipation capacity to the environment, causing the chip temperature to rise because heat is not escaping from the substrate.

.. note::

   Results may vary slightly depending on the versions of Femtet, PyFemtet, and the optimization engine it depends on.
