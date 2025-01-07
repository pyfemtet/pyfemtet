Optimization Using a Surrogate Model
==================================================

This sample demonstrates how to use Femtet to create
training data and optimize using surrogate models.


Sample Files
--------------------------------------------------

.. note::
   Keep the :download:`sample project<gal_ex13_parametric.femprj>`,
   :download:`sample script 1<gal_ex13_create_training_data.py>`
   to create training data **with Femtet** and 
   :download:`sample script 2<gal_ex13_optimize_with_surrogate.py>`
   to make a surrogate model **without Femtet** and optimize
   on same folder.


How to run this Sample
--------------------------------------------------

When you double-click on `gal_ex13_create_training_data.py`,
the creation of training data for the surrogate model using
Femtet will begin.

Once the number of Femtet analysis executions exceeds
approximately 100, please double-click on
`gal_ex13_optimize_with_surrogate.py` to run it.
(The optimization results at the bottom of the page are
based on a model created from 100 analysis data points.)

.. note::
   Since the surrogate model optimization requires no
   Femtet execution, you can run `gal_ex13_optimize_with_surrogate.py`
   during running `gal_ex13_create_training_data.py`
   without any additional Femtet license.


.. tip::
    **What's Surrogate Model?**

    The surrogate model handled by PyFemtet is a machine learning
    model that predicts values of the objective function for unknown
    design variables by learning a set of known design variables and
    objective functions.

    Generally, to create high-quality training data, more FEM
    analysis data is required than what is typically needed for
    regular optimization, as mentioned in the examples. However,
    once training data has been created, it allows for very fast
    calculations of the objective function.

    Therefore, in situations where the items for design variables
    and objective functions are somewhat fixed and problems
    frequently arise with varying ranges or target values, it
    becomes possible to quickly approximate design variables that
    meet desired target values.


.. note::

   For details on the FEM problem, please refer to
   FemtetHelp / Examples / Stress Analysis / Example 13.


Design Variables
--------------------------------------------------

.. figure:: gal_ex13_parametric.png
   
   Appearance of the Model

============== ==============================================
Variable Name  Description
============== ==============================================
length         Length of the tuning fork
width          Thickness of the tuning fork
base_radius    Thickness of the base (fixed in optimization)
============== ==============================================


Objective Function
--------------------------------------------------

- First resonance frequency (aim to 1000 and 2000)


Sample Code
--------------------------------------------------
.. literalinclude:: gal_ex13_create_training_data.py
   :language: python
   :linenos:
   :caption: gal_ex13_create_training_data.py

.. literalinclude:: gal_ex13_optimize_with_surrogate.py
   :language: python
   :linenos:
   :caption: gal_ex13_optimize_with_surrogate.py



Execution Result of the Sample Code
--------------------------------------------------
.. figure:: optimized_result_target_1000.png
   :width: 300

   Optimization result (target: 1000 Hz)

.. figure:: optimized_result_target_2000.png
   :width: 300

   Optimization result (target: 2000 Hz)


The design variables for a tuning fork with first resonance frequencies
of 1000 or 2000 were explored using a surrogate model. The resulting
design variables are listed in the upper right corner of the figure.

Using these design variables, we recreated the model in Femtet and
executed analyses, with results shown in the lower right corner of each
figure, allowing for comparison between the surrogate model and FEM results.
