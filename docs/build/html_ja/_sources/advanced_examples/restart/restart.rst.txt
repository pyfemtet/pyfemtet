Restarting Optimization
=======================================================

This sample explains how to resume an optimization that
was interrupted partway through.


Sample Files
-------------------------------------------------------

.. note::
   Keep the
   :download:`sample project<gal_ex13_parametric.femprj>`
   and
   :download:`sample script<gal_ex13_parametric_restart.py>`
   on same folder.


What This Sample Will Do
-------------------------------------------------------

For the FEM problem, we will determine the design
parameters through random sampling and conduct three
analyses.

Next, we will conduct three optimization trials using
the geneticalgorithm NSGA-II.

Finally, we will conduct three optimization trials using
the Gaussian Process Regression Bayesian Optimization
algorithm.


.. note::

   By doing this, we can switch optimization methods
   during the process while monitoring the progress of
   optimization, or add random sampling for creating
   surrogate models.


.. note::

   When restarting, the number and names of variables,
   as well as the number and names of objective
   functions and constraints must be consistent.
   However, you can change the bounds of variables,
   direction of objective functions, and content of
   constraints.


.. warning::

   When using OptunaOptimizer, the .db file with the same name
   (in this case restarting-sample.db) that is saved along with
   csv is required to be in the same folder as the csv file.

   Please do not delete or rename it.


Design Variables
--------------------------------------------------------

.. figure:: model.png
   :width: 400
   
   Appearance of the Model

================= ==============================================
Variable Name     Description
================= ==============================================
width             The thickness of the tuning fork.
length            The length of the tuning fork.
base_radius       The radius of the tuning fork's handle.
================= ==============================================


Objective Function
--------------------------------------------------------

- First Resonant Frequency (target value: 800)


Execution Result of the Samples
--------------------------------------------------------

.. figure:: result.png
   :width: 300

   Screenshot of the result


.. warning::

   When performing effective optimization on real-world
   problems, a greater number of trials is necessary.
