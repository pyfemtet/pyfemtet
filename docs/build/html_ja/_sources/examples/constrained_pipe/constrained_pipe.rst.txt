Restrict parameter combinations
===============================

This example explains how to use the constraint function
when you want to restrict parameter combinations.


Sample File
-----------
.. note::

   Keep the :download:`sample project<../_temporary_sample_files/constrained_pipe.femprj>`
   open in Femtet and double-click on the :download:`sample code<../_temporary_sample_files/constrained_pipe.py>`
   to execute it.


Design Variables and analysis conditions
----------------------------------------

.. figure:: model.png
   
   Model appearance, analysis conditions and design variables

============== ====================================
Variable Name  Description
============== ====================================
external_r     Outer radius of the pipe.
internal_r     Inner radius of the pipe.
============== ====================================


Objective Function
------------------------------

Max Mises stress of the pipe.


Sample Code
------------------------------

.. literalinclude:: ../_temporary_sample_files/constrained_pipe.py
   :language: python
   :linenos:
   :caption: constrained_pipe.py


Execution Result of the Sample Code
-------------------------------------

.. figure:: result.png
   :width: 300

   Execution result of constrained_pipe.py.
   There is no trial with pipe thickness < 1.


.. note::

   Results may vary slightly depending on the versions of Femtet, PyFemtet, and the optimization engine it depends on.
