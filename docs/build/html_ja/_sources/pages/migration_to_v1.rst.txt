migration_to_v1
===============

**To Users of PyFemtet 0.x**

In PyFemtet v1, many functions and arguments have been changed
to improve user-friendliness and development efficiency.

Versions 0.9 and later serve as transition versions to v1,
where legacy functions and arguments are still fully available;
however, features that have been changed or removed will issue warnings.

The main changes in version 1 are listed on this page along with usage examples.
If you need to modify your existing scripts, please refer to the examples below.


.. contents:: Index of this page
   :depth: 3


How to import exceptions like ModelError
----------------------------------------

They will be moved to ``pyfemtet.opt.exceptions`` module.

.. code-block:: python

   # < 1.0.0
   from pyfemtet.core import ModelError, MeshError, SolveError

   # >= 1.0.0
   from pyfemtet.opt.exceptions import ModelError, MeshError, SolveError


add_objective() and add_constraints()
-------------------------------------

``name`` argument
+++++++++++++++++

The ``name`` argument will be the first argument and required.

.. code-block:: python

   def some_objective(Femtet):
      ...

   # < 1.0.0
   femopt.add_objective(some_objective, 'objective name')

   # >= 1.0.0
   femopt.add_objective('objective name', some_objective)

   # The samples below works with both versions
   femopt.add_objective(name='objective name', fun=some_objective)
   femopt.add_objective(fun=some_objective, name='objective name')


``args`` argument
+++++++++++++++++

The ``args`` will be recognized as a sequence.

.. code-block:: python

   # < 1.0.0
   femopt.add_objective(..., args=femopt.opt)

   # The samples below works with both versions
   femopt.add_objective(..., args=(femopt.opt,))
   # or
   femopt.add_objective(..., args=[femopt.opt])


Arguments of User-Defined Functions in cases other than FemtetInterface
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In cases other than ``FemtetInterface`` (for example,
``ExcelInterface``), the first argument of user-defined
functions is given by PyFemtet.

This item is not relevant when using the FemtetInterface
and its derived classes (such as FemtetWithNXInterface).


.. code-block:: python

   # < 1.0.0
   def user_defined(opt):
       ...

   # >= 1.0.0
   def user_defined(fem, opt):
       ...

   # The samples below works with both versions
   femopt.add_objective(fun=user_defined, args=(opt,))


.. note::

   What is passed to the first argument varies depending
   on the FEMInterface you using. For example, ``ExcelInterface`` passes COM object of Excel(R)
   to your function. 
   
   For details, please refer to
   the `API Reference of Latest version <https://pyfemtet.readthedocs.io/en/latest/pages/api.html>`__ of each class.
   If the concrete class’s API reference does not include
   an object_pass_to_fun section, please refer to the
   corresponding section in its parent class.
