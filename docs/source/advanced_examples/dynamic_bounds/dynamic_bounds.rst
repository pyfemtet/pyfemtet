Function-based Bounds (dynamic_bounds)
======================================

.. warning::

    This feature is currently under development.
    Its behavior and usage may change in future versions.


Sample Files
------------

.. note::

   Keep the
   :download:`sample project<pipe.femprj>`
   and
   :download:`sample script<pipe_dynamic_bounds.py>`
   on same folder.


What is dynamic_bounds
----------------------

``dynamic_bounds`` is a feature that can be used **as an alternative to regular constraints**,
when **“the lower/upper bounds of a variable can be expressed as a function of other variables.”**

In pyfemtet, regular constraints work as follows:

- propose variables
- check whether the constraint is violated
- if violated, propose again

When constraints are tight, steps 2→3 may repeat many times.
With ``dynamic_bounds``, **invalid proposals are never generated**, eliminating this overhead.

.. note::

    ``dynamic_bounds`` is *not* a feature for speeding up constraint evaluation.
    It replaces a regular constraint **only when the constraint can be rewritten as variable bounds**.

.. note::

    ``initial_value`` must always lie within the (dynamic) lower and upper bounds.


When dynamic_bounds can be used
-------------------------------

- The constraint can be rewritten in the form “the lower/upper bound of a variable is a function of other variables.”
- Other constraints should still be implemented using ``add_constraint``.
  (``dynamic_bounds`` and regular constraints can be used together.)


Example where dynamic_bounds is applicable
------------------------------------------

Problem
^^^^^^^

- Variable a: lower 0, upper 10
- Variable b: lower 0, upper 10
- Constraint: ``a + b < 10``

This constraint can be rewritten as:

- Variable a: lower 0, upper 10
- Variable b: lower 0, upper ``10 - a``


Defining dynamic_bounds
^^^^^^^^^^^^^^^^^^^^^^^

First, define a function that computes the bounds:

.. code-block:: python

    def dynamic_bounds_of_b(opt) -> tuple[float, float]:
        """Return the dynamic lower and upper bounds of b."""
        params = opt.get_variables()
        return 0, 10. - params['a']

.. note::

    A dynamic-bounds function must accept ``opt: AbstractOptimizer``
    and must return **two numeric values** representing the lower and upper bounds.


Registering the variable
^^^^^^^^^^^^^^^^^^^^^^^^

Then register the function when adding the variable:

.. code-block:: python

    femopt.add_parameter(name='a', initial_value=0, lower_bound=0, upper_bound=10)
    femopt.add_parameter(
        name='b', initial_value=0,
        properties={
            'dynamic_bounds_fun': dynamic_bounds_of_b
        }
    )

With this setup, the bounds of ``b`` are always ``(0, 10 - a)``,
and combinations that do not satisfy ``a + b < 10`` will never be proposed.


.. warning::

    Only variables that have already been registered with ``add_parameter``
    at the time the dynamic_bounds function is evaluated can be referenced
    inside that function.

    If you reference a variable that has not yet been registered,
    no error is raised, and the value from the previous iteration
    will be used unintentionally.

    .. code-block:: python

        # OK:
        add_parameter('a', ...)
        add_parameter('b', properties={"dynamic_bounds_fun": fun_using_a})

        # NG:
        add_parameter('b', properties={"dynamic_bounds_fun": fun_using_a})
        add_parameter('a', ...)
