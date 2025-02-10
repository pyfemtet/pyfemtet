"""Optimization using parametric analysis output settings as the objective function

This demo shows how to use the values outputted by Femtet's parametric
analysis output setting feature as the objective function for optimization.
This feature allows you to perform optimization without coding the objective function.


Note:

    Please be aware of the following when using this feature.

    - The sweep table from the parametric analysis will be deleted.
    - Output settings that produce complex numbers or vectors will only use
      the first value as the objective function. (For complex numbers, it will be
      the real part, and for vector values, it will be components such as X.)


Corresponding project: gau_ex12_parametric.femprj

"""

from pyfemtet.opt import FEMOpt, FemtetInterface


if __name__ == '__main__':

    # Initialize an object to connect to
    # Femtet for referencing Femtet settings.
    fem = FemtetInterface()

    # Set the output settings of the parametric analysis as the objective function.
    # `number` is the index from the `results output settings` tab of the
    # Femtet parametric analysis dialog, and `direction` is
    # the goal of that objective function (similar to FEMOpt.add_objective).

    # Mutual inductance
    fem.use_parametric_output_as_objective(number=1, direction=1.5e-7)

    # Strength of magnetic field at the center of the coil
    fem.use_parametric_output_as_objective(number=2, direction='minimize')

    # Initialize optimization object.
    # Pass in the previously initialized fem object.
    femopt = FEMOpt(fem=fem)

    #  Set parameters.
    femopt.add_parameter('in_radius', 10, 5, 10)
    femopt.add_parameter('out_radius', 20, 20, 25)

    # Execute optimization.
    femopt.set_random_seed(42)  # Fix random seed
    femopt.optimize(n_trials=20)
