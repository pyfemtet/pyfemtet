"""Single-objective optimization: bending with consideration for springback.

Using Femtet's stress analysis solver, we will determine the bending angle
to achieve the desired material bend angle with consideration for springback.
Elasto-plastic analysis is available in an optional package.

Corresponding project: gal_ex58_parametric.femprj
"""
import numpy as np
from win32com.client import constants
from optuna.integration.botorch import BoTorchSampler

from pyfemtet.opt import FEMOpt, OptunaOptimizer


def bending(Femtet):
    """Get the material bend angle.

    Note:
        The objective or constraint function should take Femtet
        as its first argument and return a float as the output.

    Params:
        Femtet: This is an instance for manipulating Femtet with macros. For detailed information, please refer to "Femtet Macro Help".

    Returns:
        float: material bend angle.
    """
    Gogh = Femtet.Gogh

    # Set the mode after unloading.
    Gogh.Galileo.Mode = Gogh.Galileo.nMode - 1

    # Obtain the displacement of the measurement target point.
    Gogh.Galileo.Vector = constants.GALILEO_DISPLACEMENT_C
    succeed, (x, y, z) = Gogh.Galileo.GetVectorAtPoint_py(200, 0, 0)

    # Calculate the angle formed by the line segment
    # connecting the bending origin (100, 0) and the
    # deformed point with the X-axis.
    bending_point = np.array((100, 0))
    bended_point = np.array((200 + 1000 * x.Real, 1000 * z.Real))
    dx, dz = bended_point - bending_point
    degree = np.arctan2(-dz, dx)

    return degree * 360 / (2*np.pi)  # unit: degree


if __name__ == '__main__':

    # Initialize the numerical optimization problem.
    # (determine the optimization method)
    opt = OptunaOptimizer(
        sampler_class=BoTorchSampler,
        sampler_kwargs=dict(
            n_startup_trials=3,
        )
    )

    # Initialize the FEMOpt object.
    # (establish connection between the optimization problem and Femtet)
    femopt = FEMOpt(opt=opt)

    # Add design variables to the optimization problem.
    # (Specify the variables registered in the femprj file.)
    femopt.add_parameter("rot", 90, lower_bound=80, upper_bound=100)

    # Add the objective function to the optimization problem.
    # The target bending angle is 90 degrees.
    femopt.add_objective(bending, name='final angle (degree)', direction=90)

    # Run optimization.
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=10)
