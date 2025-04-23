"""External CAD (SOLIDWORKS) Integration

Using Femtet's stress analysis solver and Dassault Systemes' CAD software SOLIDWORKS,
design a lightweight and high-strength H-shaped beam.

As a preliminary step, please perform the following procedures:
- Install SOLIDWORKS
- Create a C:\temp folder
    - Note: SOLIDWORKS will save a .x_t file in this folder.
- Place the following files in the same folder:
    - cad_ex01_SW.py (this file)
    - cad_ex01_SW.SLDPRT
    - cad_ex01_SW.femprj
"""

import os

from win32com.client import constants

from pyfemtet.opt import FEMOpt
from pyfemtet.opt.interface import FemtetWithSolidworksInterface
from pyfemtet.core import ModelError


here, me = os.path.split(__file__)
os.chdir(here)


def von_mises(Femtet):
    """Obtain the maximum von Mises stress of the model.

    Note:
        The objective or constraint function should take Femtet
        as its first argument and return a float as the output.

    Warning:
        CAD integration may assign boundary conditions to unintended locations.

        In this example, if the boundary conditions are assigned as intended,
        the maximum z displacement is always negative.
        If the maximum displacement is not negative, it is assumed that
        boundary condition assignment has failed.
        Then this function raises a ModelError.

        If a ModelError, MeshError, or SolveError occurs during optimization,
        the optimization process considers the attempt a failure and skips to
        the next trial.
    """

    # Simple check for the correctness of boundary conditions.
    dx, dy, dz = Femtet.Gogh.Galileo.GetMaxDisplacement_py()
    if dz >= 0:
        raise ModelError('Assigning unintended boundary conditions.')

    # Von Mises stress calculation.
    Gogh = Femtet.Gogh
    Gogh.Galileo.Potential = constants.GALILEO_VON_MISES_C
    succeed, (x, y, z), mises = Gogh.Galileo.GetMAXPotentialPoint_py(constants.CMPX_REAL_C)

    return mises


def mass(Femtet):
    """Obtain model mass."""
    return Femtet.Gogh.Galileo.GetMass('H_beam')


def C_minus_B(Femtet, opt):
    """Calculate the difference between C and B dimensions.

    Another example uses the following snippet to access design variables:

        A = Femtet.GetVariableValue('A')
    
    However, when performing CAD integration, this method does not work
    because the variables are not set in the .femprj file.

    In CAD integration, design variables are obtained in the following way.

        # How to obtain a dictionary with the variable names of parameters
        # added by add_parameter() as keys.
        params: dict = opt.get_parameter()
        A = params['A']

    Or

        # How to obtain an array of values of parameters added in the order
        # by add_parameter().
        values: np.ndarray = opt.get_parameter('values')
        A, B, C = values

    Objective functions and constraint functions can take arbitrary variables
    after the first argument.
    The FEMOpt member variable `opt` has a method called get_parameter().
    This method allows you to retrieve design variables added by add_parameter().
    By taking `opt` as the second argument, you can execute get_parameter()
    within the objective or constraint function to retrieve design variables.
    """
    A, B, C = opt.get_parameter('values')
    return C - B


if __name__ == '__main__':

    # Initialize NX-Femtet integration object.
    # At this point, Python is connected to the Femtet.
    fem = FemtetWithSolidworksInterface(
        sldprt_path='cad_ex01_SW.SLDPRT',
        open_result_with_gui=False,  # To calculate von Mises stress, set this argument to False. See Femtet Macro Help.
    )

    # Initialize the FEMOpt object.
    # (establish connection between the optimization problem and Femtet)
    femopt = FEMOpt(fem=fem)

    # Add design variables to the optimization problem.
    # (Specify the variables registered in the femprj file.)
    femopt.add_parameter('A', 10, lower_bound=1, upper_bound=59)
    femopt.add_parameter('B', 10, lower_bound=1, upper_bound=40)
    femopt.add_parameter('C', 20, lower_bound=5, upper_bound=59)

    # Add the constraint function to the optimization problem.
    femopt.add_constraint(C_minus_B, 'C>B', lower_bound=1, args=femopt.opt)

    # Add the objective function to the optimization problem.
    femopt.add_objective(von_mises, name='von Mises (Pa)')
    femopt.add_objective(mass, name='mass (kg)')

    # Run optimization.
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=20)
