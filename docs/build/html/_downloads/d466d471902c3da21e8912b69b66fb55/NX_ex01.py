"""External CAD (NX) Integration

This script performs parametric optimization
using a project that has imported an external CAD model
that has been parametrically modeled.

Please perform the following steps in preparation.
    - Install NX.
    - Place NX_ex01.prt, NX_ex01.femprj and .py file in the same folder.
    - Create C:\\temp folder on your disk.
        Note: NX exports .x_t files during optimization.
"""

import os
from pyfemtet.opt import FEMOpt, OptunaOptimizer
from pyfemtet.opt.interface import FemtetWithNXInterface
from pyfemtet.core import ModelError


here, me = os.path.split(__file__)
os.chdir(here)


def disp(Femtet):
    """Obtain the maximum displacement in the Z direction from the analysis results.

    Note:
        The objective or constraint function
        must take a Femtet as its first argument
        and must return a single float.

    Warning:
        CAD integration may assign boundary conditions to unintended locations.

        In this example, if the boundary conditions are assigned as intended,
        the maximum displacement is always negative.
        If the maximum displacement is positive,
        it is assumed that boundary condition assignment has failed
        and a ModelError is raised.
        
        If a ModelError, MeshError, or SolveError is raised during an optimization,
        the Optimizer considers the trial a failure
        and skips it to the next trial.
    """
    _, _, ret = Femtet.Gogh.Galileo.GetMaxDisplacement_py()

    if ret >= 0:
        raise ModelError('The boundary condition assignment is incorrect.')
    return ret


def volume(Femtet):
    """Obtain the volume."""
    _, ret = Femtet.Gogh.CalcVolume_py([0])
    return ret


def C_minus_B(_, opt):
    """Calculate the difference between C and B dimensions.

    The constraint function must take a Femtet instance as its first argument,
    but this example does not use it.

    Another example uses the following snippet to access design variables:

        A = Femtet.GetVariableValue('A')
    
    However, when performing CAD integration,
    this method does not work because the variables are not set in the .femprj file.

    In CAD integration, design variables are obtained in the following way.
    
    The objective function and constraint function can take any variable after the first argument.
    The member variable opt of FEMOpt has a method called get_parameter().
    This method can retrieve design variables added by add_parameter().
    By taking opt as the second argument, you can execute get_parameter()
    in the objective function or constraint function to obtain the design variables.
    
    """
    A, B, C = opt.get_parameter('values')
    return C - B


if __name__ == '__main__':

    # Define NX-Femtet integration object.
    # At this point, Python is connected to the running Femtet.
    fem = FemtetWithNXInterface(
        prt_path='NX_ex01.prt',
        femprj_path='NX_ex01.femprj',
    )

    # Define mathematical optimization object.
    opt = OptunaOptimizer(
        sampler_kwargs=dict(
            n_startup_trials=5,
        )
    )

    # Define FEMOpt object (This process integrates mathematical optimization and FEM.).
    femopt = FEMOpt(fem=fem, opt=opt)

    # Add design variables (Use variable names set in NX) to the optimization problem.
    femopt.add_parameter('A', 10, lower_bound=1, upper_bound=59)
    femopt.add_parameter('B', 10, lower_bound=1, upper_bound=40)
    femopt.add_parameter('C', 20, lower_bound=5, upper_bound=59)

    # Add constraint to the optimization problem.
    femopt.add_constraint(C_minus_B, 'C>B', lower_bound=1, args=femopt.opt)

    # Add objective to the optimization problem.
    femopt.add_objective(disp, name='displacement', direction=0)
    femopt.add_objective(volume, name='volume', direction='minimize')

    # Run optimization.
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=20)
    femopt.terminate_all()
