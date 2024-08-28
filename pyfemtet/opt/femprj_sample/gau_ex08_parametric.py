"""Single-objective optimization: Self-inductance of a finite-length helical coil.

Using Femtet's magnetic field analysis solver, design to achieve
the target value for the self-inductance of a finite-length helical coil.

Corresponding project: gau_ex08_parametric.femprj
"""
from optuna.integration.botorch import BoTorchSampler
from pyfemtet.opt import FEMOpt, OptunaOptimizer


def inductance(Femtet):
    """Obtain the self-inductance.

    Note:
        The objective or constraint function should take Femtet
        as its first argument and return a float as the output.

    Params:
        Femtet: This is an instance for manipulating Femtet with macros. For detailed information, please refer to "Femtet Macro Help".

    Returns:
        float: Self-inductance.
    """
    Gogh = Femtet.Gogh

    coil_name = Gogh.Gauss.GetCoilList()[0]
    return Gogh.Gauss.GetL(coil_name, coil_name)  # unit: F


if __name__ == '__main__':

    # Initialize the numerical optimization problem.
    # (determine the optimization method)
    opt = OptunaOptimizer(
        sampler_class=BoTorchSampler,
        sampler_kwargs=dict(
            n_startup_trials=5,
        )
    )

    # Initialize the FEMOpt object.
    # (establish connection between the optimization problem and Femtet)
    femopt = FEMOpt(opt=opt)

    # Add design variables to the optimization problem.
    # (Specify the variables registered in the femprj file.)
    femopt.add_parameter("helical_pitch", 6, lower_bound=4.2, upper_bound=8)
    femopt.add_parameter("coil_radius", 10, lower_bound=1, upper_bound=10)
    femopt.add_parameter("n_turns", 5, lower_bound=1, upper_bound=5)

    # Add the objective function to the optimization problem.
    # The target inductance is 0.1 uF.
    femopt.add_objective(inductance, name='self-inductance (F)', direction=1e-7)

    # Run optimization.
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=20)
