"""Single-objective optimization: self-inductance of a finite length solenoid coil

Perform magnetic field analysis on gau_ex08_parametric.femprj
to find the dimensions of a finite length solenoid coil
that makes the self-inductance a specific value.
"""
from optuna.integration.botorch import BoTorchSampler
from pyfemtet.opt import FEMOpt, OptunaOptimizer


def inductance(Femtet):
    """Get the self-inductance.

    Note:
        The objective or constraint function
        must take a Femtet as its first argument
        and must return a single float.

    Params:
        Femtet: An instance for using Femtet macros. For more information, see "Femtet Macro Help / CFemtet Class".

    Returns:
        float: Self-inductance.
    """
    Gogh = Femtet.Gogh

    # Get inductance.
    cName = Gogh.Gauss.GetCoilList()[0]
    l = Gogh.Gauss.GetL(cName, cName)
    return l  # F


if __name__ == '__main__':

    # Define mathematical optimization object.
    opt = OptunaOptimizer(
        sampler_class=BoTorchSampler,
        sampler_kwargs=dict(
            n_startup_trials=5,
        )
    )

    # Define FEMOpt object (This process integrates mathematical optimization and FEM.).
    femopt = FEMOpt(opt=opt)

    # Add design variables (Use variable names set in Femtet) to the optimization problem.
    femopt.add_parameter("h", 3, lower_bound=1.5, upper_bound=6)
    femopt.add_parameter("r", 5, lower_bound=1, upper_bound=10)
    femopt.add_parameter("n", 3, lower_bound=1, upper_bound=5)

    # Add objective to the optimization problem.
    # The target inductance value is 0.1 uF.
    femopt.add_objective(inductance, name='self-inductance', direction=0.1e-06)

    # Run optimization.
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=20)
    femopt.terminate_all()
