"""A sample to implement constrained optimization.

This section describes the types of constraints and
the steps to run optimization on models that require them.

"""

from optuna_integration import BoTorchSampler
from pyfemtet.opt import FEMOpt, OptunaOptimizer


def mises_stress(Femtet):
    """Calculate the von Mises stress as the objective function.

    This function is called automatically by the FEMOpt
    object while the optimization is running.

    Args:
        Femtet: When defining an objective or constraint
            function using PyFemtet, the first argument
            must take a Femtet instance.

    Returns:
        float: A single float representing the expression value you want to constrain.
    """
    return Femtet.Gogh.Galileo.GetMaxStress_py()[2]


def radius_diff(Femtet, opt):
    """Calculate the difference between the outer and inner radii of the pipe.

    This constraint is called to ensure that the
    inner radius of the pipe does not exceed the
    outer radius while the optimization is running.

    Note:
        If you are using BoTorchSampler of OptunaOptimizer
        and use strict constraints, be aware that accessing
        the Femtet can be very slow, as it requires repeated
        calculations to propose parameters.
        We recommend that you do not access the Femtet,
        but rather get the parameters and perform the
        calculations via the Optimizer object, as in this
        function example.

        NOT recommended::

            p = Femtet.GetVariableValue('p')

        instead, use optimizer::

            params = opt.get_parameter()
            p = params['p']

    Args:
        Femtet: When defining an objective or constraint
            function using PyFemtet, the first argument
            must take a Femtet instance.
        opt: This object allows you to obtain the outer
            radius and inner radius values without going
            through Femtet.
    """
    params = opt.get_parameter()
    internal_r = params['internal_r']
    external_r = params['external_r']
    return external_r - internal_r


if __name__ == '__main__':
    # Setup optimization method
    opt = OptunaOptimizer(
        sampler_class=BoTorchSampler,
        sampler_kwargs=dict(
            n_startup_trials=3,  # The first three samples are randomly sampled.
        )
    )
    femopt = FEMOpt(opt=opt)

    # Add parameters
    femopt.add_parameter("external_r", 10, lower_bound=0.1, upper_bound=10)
    femopt.add_parameter("internal_r", 5, lower_bound=0.1, upper_bound=10)

    # Add the strict constraint not to exceed the
    # outer radius while the optimization is running.
    femopt.add_constraint(
        radius_diff,  # Constraint function (returns external radius - internal radius).
        name='wall thickness',  # You can name the function anything you want.
        lower_bound=1,  # Lower bound of constraint function (set minimum wall thickness is 1).
        args=(femopt.opt,)  # Additional arguments passed to the function.
    )

    # Add the objective
    femopt.add_objective(mises_stress, name='Mises Stress')

    # Run optimization.
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=10)
