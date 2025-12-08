"""dynamic_bounds.py"""

from pyfemtet.opt import FEMOpt, OptunaOptimizer
from pyfemtet.opt.optimizer import PoFBoTorchSampler


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


def dynamic_bounds_of_internal_r(opt):
    params = opt.get_variables()
    params['external_r']


def main():
    # Setup optimization method
    opt = OptunaOptimizer(
        sampler_class=PoFBoTorchSampler,
        sampler_kwargs=dict(
            n_startup_trials=3,  # The first three samples are randomly sampled.
        )
    )
    femopt = FEMOpt(opt=opt)

    # Add parameters
    femopt.add_parameter("external_r", 10, lower_bound=0.1, upper_bound=10)
    femopt.add_parameter("internal_r", 5, lower_bound=0.1, upper_bound=10)

    # Add the objective
    femopt.add_objective(fun=mises_stress, name='Mises Stress')

    # Run optimization.
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=10)


if __name__ == '__main__':
    main()
