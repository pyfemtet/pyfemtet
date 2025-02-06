import os
from time import sleep

from optuna.samplers import RandomSampler, NSGAIISampler, GPSampler, BaseSampler

from pyfemtet.opt import FEMOpt, FemtetInterface, OptunaOptimizer

os.chdir(os.path.dirname(__file__))


def get_res_freq(Femtet):
    Galileo = Femtet.Gogh.Galileo
    Galileo.Mode = 0
    sleep(0.01)
    return Galileo.GetFreq().Real


def main(n_trials, sampler_class: type[BaseSampler], sampler_kwargs: dict):
    """Main function

    length
    using different algorithms for each restarting.

    So this main function requires n_trials and sampler_class.

    Args:

        n_trials (int):
            How many additional succeeded trials
            to terminate optimization.

        sampler_class (type[optuna.samplers.BaseSampler]):
            The algorithm we use.

        sampler_kwargs (dict):
            The arguments of sampler.

    """


    # Connect to Femtet.
    fem = FemtetInterface(
        femprj_path='gal_ex13_parametric.femprj',
    )

    # Initialize the optimization object.
    opt = OptunaOptimizer(
        sampler_class=sampler_class,
        sampler_kwargs=sampler_kwargs,
    )

    # To restart, it is necessary to inform the new optimization program
    # about the history of the previous optimization.
    # By specifying a csv for the `history_path` argument of FEMOpt,
    # if it does not exist, a new csv file will be created,
    # and if it exists, optimization will continue from that csv file.
    #
    # Note:
    #   When restarting, the number and names of variables,
    #   as well as the number and names of objective functions
    #   and constraints must be consistent.
    #   However, you can change the bounds of variables,
    #   direction of objective functions, and content of constraints.
    #
    # Note:
    #   When using OptunaOptimizer, the .db file with the same name
    #   (in this case restarting-sample.db) that is saved along with
    #   csv is required to be  in the same folder as the csv file.
    femopt = FEMOpt(
        fem=fem,
        opt=opt,
        history_path='restarting-sample.csv'
    )

    # Set the design variables.
    femopt.add_parameter('length', 0.1, 0.02, 0.2)
    femopt.add_parameter('width', 0.01, 0.001, 0.02)
    femopt.add_parameter('base_radius', 0.008, 0.006, 0.01)

    # Set the objective function.
    femopt.add_objective(fun=get_res_freq, name='First Resonant Frequency (Hz)', direction=800)

    # Run optimization.
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=n_trials, confirm_before_exit=False)


if __name__ == '__main__':
    # First, we will perform 3 optimizations using the RandomSampler.
    main(3, RandomSampler, {})

    # Next, we will perform 3 optimizations using the NSGAIISampler.
    main(3, NSGAIISampler, {})

    # Finally, we will perform 3 optimizations using the GPSampler.
    main(3, GPSampler, {'n_startup_trials': 0, 'deterministic_objective': True})

    # After this program ends, you can continue further with
    # restarting-sample.csv and the .db file.
