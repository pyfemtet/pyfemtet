import os
from time import sleep

from optuna.samplers import RandomSampler

from pyfemtet.opt import FEMOpt, FemtetInterface, OptunaOptimizer


def get_res_freq(Femtet):
    Galileo = Femtet.Gogh.Galileo
    Galileo.Mode = 0
    sleep(0.01)
    return Galileo.GetFreq().Real


if __name__ == '__main__':

    os.chdir(os.path.dirname(__file__))

    # Connect to Femtet.
    fem = FemtetInterface(
        femprj_path='gal_ex13_parametric.femprj',
    )

    # Initialize the optimization object.
    # However, this script is not for optimization;
    # instead, it is for creating training data.
    # Therefore, we will use Optuna's random sampling
    # class to select the design variables.
    opt = OptunaOptimizer(
        sampler_class=RandomSampler,
    )

    # We will set up the FEMOpt object. To refer to
    # history_path in the optimization script, we will
    # specify a clear CSV file name.
    femopt = FEMOpt(
        fem=fem,
        opt=opt,
        history_path='training_data.csv'
    )

    # Set the design variables.
    femopt.add_parameter('length', 0.1, 0.02, 0.2)
    femopt.add_parameter('width', 0.01, 0.001, 0.02)
    femopt.add_parameter('base_radius', 0.008, 0.006, 0.01)

    # Set the objective function. Since this is random
    # sampling, specifying the direction does not affect
    # the sampling.
    femopt.add_objective(fun=get_res_freq, name='First Resonant Frequency (Hz)')

    # Create the training data.
    # If no termination condition is specified,
    # it will continue creating training data until
    # manually stopped.
    femopt.set_random_seed(42)
    femopt.optimize(
        # n_trials=100
    )
