import os

from optuna.samplers import RandomSampler

from pyfemtet.opt import FEMOpt, FemtetInterface, OptunaOptimizer


if __name__ == '__main__':

    os.chdir(os.path.dirname(__file__))

    fem = FemtetInterface(
        femprj_path='gal_ex13_parametric.femprj',
        parametric_output_indexes_use_as_objective={
            0: 'minimize',
        },
    )

    opt = OptunaOptimizer(
        sampler_class=RandomSampler,
    )

    femopt = FEMOpt(
        fem=fem,
        opt=opt,
        history_path='241225_training_data.csv'
    )

    femopt.add_parameter('length', 0.1, 0.02, 0.2)
    femopt.add_parameter('width', 0.01, 0.001, 0.02)
    femopt.add_parameter('base_radius', 0.008, 0.006, 0.01)

    femopt.optimize(
        # n_trials=30,
    )
