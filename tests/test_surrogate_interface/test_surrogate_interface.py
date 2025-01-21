import os

import numpy as np
from optuna.samplers import RandomSampler, TPESampler

from pyfemtet.opt import FEMOpt, NoFEM, OptunaOptimizer
from pyfemtet.opt.interface import PoFBoTorchInterface
from pyfemtet.core import SolveError

training_data_path = os.path.join(
    os.path.dirname(__file__),
    'training_data.csv'
)

optimized_data_path = os.path.join(
    os.path.dirname(__file__),
    'optimized_data.csv'
)

optimized_data_path_2 = os.path.join(
    os.path.dirname(__file__),
    'optimized_data_2.csv'
)


def intermediate_objectives(opt: OptunaOptimizer):
    values = opt.get_parameter(format='values')
    if values[0] <= 0:
        raise SolveError
    return values


def create_training_data():

    if os.path.exists(training_data_path):
        os.remove(training_data_path)

    db_path = training_data_path.replace('.csv', '.db')
    if os.path.exists(db_path):
        os.remove(db_path)

    fem = NoFEM()
    opt = OptunaOptimizer(sampler_class=RandomSampler)
    femopt = FEMOpt(opt=opt, fem=fem, history_path=training_data_path)

    femopt.add_parameter('x', -1, -1, 1)
    femopt.add_parameter('y', -1, -1, 1)
    femopt.add_objectives(intermediate_objectives, names='obj', n_return=2, args=(opt,))

    femopt.optimize(n_trials=30, confirm_before_exit=False)


def true_objective(fem: PoFBoTorchInterface):
    intermediate_objective_values = np.array(tuple(fem.obj.values()))
    return np.linalg.norm(intermediate_objective_values)


def optimize_with_surrogate():

    if os.path.exists(optimized_data_path):
        os.remove(optimized_data_path)

    db_path = optimized_data_path.replace('.csv', '.db')
    if os.path.exists(db_path):
        os.remove(db_path)

    fem = PoFBoTorchInterface(history_path=training_data_path, override_objective=False)
    opt = OptunaOptimizer(sampler_class=TPESampler)
    femopt = FEMOpt(opt=opt, fem=fem, history_path=optimized_data_path)

    femopt.add_parameter('x', -1, -1, 1)
    femopt.add_parameter('y', -1, -1, 1)
    femopt.add_objective(true_objective, name='true_obj', args=(fem,))

    femopt.optimize(n_trials=30, confirm_before_exit=False)


def optimize_with_surrogate_with_override():
    if os.path.exists(optimized_data_path_2):
        os.remove(optimized_data_path_2)

    db_path = optimized_data_path_2.replace('.csv', '.db')
    if os.path.exists(db_path):
        os.remove(db_path)

    fem = PoFBoTorchInterface(history_path=training_data_path)
    opt = OptunaOptimizer(sampler_class=TPESampler)
    femopt = FEMOpt(opt=opt, fem=fem, history_path=optimized_data_path_2)

    femopt.add_parameter('x', -1, -1, 1)
    femopt.add_parameter('y', -1, -1, 1)
    femopt.add_objective(name='name_0', direction='maximize')
    femopt.add_objective(name='name_1', direction='maximize')

    femopt.optimize(n_trials=30, confirm_before_exit=False)


if __name__ == '__main__':
    # create_training_data()
    # optimize_with_surrogate()
    optimize_with_surrogate_with_override()
