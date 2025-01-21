import os

import shutil
import numpy as np
from optuna.samplers import RandomSampler, TPESampler

from pyfemtet.opt import FEMOpt, NoFEM, OptunaOptimizer
from pyfemtet.opt.interface import PoFBoTorchInterface
from pyfemtet.core import SolveError

import pytest

training_data_path = os.path.join(
    os.path.dirname(__file__),
    'training_data.reccsv'
)

optimized_data_path = os.path.join(
    os.path.dirname(__file__),
    'optimized_data.reccsv'
)

optimized_data_path_2 = os.path.join(
    os.path.dirname(__file__),
    'optimized_data_2.reccsv'
)


def intermediate_objectives(opt: OptunaOptimizer):
    values = opt.get_parameter(format='values')
    if values[0] <= 0:
        raise SolveError
    return values


def create_training_data():

    # set RECORD_MODE or not
    csv_path = training_data_path.replace('.reccsv', '.csv')
    if os.path.exists(training_data_path):
        RECORD_MODE = False
        if os.path.exists(csv_path):
            os.remove(csv_path)
        db_path = csv_path.replace('.csv', '.db')
        if os.path.exists(db_path):
            os.remove(db_path)
    else:
        RECORD_MODE = True

    fem = NoFEM()
    opt = OptunaOptimizer(sampler_class=RandomSampler)
    femopt = FEMOpt(opt=opt, fem=fem, history_path=csv_path)

    femopt.add_parameter('x', -1, -1, 1)
    femopt.add_parameter('y', -1, -1, 1)
    femopt.add_objectives(intermediate_objectives, names='obj', n_return=2, args=(opt,))
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=30, confirm_before_exit=False)

    if RECORD_MODE:
        os.rename(csv_path, training_data_path)
    else:
        from pyfemtet.opt._test_utils.record_history import is_equal_result
        is_equal_result(csv_path, training_data_path)
        print('Test `create_training_data` passed!')


def true_objective(fem: PoFBoTorchInterface):
    intermediate_objective_values = np.array(tuple(fem.obj.values()))
    return np.linalg.norm(intermediate_objective_values)


def optimize_with_surrogate():

    # set RECORD_MODE or not
    csv_path = optimized_data_path.replace('.reccsv', '.csv')
    if os.path.exists(optimized_data_path):
        RECORD_MODE = False
        if os.path.exists(csv_path):
            os.remove(csv_path)
        db_path = csv_path.replace('.csv', '.db')
        if os.path.exists(db_path):
            os.remove(db_path)
    else:
        RECORD_MODE = True

    # create training data from reccsv
    training_csv_path = training_data_path.replace('.reccsv', '.csv')
    if os.path.exists(training_csv_path):
        os.remove(training_csv_path)
    shutil.copy(training_data_path, training_csv_path)

    fem = PoFBoTorchInterface(history_path=training_csv_path, override_objective=False)
    opt = OptunaOptimizer(sampler_class=TPESampler)
    femopt = FEMOpt(opt=opt, fem=fem, history_path=csv_path)

    femopt.add_parameter('x', -1, -1, 1)
    femopt.add_parameter('y', -1, -1, 1)
    femopt.add_objective(true_objective, name='true_obj', args=(fem,))
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=30, confirm_before_exit=False)

    if RECORD_MODE:
        os.rename(csv_path, optimized_data_path)
    else:
        from pyfemtet.opt._test_utils.record_history import is_equal_result
        is_equal_result(csv_path, optimized_data_path)
        print('Test `optimize_with_surrogate` passed!')


def optimize_with_surrogate_with_override():

    # set RECORD_MODE or not
    csv_path = optimized_data_path_2.replace('.reccsv', '.csv')
    if os.path.exists(optimized_data_path_2):
        RECORD_MODE = False
        if os.path.exists(csv_path):
            os.remove(csv_path)
        db_path = csv_path.replace('.csv', '.db')
        if os.path.exists(db_path):
            os.remove(db_path)
    else:
        RECORD_MODE = True

    # create training data from reccsv
    training_csv_path = training_data_path.replace('.reccsv', '.csv')
    if os.path.exists(training_csv_path):
        os.remove(training_csv_path)
    shutil.copy(training_data_path, training_csv_path)

    fem = PoFBoTorchInterface(history_path=training_csv_path)
    opt = OptunaOptimizer(sampler_class=TPESampler)
    femopt = FEMOpt(opt=opt, fem=fem, history_path=csv_path)

    femopt.add_parameter('x', -1, -1, 1)
    femopt.add_parameter('y', -1, -1, 1)
    femopt.add_objective(name='obj_0', direction='maximize')
    femopt.add_objective(name='obj_1', direction='maximize')
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=30, confirm_before_exit=False)

    if RECORD_MODE:
        os.rename(csv_path, optimized_data_path_2)
    else:
        from pyfemtet.opt._test_utils.record_history import is_equal_result
        is_equal_result(csv_path, optimized_data_path_2)
        print('Test `optimize_with_surrogate_with_override` passed!')


@pytest.mark.nofem
def test_surrogate():
    create_training_data()
    optimize_with_surrogate()
    optimize_with_surrogate_with_override()


if __name__ == '__main__':
    create_training_data()
    optimize_with_surrogate()
    optimize_with_surrogate_with_override()

