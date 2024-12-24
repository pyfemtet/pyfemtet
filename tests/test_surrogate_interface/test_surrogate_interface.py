import os

import numpy as np
from optuna.samplers import RandomSampler, TPESampler

from pyfemtet.opt import FEMOpt, NoFEM
from pyfemtet.opt.optimizer import OptunaOptimizer
from pyfemtet.opt.interface._singletaskgp import PoFBoTorchInterface

import pytest

os.chdir(os.path.dirname(__file__))

HISTORY_PATH = 'surrogate_training_data.csv'


def objective(opt: OptunaOptimizer):
    prm = opt.get_parameter()
    return (np.array(list(prm.values()))**2).sum()


def constraint(opt: OptunaOptimizer):
    prm = opt.get_parameter()
    return np.array(list(prm.values())).min()  # >0


def make_training_data():

    if os.path.exists(HISTORY_PATH):
        os.remove(HISTORY_PATH)
        os.remove(HISTORY_PATH.replace('.csv', '.db'))

    opt = OptunaOptimizer(sampler_class=RandomSampler)
    fem = NoFEM()
    femopt = FEMOpt(
        fem=fem,
        opt=opt,
        history_path=HISTORY_PATH,
    )

    femopt.add_parameter(f'x0', 1, -1, 1)
    femopt.add_parameter(f'x1', 1, -1, 1)

    # femopt.add_constraint(constraint, lower_bound=0, args=(opt,), strict=False)
    femopt.add_constraint(constraint, lower_bound=0, args=(opt,), strict=True)

    femopt.add_objective(objective, args=(opt,))

    femopt.set_random_seed(42)
    femopt.optimize(
        n_trials=2**4,
        confirm_before_exit=False,
    )


def use_surrogate_model():
    opt = OptunaOptimizer(sampler_class=TPESampler)
    fem = PoFBoTorchInterface(history_path=HISTORY_PATH)
    femopt = FEMOpt(fem=fem, opt=opt)

    femopt.add_parameter(f'x0', -1, -1, 1)
    femopt.add_fixed_parameter(f'x1', 0, -1, 1)

    femopt.add_objective(objective, direction=1., args=(opt,))

    femopt.set_random_seed(42)
    femopt.optimize(
        n_trials=30,
        confirm_before_exit=False,
        n_parallel=3
    )


@pytest.mark.nofem
def main():
    make_training_data()
    use_surrogate_model()


if __name__ == '__main__':
    main()
