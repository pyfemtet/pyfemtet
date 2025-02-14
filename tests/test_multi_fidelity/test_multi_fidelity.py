import os
from operator import index

from numpy import cos, pi
import numpy as np
from pyfemtet.opt import FEMOpt, NoFEM, OptunaOptimizer, History
from pyfemtet.opt.optimizer import PoFBoTorchSampler

from pyfemtet.opt._femopt import SubFidelity
from pyfemtet.opt.optimizer._optuna._multi_fidelity_sampler import MultiFidelityPoFBoTorchSampler

import pytest

N_STARTUP_TRIALS = 2
N_ADDITIONAL_TRIALS = 3
DIM = 9
OBJ_DIM = 3
OFFSET = 0. * pi
FIDELITY = 0.5

NEG_CON = False


def main_model(opt_: OptunaOptimizer):
    print('===== main model =====')
    x = opt_.get_parameter('values')
    return (1 + cos(x)).sum()


def main_model2(opt_: OptunaOptimizer):
    print('===== main2 model =====')
    x = opt_.get_parameter('values')
    return (1 + cos(x - pi / 2)).sum()


def sub_model(opt_: OptunaOptimizer):
    print('===== sub model =====')
    x = opt_.get_parameter('values')
    return ((x - pi - OFFSET) ** 2).sum() * 10 + 10


def sub_model2(opt_: OptunaOptimizer):
    print('===== sub2 model =====')
    x = opt_.get_parameter('values')
    return ((x - pi / 2 - pi - OFFSET) ** 2).sum() * 10 + 10


def should_solve_main(x: np.ndarray, history: History):
    n_main_solved = len(history.get_filtered_df([history.OptTrialState.succeeded]))
    n_all_solved = len(history.get_filtered_df([history.OptTrialState.succeeded, history.OptTrialState.skipped]))
    if (n_all_solved > N_STARTUP_TRIALS) and (n_all_solved % DIM * 2 == 0):
        return True
    else:
        return False


if __name__ == '__main__':

    os.chdir(os.path.dirname(__file__))

    fem = NoFEM()
    opt = OptunaOptimizer(
        sampler_class=MultiFidelityPoFBoTorchSampler if not NEG_CON else PoFBoTorchSampler,
        sampler_kwargs=dict(
            n_startup_trials=N_STARTUP_TRIALS,
        )
    )

    obj_name = 'sum of 1 + cos(x)'

    femopt = FEMOpt(
        fem,
        opt,
        # history_path=f'multi-fid-history-{DIM}dim.csv'
    )
    for i in range(DIM):
        femopt.add_parameter(f'x{i}', 0.5 * pi, 0, 2 * pi)

    femopt.add_objective(main_model, obj_name, args=(opt,), direction='minimize')
    if OBJ_DIM >= 2:
        femopt.add_objective(main_model2, obj_name + '2', args=(opt,), direction='minimize')
        if OBJ_DIM >= 3:
            femopt.add_objective(main_model, obj_name + '3', args=(opt,), direction='minimize')


    if not NEG_CON:
        femopt.opt.set_solve_condition(should_solve_main)
        sub_fem = NoFEM()
        sub_fidelity = SubFidelity(
            sub_fem,
            fidelity=FIDELITY,
        )
        sub_fidelity.add_objective(sub_model, obj_name, args=(opt,))
        if OBJ_DIM >= 2:
            sub_fidelity.add_objective(sub_model2, obj_name + '2', args=(opt,))
            if OBJ_DIM >= 3:
                sub_fidelity.add_objective(sub_model, obj_name + '3', args=(opt,))
        femopt.add_sub_fidelity(sub_fidelity)

    femopt.set_random_seed(42)
    femopt.optimize(
        N_STARTUP_TRIALS + N_ADDITIONAL_TRIALS,
        confirm_before_exit=True,
        n_parallel=1,
    )


if False:
    import pandas as pd
    def c_(tensor):
        df = pd.DataFrame(tensor)
        df.to_clipboard(index=None, header=None)
