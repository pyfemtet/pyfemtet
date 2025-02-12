import os
from operator import index

from numpy import cos, pi
import numpy as np
from pyfemtet.opt import FEMOpt, NoFEM, OptunaOptimizer, History
from pyfemtet.opt.optimizer import PoFBoTorchSampler

from pyfemtet.opt._femopt import SubFidelity
from pyfemtet.opt._femopt_core import OptTrialState
from pyfemtet.opt.optimizer._optuna._multi_fidelity_sampler import MultiFidelityPoFBoTorchSampler

import pytest

N_STARTUP_TRIALS = 2
N_ADDITIONAL_TRIALS = 30
DIM = 9

OFFSET = 0. * pi
FIDELITY = 0.9

NEG_CON = False


def main_model(opt_: OptunaOptimizer):
    print('===== main model =====')
    x = opt_.get_parameter('values')
    return (1 + cos(x)).sum()


def sub_model(opt_: OptunaOptimizer):
    print('===== sub model =====')
    x = opt_.get_parameter('values')
    return ((x - pi - OFFSET) ** 2).sum()


def should_solve_main(x: np.ndarray, history: History):
    n_main_solved = len(history.get_filtered_df([OptTrialState.succeeded]))
    n_all_solved = len(history.get_filtered_df([OptTrialState.succeeded, OptTrialState.skipped]))
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

    obj_name = 'sun of 1 + cos(x)'

    femopt = FEMOpt(fem, opt)
    for i in range(DIM):
        femopt.add_parameter(f'x{i}', 0.5 * pi, 0, 2 * pi)
    femopt.add_objective(main_model, obj_name, args=(opt,), direction='minimize')

    if not NEG_CON:
        femopt.opt.set_solve_condition(should_solve_main)
        sub_fem = NoFEM()
        sub_fidelity = SubFidelity(
            sub_fem,
            fidelity=FIDELITY,
        )
        sub_fidelity.add_objective(sub_model, obj_name, args=(opt,))
        femopt.add_sub_fidelity(sub_fidelity)

    femopt.set_random_seed(42)
    femopt.optimize(
        N_STARTUP_TRIALS + N_ADDITIONAL_TRIALS,
        confirm_before_exit=True
    )


if False:
    import pandas as pd
    def c_(tensor):
        df = pd.DataFrame(tensor)
        df.to_clipboard(index=None, header=None)
