import os
from pathlib import Path

from numpy import sin, cos, pi

from pyfemtet.opt import FEMOpt, NoFEM, OptunaOptimizer
from pyfemtet.core import SolveError
from pyfemtet.opt.optimizer import PoFBoTorchSampler, PoFConfig

import pytest


COEF = 5. / 12. * pi


def x(opt):
    params = opt.get_parameter()
    r = params['r']
    theta = params['theta']

    # if (constraint_1(opt) < 0) or (constraint_2(opt) < 0):
    if constraint_2(opt) < 0:
        raise SolveError

    return r * cos(theta)


def y(opt):
    params = opt.get_parameter()
    r = params['r']
    theta = params['theta']
    return r * sin(theta)


def constraint_1(opt):
    params = opt.get_parameter()
    r = params['r']
    theta = params['theta']
    return 2 * COEF * r - theta  # > 0


def constraint_2(opt):
    params = opt.get_parameter()
    r = params['r']
    theta = params['theta']
    return theta - COEF * r  # > 0


@pytest.mark.nofem
def test_pof_basic():
    # ===== problem =====
    seed = 42
    n_trials = 10
    n_startup_trials = 5
    hv_reference = [0, 0.5]

    # reference が存在しないならば記録モード
    record_mode = not (Path(__file__).parent / 'pof_reference.csv').is_file()

    # 記録モードならば記録ヒストリを作る
    if record_mode:
        path = Path(__file__).parent / 'pof_reference.csv'

    # そうでなければテストヒストリを作る
    else:
        path = Path(__file__).parent / 'pof_test.csv'
        if os.path.exists(path):
            os.remove(path)
        path2 = Path(__file__).parent / 'pof_test.db'
        if os.path.exists(path2):
            os.remove(path2)


    # ===== configuration =====
    Sampler = PoFBoTorchSampler
    pof_config = PoFConfig()
    sampler_kwargs = dict(
        n_startup_trials=n_startup_trials,
        pof_config=pof_config,
    )

    # ===== main =====
    fem = NoFEM()
    opt = OptunaOptimizer(
        sampler_class=Sampler,
        sampler_kwargs=sampler_kwargs,
    )

    femopt = FEMOpt(
        fem=fem,
        opt=opt,
        history_path=path,
    )

    femopt.add_parameter('r', 0.5, 0, 1)
    femopt.add_parameter('theta', 1.5*COEF, 0, 2*pi)

    femopt.add_objective(x, direction='maximize', args=femopt.opt)
    femopt.add_objective(y, direction='maximize', args=femopt.opt)

    femopt.add_constraint(constraint_1, lower_bound=0, args=(femopt.opt,))

    femopt.set_random_seed(seed)
    femopt._hv_reference = hv_reference

    femopt.optimize(
        n_trials=n_trials,
        confirm_before_exit=False,
    )

    if record_mode:
        os.rename(
            path,
            str(path).replace('.csv', '.reccsv')
        )

    else:
        # ===== check =====
        import numpy as np
        import pandas as pd
        from pyfemtet._message.messages import encoding

        def simplify_df(path_):
            df_ = pd.read_csv(path_, encoding=encoding, header=2)
            pdf_ = pd.DataFrame()
            pdf_['r'] = df_['r']
            pdf_['theta'] = df_['theta']
            pdf_['obj_0'] = df_['obj_0']
            pdf_['obj_1'] = df_['obj_1']
            return pdf_.values

        ref = simplify_df(Path(__file__).parent / 'pof_reference.reccsv')
        dif = simplify_df(Path(__file__).parent / 'pof_test.csv')

        assert (np.abs(dif - ref) / ref).max() < 0.1, "前回結果から 10% 以上の乖離あり"
        print('PoF test Passed!')


if __name__ == '__main__':
    test_pof_basic()
