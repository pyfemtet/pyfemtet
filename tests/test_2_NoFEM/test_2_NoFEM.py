import sys
from subprocess import run

import os
from time import sleep
import numpy as np
import optuna.integration
import pandas as pd
from pyfemtet.opt import FEMOpt, OptunaOptimizer, NoFEM


here, me = os.path.split(__file__)
random_max_sleep_sec = 0.1
min_sleep_sec = 0.1
record = False


def objective_x(opt):
    sleep(min_sleep_sec+np.random.rand()*random_max_sleep_sec)
    r, theta, fai = opt.get_parameter('values')
    return r * np.cos(theta) * np.cos(fai)


def objective_y(opt):
    r, theta, fai = opt.get_parameter('values')
    return r * np.cos(theta) * np.sin(fai)


def objective_z(opt):
    r, theta, fai = opt.get_parameter('values')
    return r * np.sin(theta)


def constraint_y(opt):
    y = objective_y(opt)
    return y


def constraint_z(opt):
    z = objective_z(opt)
    return z


def test_2_NoFEM_random_seed():
    """
    テストしたい状況
        一通りの機能（ランダムシードの機能）
    """
    os.chdir(here)
    csv_path = me + '.csv'

    fem = NoFEM()
    opt = OptunaOptimizer(add_init_method='LHS')
    opt.add_init_parameter([0, 0, 0])
    d = dict()
    d['r（半径）'] = 1
    d['theta（角度1）'] = 0
    d['fai（角度2）'] = 0
    opt.add_init_parameter(d)
    femopt = FEMOpt(fem, opt)
    femopt.set_random_seed(42)
    femopt.add_parameter('r（半径）', .5, 0, 1)
    femopt.add_parameter('theta（角度1）', np.pi/3, -np.pi/2, np.pi/2)  # 空間上で xy 平面となす角
    femopt.add_parameter('fai（角度2）', (7/6)*np.pi, 0, 2*np.pi)  # xy 平面上で x 軸となす角
    femopt.add_objective(objective_x, 'x(mm)', args=opt)
    femopt.add_objective(objective_y, 'y(mm)', args=opt)
    femopt.add_objective(objective_z, 'z(mm)', args=opt, direction=-1)
    femopt.add_constraint(constraint_y, 'y<=0', upper_bound=0, args=opt)  # 上書き
    femopt.add_constraint(constraint_z, 'z<=0', upper_bound=0, args=opt, strict=False)
    femopt.optimize(n_trials=30)

    if record:
        femopt.history.actor_data.to_csv(
            csv_path,
            encoding='shift-jis',
            index=None
        )
        femopt.terminate_all()

    else:
        # データの取得
        ref_df = pd.read_csv(csv_path, encoding='shift-jis').replace(np.nan, None).select_dtypes(include='number')
        def_df = femopt.history.actor_data.copy().select_dtypes(include='number')

        femopt.terminate_all()

        assert np.sum(np.abs(def_df.values - ref_df.values)) < 0.0001, 'seed 設定が反映されていません。'


def sub():
    csv_path = 'test_2_2_restart.csv'
    fem = NoFEM()
    opt = OptunaOptimizer()
    femopt = FEMOpt(fem, opt, history_path=csv_path)
    femopt.set_random_seed(42)
    femopt.add_parameter('r（半径）', .5, 0, 1)
    femopt.add_parameter('theta（角度1）', np.pi/3, -np.pi/2, np.pi/2)  # 空間上で xy 平面となす角
    femopt.add_parameter('fai（角度2）', (7/6)*np.pi, 0, 2*np.pi)  # xy 平面上で x 軸となす角
    femopt.add_objective(objective_x, 'x(mm)', args=femopt.opt)
    femopt.add_objective(objective_y, 'y(mm)', args=femopt.opt)
    femopt.add_objective(objective_z, 'z(mm)', args=femopt.opt, direction=-1)
    femopt.add_constraint(constraint_y, 'y<=0', upper_bound=0, args=femopt.opt)  # 上書き
    femopt.add_constraint(constraint_z, 'z<=0', upper_bound=0, args=femopt.opt, strict=False)
    femopt.optimize(n_trials=5)
    femopt.terminate_all()


def test_2_2_restart():
    """
    テストしたい状況
        restart
    """
    os.chdir(here)
    if os.path.exists('test_2_2_restart.csv'):
        os.remove('test_2_2_restart.csv')
        os.remove('test_2_2_restart.db')
        os.remove('test_2_2_restart.uilog')
    sub()
    sub()
    df = pd.read_csv('test_2_2_restart.csv', encoding='cp932', header=2)
    assert len(df) == 10


def parameter_constraint(theta, fai):
    return min(theta, fai) - np.pi/6  # >= 0 (fai, theta > 0.52...)


def simple():
    """シンプルな動作確認"""

    fem = NoFEM()
    opt = OptunaOptimizer()

    femopt = FEMOpt(fem, opt, scheduler_address=None)

    femopt.add_parameter('r', .5, 0, 1)
    femopt.add_parameter('theta', np.pi/3, -np.pi/2, np.pi/2)  # 空間上で xy 平面となす角
    femopt.add_parameter('fai', (7/6)*np.pi, 0, 2*np.pi)  # xy 平面上で x 軸となす角
    femopt.add_objective(objective_x, 'x', args=femopt.opt)
    femopt.add_objective(objective_y, 'y座標', args=femopt.opt)
    # femopt.add_objective(objective_z, 'z', args=femopt.opt)
    # femopt.add_constraint(objective_z, 'z<=0', upper_bound=0, args=femopt.opt)
    # femopt.add_constraint(objective_z, 'z<=0', upper_bound=0, args=femopt.opt, strict=False)

    femopt.set_random_seed(42)
    femopt.optimize(n_trials=30, n_parallel=1, wait_setup=True)
    # input('enter to quit...')
    femopt.terminate_all()



from pyfemtet._test_util import SuperSphere

s = SuperSphere(2)


def hypersphere_x(opt, idx):
    radius, *fai = opt.get_parameter('values')
    return s.x(radius, *fai)[idx]


def constraint(fai1, fai2, fai3, fai4, fai5, fai6):
    return max(fai1, fai2, fai3, fai4, fai5, fai6)


def constraint2(fai1, fai2):
    return max(fai1, fai2)


def constraint3(fai7, fai8):
    return min(fai7, fai8)


def constraint4(fai3, fai4, fai5, fai6):
    return max(fai3, fai4, fai5, fai6)


def constraint5(r, fai1):
    return fai1 - np.pi * r / 2  # >= 0


def constraint6(r, fai1):
    return np.pi * r - fai1  # >= 0


def parameter_constraint_test():
    fem = NoFEM()
    opt = OptunaOptimizer(sampler_class=optuna.integration.BoTorchSampler, sampler_kwargs=dict(n_startup_trials=5))

    femopt = FEMOpt(fem, opt, scheduler_address=None)
    femopt.add_parameter('r', .5, 0, 1)
    femopt.add_parameter(name=f'fai1', initial_value=np.pi*3/8, lower_bound=0, upper_bound=2*np.pi)

    femopt.add_objective(hypersphere_x, 'x0', args=(femopt.opt, 0), direction='maximize')
    femopt.add_objective(hypersphere_x, 'x1', args=(femopt.opt, 1), direction='maximize')

    femopt.add_parameter_constraint(constraint5, lower_bound=0)
    femopt.add_parameter_constraint(constraint6, lower_bound=0)

    femopt.set_random_seed(42)
    ret = femopt.optimize(n_trials=90, n_parallel=1, wait_setup=True)
    # input('enter to quit...')
    femopt.terminate_all()




if __name__ == '__main__':
    # min_sleep_sec = 3
    # random_max_sleep_sec = 1
    # simple()
    cum_length, took_time = parameter_constraint_test()

    # test_2_2_restart()

    # record = False
    # test_2_NoFEM_random_seed()
