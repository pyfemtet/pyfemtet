import sys
from subprocess import run

import os
from time import sleep
import numpy as np
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
    femopt.add_objective(objective_x, 'x(mm)', args=femopt)
    femopt.add_objective(objective_y, 'y(mm)', args=femopt)
    femopt.add_objective(objective_z, 'z(mm)', args=femopt, direction=-1)
    femopt.add_constraint(constraint_y, 'y<=0', upper_bound=0, args=femopt)  # 上書き
    femopt.add_constraint(constraint_z, 'z<=0', upper_bound=0, args=femopt, strict=False)
    femopt.main(n_trials=30)

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

        assert np.sum(np.abs(def_df.values - ref_df.values)) < 0.0001


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
    femopt.main(n_trials=5)
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
    df = pd.read_csv('test_2_2_restart.csv', encoding='shift-jis')
    assert len(df) == 10


def simple():
    """シンプルな動作確認"""

    fem = NoFEM()
    opt = OptunaOptimizer()
    femopt = FEMOpt(fem, opt, scheduler_address=None)

    femopt.add_parameter('r', .5, 0, 1)
    femopt.add_parameter('theta', -np.pi/3, -np.pi/2, np.pi/2)  # 空間上で xy 平面となす角
    femopt.add_parameter('fai', (7/6)*np.pi, 0, 2*np.pi)  # xy 平面上で x 軸となす角
    femopt.add_objective(objective_x, 'x', args=femopt.opt)
    femopt.add_objective(objective_y, 'y', args=femopt.opt)
    # femopt.add_objective(objective_z, 'z', args=femopt.opt)
    # femopt.add_constraint(objective_z, 'z<=0', upper_bound=0, args=femopt.opt)
    femopt.add_constraint(objective_z, 'z<=0', upper_bound=0, args=femopt.opt, strict=False)
    femopt.set_random_seed(42)
    femopt.main(n_trials=None, n_parallel=1, wait_setup=True)
    # input('enter to quit...')
    # femopt.terminate_all()


if __name__ == '__main__':
    simple()
    # test_2_2_restart()
    # record = False
    # test_2_NoFEM_random_seed()
