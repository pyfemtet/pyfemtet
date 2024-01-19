import sys
from subprocess import run

import os
from time import sleep
import numpy as np
import pandas as pd
from pyfemtet.opt import OptimizationManager, OptunaOptimizer, NoFEM


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


# def test_2_1():
#     """
#     テストしたい状況
#         一通りの機能（ランダム含む、並列含まず）
#     """
#     fem = NoFEM()
#     opt = OptunaOptimizer()
#     femopt = OptimizationManager(fem, opt)
#     femopt.set_random_seed(42)
#     femopt.add_parameter('r（半径）', .5, 0, 1)
#     femopt.add_parameter('theta（角度1）', np.pi/3, -np.pi/2, np.pi/2)  # 空間上で xy 平面となす角
#     femopt.add_parameter('fai（角度2）', (7/6)*np.pi, 0, 2*np.pi)  # xy 平面上で x 軸となす角
#     femopt.add_objective(objective_x, args=femopt)  # 自動の名前付け
#     femopt.add_objective(objective_y, 'y(mm)', args=femopt)
#     femopt.add_objective(objective_z, 'z(mm)', args=femopt, direction=-1)
#     femopt.add_constraint(constraint_y, 'y<=0', upper_bound=0, args=femopt)  # 上書き
#     femopt.add_constraint(constraint_z, 'z<=0', upper_bound=0, args=femopt, strict=False)
#     femopt.main(n_trials=20)
#
#     if record:
#         femopt.history.data.to_csv(os.path.join(here, f'test2/test2_TPE.csvdata'), index=None)
#
#     else:
#         # データの取得
#         ref_df = pd.read_csv(os.path.join(here, f'test2/test2_TPE.csvdata')).replace(np.nan, None)
#         def_df = femopt.history.data.copy()
#
#         # 並べ替え（並列しているから順番は違いうる）
#         ref_df = ref_df.iloc[:, 1:].sort_values('r').sort_values('theta').sort_values('fai').select_dtypes(include='number')
#         def_df = def_df.iloc[:, 1:].sort_values('r').sort_values('theta').sort_values('fai').select_dtypes(include='number')
#
#         assert np.sum(np.abs(def_df.values - ref_df.values)) < 0.001
#
#
# def test_2_2():
#     """
#     テストしたい状況
#         FEM なしで一通りの機能が動くか
#     """
#
#     fem = NoFEM()
#     femopt = OptimizerOptuna(fem)
#     femopt.set_random_seed(42)
#     femopt.add_parameter('r', .5, 0, 1)
#     femopt.add_parameter('theta', np.pi/3, -np.pi/2, np.pi/2)  # 空間上で xy 平面となす角
#     femopt.add_parameter('fai', (7/6)*np.pi, 0, 2*np.pi)  # xy 平面上で x 軸となす角
#     femopt.add_objective(objective_x, args=femopt)  # 名前なし目的変数（obj_0 になる）
#     femopt.add_objective(objective_x, args=femopt)  # 名前なし目的変数（obj_1 になる）
#     femopt.add_objective(objective_y, 'y(mm)', args=femopt)
#     femopt.add_objective(objective_y, 'y(mm)', args=femopt, direction='maximize')  # 上書きかつ direction 指定
#     femopt.add_objective(objective_z, 'z(mm)', args=femopt, direction=-1)  # direction 数値指定
#     femopt.add_constraint(constraint_y, 'y<=0', upper_bound=-1, args=femopt)
#     femopt.add_constraint(constraint_y, 'y<=0', upper_bound=0, args=femopt)  # 上書き
#     femopt.add_constraint(constraint_z, 'z<=0', upper_bound=0, args=femopt, strict=False)
#     femopt.main(n_trials=20, n_parallel=3)
#     femopt.terminate_monitor()
#
#


def simple():
    """シンプルな動作確認"""

    fem = NoFEM()
    opt = OptunaOptimizer()
    femopt = OptimizationManager(fem, opt, scheduler_address=None)

    femopt.add_parameter('r', .5, 0, 1)
    femopt.add_parameter('theta', -np.pi/3, -np.pi/2, np.pi/2)  # 空間上で xy 平面となす角
    femopt.add_parameter('fai', (7/6)*np.pi, 0, 2*np.pi)  # xy 平面上で x 軸となす角
    femopt.add_objective(objective_x, 'x', args=femopt.opt)
    femopt.add_objective(objective_y, 'y', args=femopt.opt)
    femopt.add_objective(objective_z, 'z', args=femopt.opt)
    femopt.add_constraint(objective_z, 'z<=0', upper_bound=0, args=femopt.opt)
    femopt.set_random_seed(42)
    femopt.main(n_trials=3, n_parallel=1)
    femopt.terminate_all()


if __name__ == '__main__':
    simple()
