import os
from time import sleep

import numpy as np

from pyfemtet.opt import OptimizerOptuna
from pyfemtet.opt import NoFEM


should_sleep = False


def objective_x(femopt):
    if should_sleep:
        sleep(3)
    r, theta, fai = femopt.get_parameter('values')
    return r * np.cos(theta) * np.cos(fai)


def objective_y(femopt):
    r, theta, fai = femopt.get_parameter('values')
    return r * np.cos(theta) * np.sin(fai)


def objective_z(femopt):
    r, theta, fai = femopt.get_parameter('values')
    return r * np.sin(theta)


def constraint_y(femopt):
    y = objective_y(femopt)
    return y


def constraint_z(femopt):
    z = objective_z(femopt)
    return z


if __name__ == '__main__':

    here, me = os.path.split(__file__)
    os.chdir(here)

    fem = NoFEM()
    femopt = OptimizerOptuna(fem, history_path='5_restart_NoFEM/history.csv')
    femopt.add_parameter('r', .5, 0, 1)
    femopt.add_parameter('theta', np.pi / 3, -np.pi / 2, np.pi / 2)  # 空間上で xy 平面となす角
    femopt.add_parameter('fai', (7 / 6) * np.pi, 0, 2 * np.pi)  # xy 平面上で x 軸となす角
    femopt.add_objective(objective_x, 'x(mm)', args=femopt)
    femopt.add_objective(objective_y, 'y(mm)', args=femopt)
    femopt.add_objective(objective_z, 'z(mm)', args=femopt)
    femopt.add_objective(objective_z, 'z2(mm)', args=femopt)  # 3 回目
    femopt.add_constraint(constraint_y, 'y<=0', upper_bound=0, args=femopt)

    # femopt.main(n_trials=20, n_parallel=3)  # 1 回目
    # femopt.main(n_trials=40, n_parallel=3)  # 2 回目
    femopt.main(n_trials=60, n_parallel=3)  # 3 回目

    print(femopt.history.data)

