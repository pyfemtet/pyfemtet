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


def simple():
    """シンプルな動作確認"""

    fem = NoFEM()
    opt = OptunaOptimizer()
    femopt = FEMOpt(fem, opt, scheduler_address=None)

    femopt.add_parameter('r', .5, 0, 1, step=0.5)
    femopt.add_parameter('theta', -np.pi/3, -np.pi/2, np.pi/2)  # 空間上で xy 平面となす角
    femopt.add_parameter('fai', (7/6)*np.pi, 0, 2*np.pi)  # xy 平面上で x 軸となす角
    femopt.add_objective(objective_x, 'x', args=femopt.opt)
    femopt.add_objective(objective_y, 'y座標', args=femopt.opt)
    # femopt.add_objective(objective_z, 'z', args=femopt.opt)
    # femopt.add_constraint(objective_z, 'z<=0', upper_bound=0, args=femopt.opt)
    # femopt.add_constraint(objective_z, 'z<=0', upper_bound=0, args=femopt.opt, strict=False)
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=30, n_parallel=1, wait_setup=True)
    input('enter to quit...')
    femopt.terminate_all()


if __name__ == '__main__':
    # min_sleep_sec = 3
    # random_max_sleep_sec = 1
    simple()

    # test_2_2_restart()

    # record = False
    # test_2_NoFEM_random_seed()
