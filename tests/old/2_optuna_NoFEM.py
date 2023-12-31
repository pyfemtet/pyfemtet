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
    fem = NoFEM()
    femopt = OptimizerOptuna(fem)
    femopt.add_parameter('r', .5, 0, 1)
    femopt.add_parameter('theta', np.pi/3, -np.pi/2, np.pi/2)  # 空間上で xy 平面となす角
    femopt.add_parameter('fai', (7/6)*np.pi, 0, 2*np.pi)  # xy 平面上で x 軸となす角
    femopt.add_objective(objective_x, 'x(mm)', args=femopt)
    femopt.add_objective(objective_y, 'y(mm)', args=femopt)
    femopt.add_objective(objective_z, 'z(mm)', args=femopt)
    femopt.add_constraint(constraint_y, 'y<=0', upper_bound=0, args=femopt)


    # 拘束なし
    # should_sleep = False
    # femopt.main(n_trials=10)  # 動くか
    # should_sleep = True
    # femopt.main(n_trials=30)  # UI から中断できるか
    # femopt.main(n_trials=40, n_parallel=3)  # 並列が動くか（n_trials/n_parallel 秒というわけにはいかない）
    # femopt.main(n_trials=90, n_parallel=3, timeout=20)  # timeout が効くか

    # 拘束あり
    # should_sleep = False
    # femopt.main(n_trials=5)  # 拘束が動くか（pruned されない結果が 5 個）
    should_sleep = True
    femopt.main(timeout=10, n_parallel=3)  # 拘束が動くか（pruned されない結果が 30 個）
    # femopt.main(n_trials=30, n_parallel=3)  # 拘束が動くか（pruned されない結果が 30 個）

    print(femopt.history.data)


    # # optuna db との比較
    # import optuna
    # study = optuna.load_study(study_name=femopt.study_name, storage=femopt.storage)
    # df = study.trials_dataframe()
    # idx = df['state'] == 'COMPLETE'
    # print(len(df[idx]))



