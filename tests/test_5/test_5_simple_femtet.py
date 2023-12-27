import os
import numpy as np
import pandas as pd
from pyfemtet.opt import OptimizerOptuna, FemtetInterface


here, me = os.path.split(__file__)
record = False


def max_disp(Femtet):
    return Femtet.Gogh.Galileo.GetMaxDisplacement_py()[1]


def volume(Femtet, femopt):
    d, h, _ = femopt.get_parameter('values')
    w = Femtet.GetVariableValue('w')
    return d * h * w


def bottom_surface(_, femopt):
    d, h, w = femopt.get_parameter('values')
    return d * w


def test_simple_femtet():
    """
    テストしたい状況
        Femtet で一通りの機能が動くか
    """
    femprj = os.path.join(here, f'{me.replace(".py", ".femprj")}')
    csvdata = os.path.join(here, f'{me.replace(".py", ".csvdata")}')
    csv = os.path.join(here, f'{me.replace(".py", "_specified_history.csv")}')

    fem = FemtetInterface(femprj)
    femopt = OptimizerOptuna(fem)
    femopt.add_parameter('d', 5, 1, 10)
    femopt.add_parameter('h', 5, 1, 10)
    femopt.add_parameter('w', 5, 1, 10)
    femopt.add_objective(max_disp)  # 名前なし目的変数（obj_0 になる）
    femopt.add_objective(max_disp)  # 名前なし目的変数（obj_1 になる）
    femopt.add_objective(volume, 'volume(mm3)', args=femopt)
    femopt.add_objective(volume, 'volume(mm3)', args=femopt)  # 上書き
    femopt.add_constraint(bottom_surface, 'surf<=20', upper_bound=20, args=femopt)

    femopt.set_random_seed(42)
    femopt.main(n_trials=30, n_parallel=3)

    femopt.terminate_monitor()
    try:
        femopt.fem.quit()
    except:
        pass

def simple():
    """シンプルな動作確認用"""

    path = os.path.join(os.path.dirname(__file__), 'test_5_simple_femtet.femprj')

    fem = FemtetInterface(path)
    femopt = OptimizerOptuna(fem)
    femopt.set_random_seed(42)
    femopt.add_parameter('d', 5, 1, 10)
    femopt.add_parameter('h', 5, 1, 10)
    femopt.add_parameter('w', 5, 1, 10)
    femopt.add_objective(max_disp, '最大変位(m)')
    femopt.add_objective(volume, '体積(mm3)', args=femopt)
    femopt.add_constraint(bottom_surface, '底面積<=20', upper_bound=30, args=femopt)
    femopt.main(n_trials=30, n_parallel=3)
    femopt.terminate_monitor()


if __name__ == '__main__':
    test_simple_femtet()


