import os
from time import sleep
import numpy as np
import pandas as pd
from pyfemtet.opt import FemtetWithNXInterface, OptimizerOptuna
from femtetutils import util


here, me = os.path.split(__file__)
record = False


def disp(Femtet):
    '''評価指標を定義する関数は、第一引数に Femtet のインスタンスを取るようにしてください。'''
    # Femtet = Dispatch('FemtetMacro.Femtet')
    # Femtet.OpenCurrentResult(True)
    Gogh = Femtet.Gogh
    _, _, ret = Gogh.Galileo.GetMaxDisplacement_py()
    return ret


def volume(Femtet):
    '''評価指標を定義する関数は、第一引数に Femtet のインスタンスを取るようにしてください。'''
    Gogh = Femtet.Gogh
    _, ret = Gogh.CalcVolume_py([0])
    return ret

def C_minus_B(_, femopt):
    A, B, C = femopt.get_parameter('values')
    return C - B


def test_3_1():

    # NX-Femtet 連携クラスのインスタンス化
    fem = FemtetWithNXInterface(
        prt_path=os.path.join(here, 'test3/test3.prt'),
        femprj_path=os.path.join(here, 'test3/test3.femprj'),
        connect_method='new'
    )

    # 最適設計クラスのインスタンス化
    femopt = OptimizerOptuna(fem)

    # 乱数シードの設定
    femopt.set_random_seed(42)

    # 変数設定
    femopt.add_parameter('A', 10, lower_bound=1, upper_bound=59)
    femopt.add_parameter('B', 10, lower_bound=1, upper_bound=40)
    femopt.add_parameter('C', 20, lower_bound=5, upper_bound=59)
    femopt.add_constraint(C_minus_B, 'C>B', lower_bound=1, args=femopt)

    # 目的関数設定
    femopt.add_objective(disp, name='変位', direction=0)
    femopt.add_objective(volume, name='体積', direction='minimize')

    # 最適化の実行
    femopt.main(n_trials=10, use_lhs_init=False)
    femopt.terminate_monitor()

    # Femtet 終了
    tmppath = os.path.join(here, 'tmp.femprj')
    femopt.fem.Femtet.SaveProjectIgnoreHistory(tmppath, True)
    sleep(10)
    util.close_femtet(femopt.fem.Femtet.hWnd)
    os.remove(tmppath)


if __name__ == '__main__':
    test_3_1()


