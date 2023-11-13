import numpy as np

from PyFemtet.opt._NX_Femtet import NX_Femtet
from PyFemtet.opt import FemtetScipy

from win32com.client import constants
from win32com.client import Dispatch

import os

def disp(Femtet):
    '''評価指標を定義する関数は、第一引数に Femtet のインスタンスを取るようにしてください。'''
    # Femtet = Dispatch('FemtetMacro.Femtet')
    # Femtet.OpenCurrentResult(True)
    Gogh = Femtet.Gogh
    Gogh.Activate()
    _, _, ret = Gogh.Galileo.GetMaxDisplacement_py()
    return ret

if __name__=='__main__':
    here, me = os.path.split(__file__)
    os.chdir(here)
    FEM = NX_Femtet('NXTEST.prt') # この機能を使う際はエントリポイントをガードしてください。
    FEM.set_bas('NXTEST.bas')
    # FEM.set_excel('NXTEST.xlsm')
    FEMOpt = FemtetScipy(FEM)
    FEMOpt.add_parameter('Cut_x', 20, lower_bound=0, upper_bound=50)
    FEMOpt.add_parameter('Cut_y', 20, lower_bound=0, upper_bound=50)
    FEMOpt.add_objective(disp, direction=0)

    # FEMOpt._initRecord()
    # print(FEMOpt.f(np.array((5,5))))
    # FEMOpt.set_process_monitor() # TODO: サブスレッドからサブプロセスを呼び出せないので設計をなんとかする
    FEMOpt.main()
    