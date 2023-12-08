import numpy as np

from PyFemtet.opt._NX_Femtet import NX_Femtet
from PyFemtet.opt import FemtetOptuna

from win32com.client import constants
from win32com.client import Dispatch

import os

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


if __name__=='__main__':
    here, me = os.path.split(__file__)
    os.chdir(here)
    FEM = NX_Femtet('NX_ex01.prt') # この機能を使う際はエントリポイントをガードしてください。
    FEMOpt = FemtetOptuna(FEM)
    FEMOpt.add_parameter('A_x', 50, lower_bound=25, upper_bound=95)
    FEMOpt.add_parameter('A_y', 45, lower_bound=5, upper_bound=45)
    FEMOpt.add_parameter('B_x', 30, lower_bound=25, upper_bound=95)
    FEMOpt.add_parameter('B_y', 12, lower_bound=5, upper_bound=45)
    FEMOpt.add_parameter('C_x', 90, lower_bound=25, upper_bound=95)
    FEMOpt.add_parameter('C_y', 45, lower_bound=5, upper_bound=45)
    FEMOpt.add_parameter('Cut_x', 10, lower_bound=5, upper_bound=45)
    FEMOpt.add_parameter('Cut_y', 20, lower_bound=5, upper_bound=45)
    FEMOpt.add_objective(disp, direction=0)
    FEMOpt.add_objective(volume, direction='minimize')

    FEMOpt.set_process_monitor()
    FEMOpt.main()
    