import numpy as np

from PyFemtet.opt._SW_Femtet import SW_Femtet
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

# def Ax_is_greater_than_Bx(Femtet, FEMOpt):
#     '''評価指標を定義する関数は、第一引数に Femtet のインスタンスを取るようにしてください。'''
#     value_dict = FEMOpt.get_parameter('dict')
#     return value_dict['A_x'] - value_dict['B_x']

# def Ay_is_greater_than_By(Femtet, FEMOpt):
#     '''評価指標を定義する関数は、第一引数に Femtet のインスタンスを取るようにしてください。'''
#     value_dict = FEMOpt.get_parameter('dict')
#     return value_dict['A_y'] - value_dict['B_x']


if __name__=='__main__':
    here, me = os.path.split(__file__)
    os.chdir(here)
    FEM = SW_Femtet(r'SWFemtetWithProcessMonitor\SWTEST.SLDPRT') # この機能を使う際はエントリポイントをガードしてください。
    FEMOpt = FemtetOptuna(FEM)
    FEMOpt.add_parameter('A_x', 10, lower_bound=5, upper_bound=50)
    FEMOpt.add_parameter('A_y',  7, lower_bound=5, upper_bound=25)
    FEMOpt.add_parameter('B_x', 15, lower_bound=5, upper_bound=50)
    FEMOpt.add_parameter('B_y', 20, lower_bound=5, upper_bound=25)
    FEMOpt.add_parameter('C_x', 40, lower_bound=5, upper_bound=50)
    FEMOpt.add_parameter('C_y', 15, lower_bound=5, upper_bound=25)
    FEMOpt.add_parameter('Cut_x', 5, lower_bound=5, upper_bound=20)
    FEMOpt.add_parameter('Cut_y', 5, lower_bound=5, upper_bound=20)
    FEMOpt.add_objective(disp, direction=0)
    FEMOpt.add_objective(volume, direction='minimize')
    # FEMOpt.add_constraint(
    #     Ax_is_greater_than_Bx,
    #     lower_bound=0,
    #     args=FEMOpt
    #     )
    # FEMOpt.add_constraint(
    #     Ay_is_greater_than_By,
    #     lower_bound=0,
    #     args=FEMOpt
    #     )

    # FEMOpt.set_process_monitor()
    FEMOpt.main()
    
    # FEMOpt.FEM.swApp.ExitApp()
    