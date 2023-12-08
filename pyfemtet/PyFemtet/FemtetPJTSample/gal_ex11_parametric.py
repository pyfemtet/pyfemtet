from PyFemtet.opt import FemtetOptuna
from PyFemtet.opt.visualization import HypervolumeMonitor

from win32com.client import Dispatch, constants


def fundamental_resonance(Femtet):
    '''mode[0] の共振周波数の取得'''
    # 結果取得用関数の作成に使用した前処理
    # Femtet = Dispatch('FemtetMacro.Femtet')
    # Femtet.OpenCurrentResult(True)
    Gogh = Femtet.Gogh
    Gogh.Galileo.Mode = 0
    freq:'complex' = Gogh.Galileo.GetFreq()
    # print(freq.Real)
    return freq.Real

def mass(Femtet):
    rho = Femtet.GetVariableValue('rho')
    _, volume = Femtet.Gogh.CalcVolume_py([0])
    return rho * volume # kg/m3 * m3

def thickness(Femtet):
    external_r = Femtet.GetVariableValue('external_r')
    internal_r = Femtet.GetVariableValue('internal_r')
    # Femtet.Gogh
    return external_r - internal_r
    

if __name__=='__main__':

    FEMOpt = FemtetOptuna()

    FEMOpt.add_parameter('internal_r', initial_value=1.1, lower_bound=0.2, upper_bound=3)
    FEMOpt.add_parameter('external_r', initial_value=1.5, lower_bound=0.1, upper_bound=2.9)
    FEMOpt.add_parameter('rho', initial_value=8000, lower_bound=6000, upper_bound=9000)

    FEMOpt.add_objective(fundamental_resonance, '共振周波数(Hz)', 50)
    FEMOpt.add_objective(mass, '質量(kg)', 1000)
    
    FEMOpt.add_constraint(thickness, '厚さ(m)', lower_bound=0.3)
    
    FEMOpt.set_process_monitor()
    
    FEMOpt.main()
    
    # import matplotlib.pyplot as plt
    # plt.show() # メインルーチンをここで止め、プロセスモニタが消えないようにする
    





