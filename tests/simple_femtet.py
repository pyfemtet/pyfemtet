import os
os.chdir(os.path.split(__file__)[0])
from PyFemtet.opt import FemtetOptuna

# from win32com.client import Dispatch
# Femtet = Dispatch('FemtetMacro.Femtet')
# Femtet.OpenCurrentResult(True)
def max_displacement(Femtet):
    dy = Femtet.Gogh.Galileo.GetMaxDisplacement_py()[1]
    return dy*1000

def volume(Femtet):
    _, v = Femtet.Gogh.CalcVolume_py([0])
    return v * 1e9

def bottom_area_1(Femtet):
    _, v = Femtet.Gogh.CalcArea_py([0])
    return v * 1e6

def bottom_area_2(Femtet, FEMOpt):
    param = FEMOpt.get_parameter()
    a = param['w'] * param['d']
    return a


if __name__=='__main__':
    FEMOpt = FemtetOptuna(femprj_path='simple_femtet/simple.femprj')
    
    # add_parameter
    FEMOpt.add_parameter('w', 10, 5, 20)
    FEMOpt.add_parameter('d', 10, 5, 20)
    FEMOpt.add_parameter('h', 100, 50, 200)
    
    # add_objective
    FEMOpt.add_objective(max_displacement, '変位(mm)')
    FEMOpt.add_objective(volume, '体積(mm3)')

    # add non-strict constraint
    FEMOpt.add_constraint(bottom_area_1, '底面積1(mm2)', 100, strict=False)

    # add strict constraint
    FEMOpt.add_constraint(bottom_area_2, '底面積2(mm2)', 99, args=FEMOpt)

    # overwrite constraint
    FEMOpt.add_constraint(bottom_area_2, '底面積2(mm2)', 50, args=FEMOpt)
    
    FEMOpt.main(n_trials=12, n_parallel=3)
    
    # import optuna
    # FEMOpt.

    
    

