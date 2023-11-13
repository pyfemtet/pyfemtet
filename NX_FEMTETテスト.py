import numpy as np

from PyFemtet.opt._NX_Femtet import NX_Femtet
from PyFemtet.opt import FemtetScipy

from win32com.client import constants
def flow(Femtet):
    '''評価指標を定義する関数は、第一引数に Femtet のインスタンスを取るようにしてください。'''
    Gogh = Femtet.Gogh
    Gogh.Pascal.Vector = constants.PASCAL_VELOCITY_C
    _, ret = Gogh.SimpleIntegralVectorAtFace_py([2], [0], constants.PART_VEC_Y_PART_C)
    flow = ret.Real
    return flow

if __name__=='__main__':
    FEM = NX_Femtet() # この機能を使う際はエントリポイントをガードしてください。
    FEMOpt = FemtetScipy(FEM)
    FEMOpt.add_parameter('h', 20)
    FEMOpt.add_parameter('r', 10)
    FEMOpt.add_objective(flow)
    FEMOpt._initRecord()
    print(FEMOpt.f(np.array((20,10))))
    