import os
from pyfemtet.opt import OptimizerOptuna, FemtetInterface

here, me = os.path.split(__file__)


# from win32com.client import Dispatch
# FemtetInterface = Dispatch('FemtetMacro.FemtetInterface')
# FemtetInterface.OpenCurrentResult(True)
def max_displacement(Femtet):
    dy = Femtet.Gogh.Galileo.GetMaxDisplacement_py()[1]
    return dy * 1000


def volume(Femtet):
    _, v = Femtet.Gogh.CalcVolume_py([0])
    return v * 1e9


def bottom_area_1(Femtet):
    _, v = Femtet.Gogh.CalcArea_py([0])
    return v * 1e6


def bottom_area_2(Femtet, femopt):
    param = femopt.get_parameter()
    a = param['w'] * param['d']
    return a


if __name__ == '__main__':

    path = os.path.join(here, f'{me.replace(".py", "")}/simple.femprj')

    fem = FemtetInterface(femprj_path=path, model_name=None, connect_method='auto')
    # femopt = OptimizerOptuna(fem)
    femopt = OptimizerOptuna()  # 開いている Femtet と接続する

    # add_parameter
    femopt.add_parameter('w', 10, 5, 20)
    femopt.add_parameter('d', 10, 5, 20)
    femopt.add_parameter('h', 100, 50, 200)

    # add_objective
    # femopt.add_objective(max_displacement, '変位(mm)')
    # femopt.add_objective(volume, '体積(mm3)')
    femopt.add_objective(max_displacement, '変位(mm)')
    femopt.add_objective(volume, '体積(mm3)')

    # add non-strict constraint
    # femopt.add_constraint(bottom_area_1, '底面積1(mm2)', 100, strict=False)

    # add strict constraint
    # femopt.add_constraint(bottom_area_2, '底面積2(mm2)', 99, args=femopt)

    # overwrite constraint
    femopt.add_constraint(bottom_area_2, '底面積2(mm2)', 50, args=femopt)

    # add Gogh-accessing function as strict
    # femopt.add_constraint(bottom_area_1, 'エラーになる拘束', 100)  # エラーになるか


    # femopt.main(n_trials=3)  # 動くか
    # femopt.main(n_trials=20)  # monitor 及び 中断が効くか
    femopt.main(n_trials=30, n_parallel=3)  # 並列が動くか


    print(femopt.history.data)





