import os
import numpy as np
from pyfemtet.opt import OptimizerOptuna, FemtetWithNXInterface


here, me = os.path.split(__file__)
os.chdir(here)


def disp(Femtet):
    _, _, ret = Femtet.Gogh.Galileo.GetMaxDisplacement_py()
    return ret


def volume(Femtet):
    Gogh = Femtet.Gogh
    _, ret = Gogh.CalcVolume_py([0])
    return ret


if __name__ == '__main__':
    fem = FemtetWithNXInterface('NX_ex01.prt')
    femopt = OptimizerOptuna(fem)

    femopt.add_parameter('A_x', 50, lower_bound=25, upper_bound=95)
    femopt.add_parameter('A_y', 45, lower_bound=5, upper_bound=45)
    femopt.add_parameter('B_x', 30, lower_bound=25, upper_bound=95)
    femopt.add_parameter('B_y', 12, lower_bound=5, upper_bound=45)
    femopt.add_parameter('C_x', 90, lower_bound=25, upper_bound=95)
    femopt.add_parameter('C_y', 45, lower_bound=5, upper_bound=45)
    femopt.add_parameter('Cut_x', 10, lower_bound=5, upper_bound=45)
    femopt.add_parameter('Cut_y', 20, lower_bound=5, upper_bound=45)
    femopt.add_objective(disp, direction=0)
    femopt.add_objective(volume, direction='minimize')

    femopt.main(n_trials=3, use_lhs_init=False)
