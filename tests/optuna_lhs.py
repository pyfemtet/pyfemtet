from time import sleep

from PyFemtet.opt import FemtetOptuna
from PyFemtet.opt.core import NoFEM

import numpy as np

def objective_x(FEMOpt):
    r, theta = FEMOpt.get_parameter('value')
    return r * np.cos(theta)

def objective_y(FEMOpt):
    r, theta = FEMOpt.get_parameter('value')
    sleep(1)
    return r * np.sin(theta)

def constraint_y(FEMOpt):
    y = objective_y(FEMOpt)
    return y
    
    
def main():
    FEM = NoFEM()
    FEMOpt = FemtetOptuna(FEM)
    FEMOpt.add_parameter('r', 0, 0, 1)
    FEMOpt.add_parameter('theta', 0, 0, 2*np.pi)
    FEMOpt.add_objective(objective_x, 'x', args=FEMOpt)
    FEMOpt.add_objective(objective_y, 'y', args=FEMOpt)
    FEMOpt.add_constraint(constraint_y, 'y<=0', upper_bound=0, args=FEMOpt)
    FEMOpt.set_process_monitor()
    FEMOpt.main()

if __name__=='__main__':
    main()

