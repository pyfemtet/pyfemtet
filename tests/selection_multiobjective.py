import numpy as np

from PyFemtet.opt import FemtetOptuna
from PyFemtet.opt.core import NoFEM

def x(FEMOpt):
    r, theta, fai = FEMOpt.get_parameter('value')
    return r * np.cos(theta) * np.cos(theta)

def y(FEMOpt):
    r, theta, fai = FEMOpt.get_parameter('value')
    return r * np.cos(theta) * np.sin(theta)

def z(FEMOpt):
    r, theta, fai = FEMOpt.get_parameter('value')
    return r * np.sin(theta)

FEM = NoFEM()
FEMOpt = FemtetOptuna(FEM)

FEMOpt.add_parameter('r', 0.5, 0, 1)
FEMOpt.add_parameter('theta', 0, -np.pi/2, np.pi/2)
FEMOpt.add_parameter('fai', 0, 0, 2*np.pi)

FEMOpt.add_objective(x, args=FEMOpt)
FEMOpt.add_objective(y, args=FEMOpt)
FEMOpt.add_objective(z, args=FEMOpt)

FEMOpt.set_process_monitor()

FEMOpt.main(n_trials=10)

