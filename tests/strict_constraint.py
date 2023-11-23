import numpy as np

from PyFemtet.opt import FemtetOptuna
from PyFemtet.opt.core import NoFEM
from PyFemtet.opt.visualization import MultiobjectivePairPlot

def x(FEMOpt):
    r, theta, fai = FEMOpt.get_parameter('value')
    return r * np.cos(theta) * np.cos(fai)

def y(FEMOpt):
    r, theta, fai = FEMOpt.get_parameter('value')
    return r * np.cos(theta) * np.sin(fai)

def z(FEMOpt):
    r, theta, fai = FEMOpt.get_parameter('value')
    return r * np.sin(theta)

def strict_constraint(FEMOpt):
    r, theta, fai = FEMOpt.get_parameter('value')
    return fai

def weak_constraint(FEMOpt):
    return x(FEMOpt)

FEM = NoFEM()
FEMOpt = FemtetOptuna(FEM)

FEMOpt.add_parameter('r', 0.7, 0, 1)
FEMOpt.add_parameter('theta', 0, -np.pi/2, np.pi/2)
FEMOpt.add_parameter('fai', 0, 0, 2*np.pi)

FEMOpt.add_objective(x, 'x', args=FEMOpt)
FEMOpt.add_objective(y, 'y', args=FEMOpt)
FEMOpt.add_objective(z, 'z', args=FEMOpt)

FEMOpt.add_constraint(strict_constraint, 'fai<pi', upper_bound=np.pi, args=FEMOpt)
FEMOpt.add_constraint(weak_constraint, 'x>0', lower_bound=0, strict=False, args=FEMOpt)

FEMOpt.set_process_monitor(MultiobjectivePairPlot)

FEMOpt.main(n_trials=100)

