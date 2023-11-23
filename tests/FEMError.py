# -*- coding: utf-8 -*-
import numpy as np
from PyFemtet.opt import FemtetScipy, FemtetOptuna
from PyFemtet.opt.core import NoFEM, ModelError, MeshError, SolveError

np.random.seed(3)

def parabora(FEMObj):
    x = FEMObj.get_parameter('value')
    if np.random.rand()>.5:
        return (x**2).sum()
    else:
        raise ModelError

def parabora2(FEMObj):
    x = FEMObj.get_parameter('value')
    if np.random.rand()>.5:
        return ((x-2)**2).sum()
    else:
        raise ModelError

FEM = NoFEM()
# FEMOpt = FemtetScipy(FEM)
FEMOpt = FemtetOptuna(FEM)

FEMOpt.add_parameter('x', 1, 0, 2)
FEMOpt.add_objective(parabora, args=(FEMOpt,))
FEMOpt.add_objective(parabora2, args=(FEMOpt,))
FEMOpt.set_process_monitor()
FEMOpt.main(n_trials=100)
print(FEMOpt.history)
 