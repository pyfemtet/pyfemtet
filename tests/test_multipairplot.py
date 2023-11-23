import numpy as np

from PyFemtet.opt.visualization import MultiobjectivePairPlot
from PyFemtet.opt.visualization import UpdatableSuperFigure
from PyFemtet.opt.visualization import HypervolumeMonitor

from PyFemtet.opt import FemtetOptuna
from PyFemtet.opt.core import NoFEM


if __name__=='__main__':
    np.random.seed(4)
    
    FEM = NoFEM()
    FEMOpt = FemtetOptuna(FEM)
    
    n = 2
    
    for i in range(n):
        FEMOpt.add_objective(lambda:None, f"obj{i}")
    
    FEMOpt._initRecord()
    
    for i in range(n):
        FEMOpt.history[f'obj{i}'] = np.random.rand(10)*np.random.rand()*10
    
    FEMOpt._calcNonDomi()
    FEMOpt._calc_hypervolume()
    FEMOpt.history['fit'] = True
    idx = (np.random.rand(1)*10).astype(int)
    FEMOpt.history['fit'][idx] = False
    FEMOpt.history['n_trial'] = range(10)
    
    # MultiobjectivePairPlot(FEMOpt)
    # HypervolumeMonitor(FEMOpt)
    pm = UpdatableSuperFigure(FEMOpt, HypervolumeMonitor, MultiobjectivePairPlot)
    