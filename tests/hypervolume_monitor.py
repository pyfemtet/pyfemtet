from PyFemtet.opt import FemtetOptuna
from PyFemtet.opt.core import NoFEM
from PyFemtet.opt.visualization import HypervolumeMonitor

def parabora1(FEMOpt):
    x = FEMOpt.get_parameter('value')
    return (x**2).sum()
def parabora2(FEMOpt):
    x = FEMOpt.get_parameter('value')
    return ((x-1)**2).sum()
def parabora3(FEMOpt):
    x = FEMOpt.get_parameter('value')
    from time import sleep
    sleep(1)
    return ((x-2)**2).sum()

FEMOpt = FemtetOptuna(NoFEM())

FEMOpt.add_parameter('x', -1, -5, 5)
FEMOpt.add_objective(parabora1, '放物線1', args=FEMOpt)
FEMOpt.add_objective(parabora2, '放物線2', args=FEMOpt)
FEMOpt.add_objective(parabora3, '放物線3', args=FEMOpt)

FEMOpt.set_process_monitor(HypervolumeMonitor)

FEMOpt.main(n_trials=100)

