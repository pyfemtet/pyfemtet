"""Test to set RANDOM_SEED.

Return True if:
    set specific cases
"""
import os
os.chdir(os.path.split(__file__)[0])
import pickle
from PyFemtet.opt import FemtetOptuna
from PyFemtet.opt.core import NoFEM


def parabola(FEMOpt)->float:
    """N-dimensional parabola function.

    Parameters
    ----------
    FEMOpt : FemtetOptimizationCore

    Returns
    -------
    float

    """
    x = FEMOpt.get_parameter('values')
    return (x**2).sum()


if __name__=='__main__':

    FEMOpt = FemtetOptuna(FEM=NoFEM())
    FEMOpt.set_random_seed(42)
    FEMOpt.add_parameter('x', 0, 0, 1)
    FEMOpt.add_parameter('y', 0, 0, 1)
    FEMOpt.add_parameter('z', 0, 0, 1)
    FEMOpt.add_objective(parabola, 'N次元放物線', args=(FEMOpt,))
    FEMOpt.main(n_trials=30)

    h = FEMOpt.history

    # with open('random_set_seed/random_seed_42.pkl', 'wb') as f:
    #     pickle.dump(h, f)

    with open('random_set_seed/random_seed_42.pkl', 'rb') as f:
        ref_h = pickle.load(f)

    parameter_names = FEMOpt.get_history_columns('parameter')
    ref_parameter_set = ref_h[parameter_names]
    dif_parameter_set = h[parameter_names]
    result = (ref_parameter_set == dif_parameter_set).all().all()
    
    print(result)

    

