import os
from optuna_integration import BoTorchSampler
from pyfemtet.opt import FEMOpt, OptunaOptimizer, FemtetInterface


def mises_stress(Femtet):
    """ミーゼス応力を返します。"""
    return Femtet.Gogh.Galileo.GetMaxStress_py()[2]


def sumation(Femtet, opt):
    """三辺の和を返します。"""
    d = Femtet.GetVariableValue('d')
    w = Femtet.GetVariableValue('w')
    h = Femtet.GetVariableValue('h')
    # d, h, w = opt.get_parameter('values')
    return d + w + h


if __name__ == '__main__':

    here = os.path.dirname(__file__)
    os.chdir(here)

    # opt = OptunaOptimizer()
    opt = OptunaOptimizer(
        sampler_class=BoTorchSampler,
        sampler_kwargs=dict(
            n_startup_trials=3
        ),
    )

    fem = FemtetInterface(femprj_path='constraint.femprj')
    femopt = FEMOpt(opt=opt, fem=fem)

    femopt.add_parameter('d', 20, 1, 50)
    femopt.add_parameter('h', 20, 10, 80)
    femopt.add_parameter('w', 20, 20, 110)

    # femopt.add_constraint(sumation, '3辺の和', lower_bound=30, upper_bound=120, args=(opt,), using_fem=False)
    femopt.add_constraint(sumation, '3辺の和', lower_bound=30, upper_bound=120, args=(opt,), using_fem=True)

    femopt.add_objective(mises_stress, '最大ミーゼス応力')

    femopt.set_random_seed(42)
    femopt.optimize(timeout=600)
