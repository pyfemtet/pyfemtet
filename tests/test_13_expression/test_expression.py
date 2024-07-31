import os

from numpy import pi

from pyfemtet._test_util import SuperSphere
from pyfemtet.opt import FEMOpt, NoFEM, FemtetInterface, AbstractOptimizer

here = os.path.dirname(__file__)
os.chdir(here)

s = SuperSphere(2)


def obj(opt: AbstractOptimizer, idx: int):
    d = opt.get_parameter('dict')
    return s.x(d['radius'], d['fai'] / 180 * pi)[idx]


def cns1(Femtet):
    return Femtet.GetVariableValue('radius')



def no_fem():
    fem = NoFEM()

    # femopt の準備
    femopt = FEMOpt(fem=fem)

    # parameter の準備
    femopt.add_parameter('radius', 0.5, 0, 1, step=0.1, property={'unit': 'mm'})
    femopt.add_parameter('fai_virtual', pi, pi/2, 3/2*pi, property={'unit': 'radian'}, direct_to_fem=False)
    femopt.add_expression('fai', fai, direct_to_fem=True)

    # objective の準備
    femopt.add_objective(obj, 'x', args=(femopt.opt, 0))
    femopt.add_objective(obj, 'y', args=(femopt.opt, 1))

    # constraints の準備
    femopt.add_constraint(obj, 'x(cns)', args=(femopt.opt, 0), upper_bound=0, strict=False)
    femopt.add_constraint(obj, 'y(cns)', args=(femopt.opt, 1), upper_bound=0.5, strict=True)

    # 実行
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=30)

    # 終了
    femopt.terminate_all()


def fai(prm):
    return float(prm['fai_virtual']) / pi * 180


def femtet():
    fem = FemtetInterface(
        femprj_path=os.path.join(here, 'test_expression.femprj'),
        parametric_output_indexes_use_as_objective={
            0: 'minimize',
            1: 'minimize',
        }
    )

    # femopt の準備
    femopt = FEMOpt(fem=fem)

    # parameter の準備
    femopt.add_parameter('radius', 0.5, 0, 1, step=0.1, property={'unit': 'mm'})
    femopt.add_parameter('fai_virtual', pi, pi/2, 3/2*pi, property={'unit': 'radian'}, direct_to_fem=False)
    femopt.add_expression('fai', fai, direct_to_fem=True)

    # constraints の準備
    femopt.add_constraint(cns1, upper_bound=0.8, strict=True)
    femopt.add_constraint(cns1, lower_bound=0.3, strict=False)

    # 実行
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=30)

    # 終了
    femopt.terminate_all()


if __name__ == '__main__':
    no_fem()
    # femtet()
