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


def obj_2(opt: AbstractOptimizer, idx: int):
    d = opt.get_parameter('dict')
    return s.x(d['radius'], d['ファイ'] / 180 * pi)[idx]


def cns1(Femtet):
    return Femtet.GetVariableValue('radius')


def no_fem():
    fem = NoFEM()

    # femopt の準備
    femopt = FEMOpt(fem=fem)

    # parameter の準備
    femopt.add_parameter('radius', 0.5, 0, 1, step=0.1)
    femopt.add_parameter('fai_virtual', pi, pi/2, 3/2*pi, properties={'unit': 'radian'}, pass_to_fem=False)
    femopt.add_expression('fai', fai2, pass_to_fem=True, properties={'unit': 'degree'})

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


def fai(fai_virtual):
    return float(fai_virtual) / pi * 180


def fai2(fai_virtual):
    return float(fai_virtual) / pi * 180


def fai3(fai_virtual_no_exists):
    return float(fai_virtual_no_exists) / pi * 180


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
    femopt.add_parameter('radius', 0.5, 0, 1, step=0.1, properties={'unit': 'mm'})
    femopt.add_parameter('fai_virtual', pi, pi/2, 3/2*pi, properties={'unit': 'radian'}, pass_to_fem=False)
    femopt.add_expression('fai', fai, pass_to_fem=True)

    # constraints の準備
    femopt.add_constraint(cns1, upper_bound=0.8, strict=True)
    femopt.add_constraint(cns1, lower_bound=0.3, strict=False)

    # 実行
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=30)

    # 終了
    femopt.terminate_all()


def no_fem_2():
    """expression を使わなければ日本語が使えるはず"""

    fem = NoFEM()

    # femopt の準備
    femopt = FEMOpt(fem=fem)

    # parameter の準備
    femopt.add_parameter('radius', 0.5, 0, 1, step=0.1, properties={'unit': 'mm'})
    femopt.add_parameter('ファイ', 180, 90, 270, properties={'unit': 'degree'}, pass_to_fem=False)

    # objective の準備
    femopt.add_objective(obj_2, 'x', args=(femopt.opt, 0))
    femopt.add_objective(obj_2, 'y', args=(femopt.opt, 1))

    # constraints の準備
    femopt.add_constraint(obj_2, 'x(cns)', args=(femopt.opt, 0), upper_bound=0, strict=False)
    femopt.add_constraint(obj_2, 'y(cns)', args=(femopt.opt, 1), upper_bound=0.5, strict=True)

    # 実行
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=30)

    # 終了
    femopt.terminate_all()


def no_fem_3():
    """expression で存在しない引数を指定したらエラーになるはず"""

    fem = NoFEM()

    # femopt の準備
    femopt = FEMOpt(fem=fem)

    # parameter の準備
    femopt.add_parameter('radius', 0.5, 0, 1, step=0.1, properties={'unit': 'mm'})
    femopt.add_parameter('fai_virtual', 180, 90, 270, properties={'unit': 'degree'}, pass_to_fem=False)
    femopt.add_expression('fai', fai3)

    # objective の準備
    femopt.add_objective(obj_2, 'x', args=(femopt.opt, 0))
    femopt.add_objective(obj_2, 'y', args=(femopt.opt, 1))

    # constraints の準備
    femopt.add_constraint(obj_2, 'x(cns)', args=(femopt.opt, 0), upper_bound=0, strict=False)
    femopt.add_constraint(obj_2, 'y(cns)', args=(femopt.opt, 1), upper_bound=0.5, strict=True)

    # 実行
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=30)  # エラーになったら OK, actor が死ぬので terminate_all() は不要というか不可


if __name__ == '__main__':
    no_fem()
    # no_fem_2()
    # no_fem_3()
    femtet()
