import os

from pyfemtet.opt import FEMOpt, FemtetInterface, NoFEM


here, me = os.path.split(__file__)


def x(Femtet):
    return 1


def test_parametric():
    femprj_path = os.path.join(here, f'{me.replace(".py", ".femprj")}')
    fem = FemtetInterface(
        femprj_path=femprj_path,
        use_parametric_as_objective=True,
    )

    femopt = FEMOpt(fem=fem)
    femopt.add_parameter('w', 0.5, 0.1, 1)
    femopt.add_parameter('d', 0.5, 0.1, 1)
    femopt.add_parameter('h', 0.5, 0.1, 1)

    femopt.add_objective(x, 'manual objective')

    femopt.opt.seed = 42
    femopt.optimize(n_trials=9, n_parallel=3)


if __name__ == '__main__':
    test_parametric()
