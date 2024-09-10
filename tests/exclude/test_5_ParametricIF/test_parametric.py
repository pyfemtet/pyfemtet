import os

from pyfemtet.opt import FEMOpt, FemtetInterface, NoFEM


here, me = os.path.split(__file__)


def test_parametric():
    femprj_path = os.path.join(here, f'{me.replace(".py", ".femprj")}')
    fem = FemtetInterface(
        femprj_path=femprj_path,
        parametric_output_indexes_use_as_objective={
            0: 'minimize',
            1: 'maximize',
            2: 0.0
        },
    )

    femopt = FEMOpt(fem=fem)
    femopt.add_parameter('w', 0.5, 0.1, 1)
    femopt.add_parameter('d', 0.5, 0.1, 1)
    femopt.add_parameter('h', 0.5, 0.1, 1)

    femopt.opt.seed = 42
    femopt.optimize(n_trials=9, n_parallel=3)
    femopt.terminate_all()


if __name__ == '__main__':
    test_parametric()
