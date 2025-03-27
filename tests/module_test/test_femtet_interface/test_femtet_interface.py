from contextlib import closing

import numpy as np

from tests import get
from pyfemtet.opt.interface import FemtetInterface
from pyfemtet.opt.optimizer import AbstractOptimizer


def test_femtet_interface():

    fem = FemtetInterface(
        femprj_path=get(__file__, 'test_femtet_interface.femprj'),
    )

    with closing(fem):
        fem.update_parameter({'y': 2})
        fem.update()


def test_run_femtet_interface():

    fem = FemtetInterface(
        femprj_path=get(__file__, 'test_femtet_interface.femprj'),
    )

    with closing(fem):
        opt = AbstractOptimizer()
        opt.fem = fem
        fem.use_parametric_output_as_objective(1, 'minimize')
        opt._load_problem_from_fem()
        assert ([name for name, obj in opt.objectives.items()] == ['0: 定常解析 / 温度[deg] / 最大値 / 全てのボディ属性'])

        fem.update_parameter({'y': 2})
        fem.update()
        assert np.allclose(
            [obj.eval(fem) for obj in opt.objectives.values()],
            [100],
            rtol=0.01,
        )


if __name__ == '__main__':
    test_run_femtet_interface()
