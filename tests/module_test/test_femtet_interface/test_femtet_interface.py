from pyfemtet._util.closing import closing

import numpy as np
from win32com.client import Dispatch
from femtetutils.util import execute_femtet

from tests import get
from pyfemtet.opt.interface import FemtetInterface
from pyfemtet.opt.optimizer import AbstractOptimizer

import pytest


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


@pytest.mark.skip(reason='requires manual control')
def test_femtet_interface_api_calling():

    fem = FemtetInterface(allow_without_project=True)

    # Femtet で新しいプロジェクトを開き、
    # ダイアログが出るようなマクロ実行を行う
    fem.Femtet.OpenNewProject()
    point = Dispatch('FemtetMacro.GaudiPoint')
    point.X = 0.
    point.Y = 0.
    point.Z = 0.
    fem.Femtet.Gaudi.CreateVertex(point)
    fem.close(force=False, timeout=3)


def test_femtet_always_open_copy_flag():
    execute_femtet()

    opt = AbstractOptimizer()

    fem = FemtetInterface(
        femprj_path=get(__file__, 'test_femtet_interface.femprj'),
        always_open_copy=True,
        connect_method='existing',
    )

    fem._setup_before_parallel()
    fem._setup_after_parallel(opt)

    print(fem.Femtet.ProjectTitle)
    assert fem.Femtet.ProjectTitle != 'test_femtet_interface'

    fem.close()

    fem = FemtetInterface(
        femprj_path=get(__file__, 'test_femtet_interface.femprj'),
        always_open_copy=False,
        connect_method='existing',
    )

    fem._setup_before_parallel()
    fem._setup_after_parallel(opt)

    print(fem.Femtet.Project)
    assert fem.Femtet.ProjectTitle == 'test_femtet_interface'

    fem.close()


if __name__ == '__main__':
    # test_run_femtet_interface()
    # test_femtet_interface_api_calling()
    test_femtet_always_open_copy_flag()
