import os
import sys
import subprocess

import pytest

from pyfemtet.opt.interface import FemtetWithExcelSettingsInterface
from pyfemtet.opt.optimizer import AbstractOptimizer

from tests import get
from tests.utils.closing import closing


# pytest 特有の windows fatal exception 対策
def _run(fun_name):
    here, filename = os.path.split(__file__)
    module_name = filename.removesuffix('.py')

    subprocess.run(
        f'{sys.executable} '
        f'-c '
        f'"'
        f'import os;'
        f'import sys;'
        f'sys.path.append(os.getcwd());'
        f'import {module_name} as tst;'
        f'tst.{fun_name}()'
        f'"',
        cwd=os.path.abspath(here),
        shell=True,
    ).check_returncode()


def impl_load_femtet_with_excel_settings():
    opt = AbstractOptimizer()

    fem: FemtetWithExcelSettingsInterface = FemtetWithExcelSettingsInterface(
        femprj_path=get(__file__, 'test_femtet_interface.femprj'),
    )

    fem.init_excel(
        input_xlsm_path=get(__file__, 'test_excel_interface.xlsm'),
        input_sheet_name='input',
        output_sheet_name='output',
    )

    with closing(fem):

        opt.fem = fem

        opt._load_problem_from_fem()
        opt._finalize_history()
        fem._setup_before_parallel()
        fem._setup_after_parallel(opt)

        fem.update_parameter(dict(x=2))
        fem.update()

        for obj_name, obj in opt.objectives.items():
            print(obj_name, obj.eval(fem))
            if obj_name == 'output_formula':
                assert obj.eval(fem) == 2
            elif obj_name == '0: 定常解析 / 温度[deg] / 最大値 / 全てのボディ属性':
                assert obj.eval(fem) == 100
            else:
                assert False, '予期しない目的変数名'


@pytest.mark.femtet
@pytest.mark.excel
def test_load_femtet_with_excel_settings():
    _run(impl_load_femtet_with_excel_settings.__name__)


if __name__ == '__main__':
    test_load_femtet_with_excel_settings()
