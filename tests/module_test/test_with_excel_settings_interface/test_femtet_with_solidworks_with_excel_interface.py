import os
import sys
import subprocess

import pytest

from pyfemtet.opt.interface import FemtetWithSolidworksWithExcelSettingsInterface
from pyfemtet.opt.optimizer import AbstractOptimizer
from pyfemtet.opt.problem.variable_manager import *

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


def impl_load_femtet_with_sw_with_excel_settings():
    opt = AbstractOptimizer()

    fem: FemtetWithSolidworksWithExcelSettingsInterface = FemtetWithSolidworksWithExcelSettingsInterface(
        femprj_path=get(__file__, 'test_femtet_with_cad_interface.femprj'),
        sldprt_path=get(__file__, 'test_solidworks_interface.sldprt'),
        close_solidworks_on_terminate=True,
    )

    fem.init_excel(
        input_xlsm_path=get(__file__, 'test_excel_interface.xlsm'),
        input_sheet_name='input',
        output_sheet_name='output',
    )

    with closing(fem):

        opt.fem = fem

        opt._refresh_problem()
        opt._finalize_history()
        fem._setup_before_parallel()
        fem._setup_after_parallel(opt)

        variable = Variable()
        variable.name = 'x'
        variable.value = 20

        fem.update_parameter(dict(x=variable))
        fem.update()

        for obj_name, obj in opt.objectives.items():
            print(obj_name, obj.eval(fem))
            if obj_name == 'output_formula':
                assert obj.eval(fem) == 20
            elif obj_name == '0: 定常解析 / 温度[deg] / 最大値 / 全てのボディ属性':
                assert obj.eval(fem) == 0
            else:
                assert False, '予期しない目的変数名'


@pytest.mark.femtet
@pytest.mark.cad
@pytest.mark.excel
@pytest.mark.skip('pytest with Solidworks is unstable')
def test_load_femtet_with_sw_with_excel_settings():
    _run(impl_load_femtet_with_sw_with_excel_settings.__name__)


if __name__ == '__main__':
    test_load_femtet_with_sw_with_excel_settings()
