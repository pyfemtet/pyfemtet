import numpy as np

import pytest

from pyfemtet.opt.problem.variable_manager import *
from pyfemtet.opt.interface._excel_interface import ExcelInterface
from pyfemtet.opt.optimizer import AbstractOptimizer

from tests import get
from tests.utils.closing import closing


@pytest.mark.excel
def test_load_single_excel():

    opt = AbstractOptimizer()

    fem = ExcelInterface(
        input_xlsm_path=get(__file__, 'test_excel_interface.xlsm'),
        input_sheet_name='設計変数',
        output_sheet_name='目的関数',
        constraint_sheet_name='拘束関数',
        visible=True,
        interactive=True,
        display_alerts=True,
    )

    with closing(fem):

        opt.fem = fem

        opt._load_problem_from_fem()

        for var in opt.get_variables(format='raw').values():
            print(f'----- {var.name} ----')
            for attr in dir(var):
                if not attr.startswith('_'):
                    print(attr, getattr(var, attr))

            if var.name == 'x':
                assert isinstance(var, NumericParameter)
                assert var.value == 0.5
                assert var.lower_bound == 0
                assert var.upper_bound == 10
                assert var.step == 1

            elif var.name == 'y':
                assert isinstance(var, NumericParameter)
                assert var.value == 6
                assert var.lower_bound == 1
                assert var.upper_bound == 10
                assert var.step is None

            elif var.name == 'z':
                assert isinstance(var, CategoricalParameter)
                assert var.value == 'A'
                assert var.choices == ['A', 'B', 'C']

            elif var.name == 'no_use_1':
                assert isinstance(var, NumericParameter)
                assert var.value == 3
                assert var.properties['fix'] is True


@pytest.mark.excel
@pytest.mark.femtet
def test_run_multiple_excel():
    opt = AbstractOptimizer()

    fem = ExcelInterface(
        input_xlsm_path=get(__file__, 'test_excel_interface.xlsm'),
        input_sheet_name='input',
        output_sheet_name='output',

        procedure_xlsm_path=get(__file__, 'test_excel_interface.xlsm'),
        procedure_name='update_output_macro',

        setup_xlsm_path=get(__file__, 'test_excel_control_femtet.xlsm'),
        setup_procedure_name='PrePostProcessing.setup',

        teardown_xlsm_path=get(__file__, 'test_excel_control_femtet.xlsm'),
        teardown_procedure_name='PrePostProcessing.teardown',
    )

    # pytest のときのみ setup マクロが実行できない系のエラーが発生したら
    # - pytest なしで実行できることを確認する
    # - Excel のトラストセンターで VBA を有効にする
    # - Excel のトラストセンターで 信頼できる場所に追加する
    with closing(fem):

        opt.fem = fem

        print('loading problems...')
        opt._load_problem_from_fem()
        print('setting up before parallel...')
        fem._setup_before_parallel()
        print('setting up after parallel...')
        fem._setup_after_parallel(opt)

        print('updating...')

        x = Variable()
        x.name = 'x'
        x.value = .7

        y = Variable()
        y.name = 'y'
        y.value = .7

        z = Variable()
        z.name = 'z'
        z.value = .7

        fem.update_parameter(dict(x=x, y=y, z=z))
        fem.update()

        print([obj.eval(fem) for obj in opt.objectives.values()])
        assert np.allclose(
            [obj.eval(fem) for obj in opt.objectives.values()],
            [2.1, 2.1],
            rtol=0.01,
        )


if __name__ == '__main__':
    # test_load_single_excel()
    test_run_multiple_excel()
