import os

from pyfemtet.opt.variable_manager import *
from pyfemtet.opt.interface.excel_interface import ExcelInterface
from pyfemtet.opt.optimizer import AbstractOptimizer


def test_load_single_excel():

    opt = AbstractOptimizer()

    fem = ExcelInterface(
        input_xlsm_path=os.path.join(os.path.dirname(__file__), 'test_excel_interface.xlsm'),
        input_sheet_name='設計変数',
        output_sheet_name='目的関数',
        constraint_sheet_name='拘束関数',
        visible=True,
        interactive=True,
        display_alerts=True,
    )

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


if __name__ == '__main__':
    test_load_single_excel()
