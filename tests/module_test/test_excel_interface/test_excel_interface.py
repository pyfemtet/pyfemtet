import os
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


if __name__ == '__main__':
    test_load_single_excel()
