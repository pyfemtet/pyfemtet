import os
from pyfemtet.opt import FEMOpt

from pyfemtet.opt.interface._femtet_excel import FemtetWithExcelSettingsInterface

import pytest


@pytest.mark.fem
def test_femtet_with_excel_settings():

    os.chdir(os.path.dirname(__file__))

    fem = FemtetWithExcelSettingsInterface(
        femprj_path='sample.femprj',
        input_xlsm_path='sample.xlsm',
        input_sheet_name='sample',
        parametric_output_indexes_use_as_objective={0: 'minimize'}
    )

    femopt = FEMOpt(fem=fem)

    femopt.optimize(n_trials=4, n_parallel=2, confirm_before_exit=False)

    for e in femopt._opt_exceptions:
        if e is not None:
            raise e


if __name__ == '__main__':
    test_femtet_with_excel_settings()
