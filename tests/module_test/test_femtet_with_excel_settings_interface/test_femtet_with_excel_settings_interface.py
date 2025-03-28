from pyfemtet.opt.interface import FemtetWithExcelSettingsInterface
from pyfemtet.opt.interface import ExcelInterface
from pyfemtet.opt.optimizer import AbstractOptimizer

from tests import get


def test_load_femtet_with_excel_settings():
    opt = AbstractOptimizer()
    fem = FemtetWithExcelSettingsInterface(
        femprj_path=get(__file__, 'test_femtet_interface.femprj'),
    )

    ExcelInterface(


    )

    fem.init_excel(
        input_xlsm_path=get(__file__, 'test_excel_interface.xlsm'),
        input_sheet_name='input',

    )
