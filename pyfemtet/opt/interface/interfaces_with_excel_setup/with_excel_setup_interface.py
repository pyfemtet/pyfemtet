from __future__ import annotations

from typing import TYPE_CHECKING

from pyfemtet.opt.problem import SupportedVariableTypes
from pyfemtet.opt.interface.interface import AbstractFEMInterface
from pyfemtet.opt.interface.excel_interface.excel_interface import ExcelInterface

if TYPE_CHECKING:
    from pyfemtet.opt.optimizer import AbstractOptimizer


def _get_name(FEMClass: type):
    return FEMClass.__name__.removesuffix('Interface') + 'WithExcelSettingsInterface'


def _class_factory(FEMClass: type[AbstractFEMInterface]):

    class _WithExcelSettingsInterface(FEMClass, ExcelInterface):

        __name__ = _get_name(FEMClass=FEMClass)

        def init_excel(self, *args, **kwargs):
            ExcelInterface.__init__(self, *args, **kwargs)

        def _setup_before_parallel(self):
            ExcelInterface._setup_before_parallel(self)
            FEMClass._setup_before_parallel(self)

        def _setup_after_parallel(self, opt: AbstractOptimizer):
            ExcelInterface._setup_after_parallel(self, opt)
            FEMClass._setup_after_parallel(self, opt)

        def load_variable(self, opt: AbstractOptimizer, raise_if_no_keyword=True) -> None:
            ExcelInterface.load_variables(self, opt, raise_if_no_keyword)
            FEMClass.load_variables(self, opt)

        def load_objectives(self, opt: AbstractOptimizer, raise_if_no_keyword=True) -> None:
            ExcelInterface.load_objectives(self, opt, raise_if_no_keyword)
            FEMClass.load_objectives(self, opt)

        def load_constraints(self, opt: AbstractOptimizer, raise_if_no_keyword=False) -> None:
            ExcelInterface.load_constraints(self, opt, raise_if_no_keyword)
            FEMClass.load_constraints(self, opt)

        def update_parameter(self, x: SupportedVariableTypes) -> None:
            ExcelInterface.update_parameter(self, x)
            FEMClass.update_parameter(self, x)

        def update(self) -> None:
            ExcelInterface.update(self)
            FEMClass.update(self)

        def close(self):
            ExcelInterface.close(self)
            FEMClass.close(self)

    return _WithExcelSettingsInterface


def _debug_1():
    from pyfemtet.opt.interface.femtet_interface import FemtetInterface
    FemtetWithExcelSettingsInterface = _class_factory(FemtetInterface)
    print(FemtetWithExcelSettingsInterface)


if __name__ == '__main__':
    _debug_1()
