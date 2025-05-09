from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import platform


__all__ = []  # appended later


if platform.system() == 'Windows':

    from pyfemtet.opt.interface._excel_interface import ExcelInterface

    from pyfemtet.opt.interface._femtet_interface import FemtetInterface
    from pyfemtet.opt.interface._femtet_with_nx_interface import FemtetWithNXInterface
    from pyfemtet.opt.interface._femtet_with_solidworks import FemtetWithSolidworksInterface

    from .with_excel_settings import _class_factory

    if TYPE_CHECKING:
        class FemtetWithExcelSettingsInterface(FemtetInterface, ExcelInterface):
            init_excel = ExcelInterface
    else:
        FemtetWithExcelSettingsInterface = _class_factory(FemtetInterface)
    __all__.append('FemtetWithExcelSettingsInterface')

    if TYPE_CHECKING:
        class FemtetWithNXWithExcelSettingsInterface(FemtetWithNXInterface, ExcelInterface):
            def init_excel(self, *args, **kwargs):
                pass
    else:
        FemtetWithNXWithExcelSettingsInterface = _class_factory(FemtetWithNXInterface)
    __all__.append('FemtetWithNXWithExcelSettingsInterface')

    if TYPE_CHECKING:
        class FemtetWithSolidworksWithExcelSettingsInterface(FemtetWithSolidworksInterface, ExcelInterface):
            def init_excel(self, *args, **kwargs):
                pass
    else:
        FemtetWithSolidworksWithExcelSettingsInterface = _class_factory(FemtetWithSolidworksInterface)
    __all__.append('FemtetWithSolidworksWithExcelSettingsInterface')


from pyfemtet.opt.interface._surrogate_model_interface import BoTorchInterface, PoFBoTorchInterface

if TYPE_CHECKING:
    class BoTorchWithExcelSettingsInterface(BoTorchInterface, ExcelInterface):
        def init_excel(self, *args, **kwargs):
            pass
else:
    BoTorchWithExcelSettingsInterface = _class_factory(BoTorchInterface)
__all__.append('BoTorchWithExcelSettingsInterface')

if TYPE_CHECKING:
    class PoFBoTorchWithExcelSettingsInterface(PoFBoTorchInterface, ExcelInterface):
        def init_excel(self, *args, **kwargs):
            pass
else:
    PoFBoTorchWithExcelSettingsInterface = _class_factory(PoFBoTorchInterface)
__all__.append('PoFBoTorchWithExcelSettingsInterface')
