import platform

from ._base_interface import AbstractFEMInterface, NoFEM

if platform.system() == 'Windows':
    from ._femtet_interface import FemtetInterface
    from ._femtet_with_nx_interface import FemtetWithNXInterface
    from ._femtet_with_solidworks import FemtetWithSolidworksInterface
    from ._excel_interface import ExcelInterface
    from ._with_excel_settings import *

from ._surrogate_model_interface import AbstractSurrogateModelInterfaceBase
from ._surrogate_model_interface import BoTorchInterface
from ._surrogate_model_interface import PoFBoTorchInterface


__all__ = [
    'AbstractFEMInterface',
    'NoFEM',
    'FemtetInterface',
    'FemtetWithNXInterface',
    'FemtetWithSolidworksInterface',
    'ExcelInterface',
    'FemtetWithExcelSettingsInterface',
    'AbstractSurrogateModelInterfaceBase',
    'BoTorchInterface',
    'PoFBoTorchInterface',
]
