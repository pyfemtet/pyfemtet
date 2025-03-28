import platform

from .interface import AbstractFEMInterface, NoFEM

if platform.system() == 'Windows':
    from .femtet_interface import FemtetInterface
    from .femtet_with_nx_interface import FemtetWithNXInterface
    from .femtet_with_solidworks import FemtetWithSolidworksInterface
    from .excel_interface import ExcelInterface
    from .with_excel_settings import *

from .surrogate_model_interface import BoTorchInterface
from .surrogate_model_interface import PoFBoTorchInterface
