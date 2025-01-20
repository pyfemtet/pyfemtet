from typing import TYPE_CHECKING
import platform

from pyfemtet.opt.interface._base import FEMInterface
from pyfemtet.opt.interface._base import NoFEM

if (platform.system() == 'Windows') or TYPE_CHECKING:
    # win32
    from pyfemtet.opt.interface._femtet import FemtetInterface
    from pyfemtet.opt.interface._femtet_with_sldworks import FemtetWithSolidworksInterface
    from pyfemtet.opt.interface._femtet_with_nx import FemtetWithNXInterface
    from pyfemtet.opt.interface._excel_interface import ExcelInterface

else:
    class NotAvailableInWindows:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('This feature is Windows only.')


    class FemtetInterface(NotAvailableInWindows):
        pass
    class FemtetWithSolidworksInterface(FemtetInterface):
        pass
    class FemtetWithNXInterface(FemtetInterface):
        pass
    class ExcelInterface(NotAvailableInWindows):
        pass

from pyfemtet.opt.interface._surrogate._base import SurrogateModelInterfaceBase
from pyfemtet.opt.interface._surrogate._singletaskgp import PoFBoTorchInterface

__all__ = [
    'FEMInterface',
    'NoFEM',
    'FemtetInterface',
    'FemtetWithSolidworksInterface',
    'FemtetWithNXInterface',
    'SurrogateModelInterfaceBase',
    'PoFBoTorchInterface',
]
