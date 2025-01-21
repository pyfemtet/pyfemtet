import platform
from typing import TYPE_CHECKING

from pyfemtet.opt.interface._base import FEMInterface
from pyfemtet.opt.interface._base import NoFEM

if (platform.system() == 'Windows') or TYPE_CHECKING:
    from pyfemtet.opt.interface._femtet import FemtetInterface
    from pyfemtet.opt.interface._femtet_with_sldworks import FemtetWithSolidworksInterface
    from pyfemtet.opt.interface._femtet_with_nx import FemtetWithNXInterface
    from pyfemtet.opt.interface._excel_interface import ExcelInterface

else:
    class NotAvailableForWindows:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError


    FemtetInterface = type('FemtetInterface', (NotAvailableForWindows,), {})
    FemtetWithSolidworksInterface = type('FemtetWithSolidworksInterface', (FemtetInterface,), {})
    FemtetWithNXInterface = type('FemtetWithNXInterface', (FemtetInterface,), {})
    ExcelInterface = type('FemtetInterface', (NotAvailableForWindows,), {})

from pyfemtet.opt.interface._surrogate._base import SurrogateModelInterfaceBase
from pyfemtet.opt.interface._surrogate._singletaskgp import PoFBoTorchInterface


__all__ =[
    'FEMInterface',
    'NoFEM',
    'FemtetInterface',
    'FemtetWithSolidworksInterface',
    'FemtetWithNXInterface',
    'ExcelInterface',
    'SurrogateModelInterfaceBase',
    'PoFBoTorchInterface',
]
