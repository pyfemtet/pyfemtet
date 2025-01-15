from typing import TYPE_CHECKING
from pyfemtet._imports import _LazyImport, _lazy_class_factory

from pyfemtet.opt.interface._base import FEMInterface
from pyfemtet.opt.interface._base import NoFEM

if TYPE_CHECKING:
    from pyfemtet.opt.interface._femtet import FemtetInterface
    from pyfemtet.opt.interface._femtet_with_sldworks import FemtetWithSolidworksInterface
    from pyfemtet.opt.interface._femtet_with_nx import FemtetWithNXInterface
    from pyfemtet.opt.interface._surrogate import PoFBoTorchInterface
    from pyfemtet.opt.interface._excel_interface import ExcelInterface

else:
    FemtetInterface = _lazy_class_factory(_LazyImport('pyfemtet.opt.interface._femtet'), 'FemtetInterface')
    FemtetWithSolidworksInterface = _lazy_class_factory(_LazyImport('pyfemtet.opt.interface._femtet_with_sldworks'), 'FemtetWithSolidworksInterface')
    FemtetWithNXInterface = _lazy_class_factory(_LazyImport('pyfemtet.opt.interface._femtet_with_nx'), 'FemtetWithNXInterface')
    PoFBoTorchInterface = _lazy_class_factory(_LazyImport('pyfemtet.opt.interface._surrogate'), 'PoFBoTorchInterface')
    ExcelInterface = _lazy_class_factory(_LazyImport('pyfemtet.opt.interface._excel_interface'), 'ExcelInterface')



__all__ = [
    'FEMInterface',
    'NoFEM',
    'FemtetInterface',
    'FemtetWithNXInterface',
    'FemtetWithSolidworksInterface',
    'PoFBoTorchInterface',
]
