import platform

from ._base_interface import AbstractFEMInterface, NoFEM

if platform.system() == 'Windows':
    from ._femtet_interface import FemtetInterface
    from ._femtet_with_nx_interface import FemtetWithNXInterface
    from ._femtet_with_solidworks import FemtetWithSolidworksInterface
    from ._excel_interface import ExcelInterface
    from ._with_excel_settings import *
    from ._with_excel_settings import __all__ as _with_excel_settings__all__
else:
    class DummyInterface:
        def __init__(self, args, kwargs):
            raise RuntimeError(f'{type(self).__name__} is only for Windows OS.')

    class FemtetInterface(DummyInterface, AbstractFEMInterface): pass
    class FemtetWithNXInterface(DummyInterface, AbstractFEMInterface): pass
    class FemtetWithSolidworksInterface(DummyInterface, AbstractFEMInterface): pass
    class ExcelInterface(DummyInterface, AbstractFEMInterface): pass


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
    'AbstractSurrogateModelInterfaceBase',
    'BoTorchInterface',
    'PoFBoTorchInterface',
]

if platform.system() == 'Windows':
    __all__.extend(_with_excel_settings__all__)
