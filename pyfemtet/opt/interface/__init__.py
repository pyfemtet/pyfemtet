import platform

from .interface import AbstractFEMInterface, NoFEM

if platform.system() == 'Windows':
    from .femtet_interface import FemtetInterface
