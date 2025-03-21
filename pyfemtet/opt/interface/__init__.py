import platform

from pyfemtet.opt.interface.interface import AbstractFEMInterface, NoFEM

if platform.system() == 'Windows':
    from pyfemtet.opt.interface.femtet_interface.femtet_interface import FemtetInterface
