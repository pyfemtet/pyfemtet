from pyfemtet.opt.interface._base_interface import AbstractFEMInterface


class DummyInterface:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(f'{type(self).__name__} is only for Windows OS.')


class FemtetInterface(DummyInterface, AbstractFEMInterface): pass


class FemtetWithNXInterface(DummyInterface, AbstractFEMInterface): pass


class FemtetWithSolidworksInterface(DummyInterface, AbstractFEMInterface): pass


class ExcelInterface(DummyInterface, AbstractFEMInterface): pass
