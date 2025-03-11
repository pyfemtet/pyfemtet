from __future__ import annotations

try:
    # noinspection PyUnresolvedReferences
    from pythoncom import CoInitialize, CoUninitialize
    from win32com.client import Dispatch, Constants, constants
except ModuleNotFoundError:
    CoInitialize = lambda: None
    CoUninitialize = lambda: None
    Dispatch = type('NoDispatch', (object,), {})
    Constants = type('NoConstants', (object,), {})
    constants = Constants()

from v1.problem import *
from v1.logger import get_module_logger
from v1.interface.interface import COMInterface


logger = get_module_logger('opt.interface', False)



class FemtetInterface(COMInterface):

    com_members = {'Femtet': 'FemtetMacro.Femtet'}

    def __init__(self):
        self.Femtet = Dispatch('FemtetMacro.Femtet')

    def update_parameter(self, x: TrialInput) -> None:
        pass

    def update(self) -> None:
        pass

    def close(self):  # context manager による予約語
        print('Femtet を終了する')
