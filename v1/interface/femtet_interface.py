from __future__ import annotations

import os

try:
    # noinspection PyUnresolvedReferences
    from pythoncom import CoInitialize, CoUninitialize
    from win32com.client import Dispatch, Constants, constants
except ModuleNotFoundError:
    # noinspection PyPep8Naming
    def CoInitialize(): ...
    # noinspection PyPep8Naming
    def CoUninitialize(): ...
    Dispatch = type('NoDispatch', (object,), {})
    Constants = type('NoConstants', (object,), {})
    constants = Constants()

from v1.problem import *
from v1.logger import get_module_logger
from v1.interface.interface import COMInterface


logger = get_module_logger('opt.interface', False)


class FemtetInterface(COMInterface):

    com_members = {'Femtet': 'FemtetMacro.Femtet'}

    def __init__(
            self,
            femprj_path: str = None,
            model_name: str = None,
            connect_method: str = 'auto',  # dask worker では __init__ の中で 'new' にするので super() の引数にしない。（しても意味がない）
            save_pdt: str = 'all',  # 'all' or None
            strictly_pid_specify: bool = True,  # dask worker では True にしたいので super() の引数にしない。
            allow_without_project: bool = False,  # main でのみ True を許容したいので super() の引数にしない。
            open_result_with_gui: bool = True,
    ):

        # 引数の処理
        self.femprj_path = os.path.abspath(femprj_path) if femprj_path is not None else None
        self.model_name = model_name
        self.connect_method = connect_method
        self.allow_without_project = allow_without_project
        self.original_femprj_path = self.femprj_path
        self.open_result_with_gui = open_result_with_gui
        self.save_pdt = save_pdt

        # その他のメンバーの初期化
        self.Femtet = None
        self.quit_when_destruct = False
        self.strictly_pid_specify = strictly_pid_specify
        self._femtet_pid = 0
        self._connected_method = 'unconnected'
        self._parameters = None
        self._max_api_retry = 3
        self._original_autosave_enabled = _get_autosave_enabled()
        _set_autosave_enabled(False)


    def update_parameter(self, x: TrialInput) -> None:
        pass

    def update_model(self) -> None:
        ...

    def update(self) -> None:
        pass

    def close(self):  # context manager による予約語
        print('Femtet を終了する')

    @property
    def _obj_pass_to_function(self):
        return self.Femtet
