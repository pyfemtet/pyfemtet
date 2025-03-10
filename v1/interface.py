from __future__ import annotations
from typing import TYPE_CHECKING

import os

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
from v1.dask_util import *
from v1.exceptions import *
from v1.logger import get_module_logger

logger = get_module_logger('opt.interface', False)

if TYPE_CHECKING:
    from v1.optimizer import AbstractOptimizer


__all__ = [
    'AbstractFEMInterface',
    'FemtetInterface',
    'NoFEM',
]



class AbstractFEMInterface:

    _load_problem_from_fem: bool = False

    def update_parameter(self, x: TrialInput) -> None:
        raise NotImplementedError

    def update(self) -> None:
        raise NotImplementedError

    @staticmethod
    def _get_worker_space() -> str:
        worker = get_worker()
        if worker is None:
            return os.getcwd()
        else:
            return worker.local_directory

    @staticmethod
    def _distribute_files(paths: list[str]) -> None:

        client = get_client()
        if client is None:
            return

        for path in paths:

            if not os.path.exists(path):
                raise FileNotFoundError

            client.upload_file(path, load=False)

    def _setup_after_parallel(self) -> None:
        pass

    def load_variables(self, opt: AbstractOptimizer):
        pass

    def load_objectives(self, opt: AbstractOptimizer):
        pass

    def load_constraints(self, opt: AbstractOptimizer):
        pass

    def close(self):  # context manager による予約語
        pass

    def _check_using_fem(self, fun: callable) -> bool:
        return False


class COMInterface(AbstractFEMInterface):

    com_members = {}

    def __getstate__(self):
        """Pickle するメンバーから COM を除外する"""
        state = self.__dict__.copy()
        for key in self.com_members.keys():
            del state[key]
        return state

    def __setstate__(self, state):
        """UnPickle 時に COM を再構築する"""
        CoInitialize()
        for key, value in self.com_members.items():
            state.update({key: Dispatch(value)})
        self.__dict__.update(state)


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


class NoFEM(AbstractFEMInterface):

    def update_parameter(self, x: TrialInput) -> None:
        return None

    def update(self) -> None:
        return None
