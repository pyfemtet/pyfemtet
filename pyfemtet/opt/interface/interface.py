from __future__ import annotations
from typing import TYPE_CHECKING

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

from pyfemtet._util.dask_util import *
from pyfemtet.logger import get_module_logger
from pyfemtet.opt.problem import *

logger = get_module_logger('opt.interface', False)

if TYPE_CHECKING:
    from pyfemtet.opt.optimizer import AbstractOptimizer

__all__ = [
    'AbstractFEMInterface',
    'NoFEM',
]


class AbstractFEMInterface:

    kwargs: dict = {}
    _load_problem_from_fem: bool = False

    # ===== update =====

    def update_parameter(self, x: TrialInput) -> None:
        raise NotImplementedError

    def update(self) -> None:
        raise NotImplementedError

    # ===== Function =====

    @property
    def _object_pass_to_fun(self):
        return self

    # ===== dask util =====

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

    # ===== setup =====

    def _setup_before_parallel(self) -> None:
        pass

    def _setup_after_parallel(self) -> None:
        pass

    def _check_param_and_raise(self, prm_name) -> None:
        pass

    def load_variables(self, opt: AbstractOptimizer):
        pass

    def load_objectives(self, opt: AbstractOptimizer):
        pass

    def load_constraints(self, opt: AbstractOptimizer):
        pass

    def close(self, *args, **kwargs):  # context manager による予約語
        pass

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _check_using_fem(self, fun: callable) -> bool:
        return False

    # ===== postprocessing after recording =====

    def _create_postprocess_args(self) -> dict[str, ...]:
        return {}

    @staticmethod
    def _postprocess_after_recording(
            dask_scheduler,
            trial_name: str,
            **kwargs
    ) -> ...:  # _postprocess_after_recording
        pass


class COMInterface(AbstractFEMInterface):

    com_members = {}

    def __getstate__(self):
        """Pickle するメンバーから COM を除外する"""
        state = self.__dict__.copy()
        for key in self.com_members.keys():
            del state[key]
        return state

    def __setstate__(self, state):
        """UnPickle 時に COM を再構築する

        ただしメインプロセスでしか呼ばれない模様
        dask のバージョン依存？
        """
        CoInitialize()
        for key, value in self.com_members.items():
            state.update({key: Dispatch(value)})
        self.__dict__.update(state)


class NoFEM(AbstractFEMInterface):

    def update_parameter(self, x: TrialInput) -> None:
        return None

    def update(self) -> None:
        return None
