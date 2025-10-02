from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Any

import os
import tempfile

import shutil
import pandas as pd

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
from pyfemtet.opt.problem.problem import *

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
    current_prm_values: TrialInput
    _tmp_dir: tempfile.TemporaryDirectory
    _file_suffix: str

    # ===== update =====

    def update_parameter(self, x: TrialInput) -> None:
        # FEM オブジェクトに与えられた変数を設定する。
        #   目的は Function 内でユーザーが FEM オブジェクト経由で
        #   変数を取得できるようにするためなので、各具象クラスでは
        #   FEM オブジェクトから新しい変数を取得できるように
        #   することが望ましい
        self.current_prm_values = x

    def update(self) -> None:
        # 現在の設計変数に基づいて solve を行い、
        # Objective が正しく値を計算できるようにする
        raise NotImplementedError

    # ===== Function =====

    @property
    def object_pass_to_fun(self):
        """The object pass to the first argument of user-defined objective functions.

        Returns:
            self (AbstractFEMInterface)
        """
        return self

    # ===== dask util =====

    def _get_file_suffix(self, opt: AbstractOptimizer | None) -> str:

        file_suffix = 'copy'

        if opt is not None:
            if opt._worker_index is not None:
                file_suffix = file_suffix + f'_{opt._worker_index}'

        if hasattr(self, '_file_suffix'):
            if self._file_suffix is not None:
                file_suffix = file_suffix + f'_{self._file_suffix}'

        return file_suffix

    def _rename_and_get_path_on_worker_space(self, orig_path, suffix, ignore_no_exist=False) -> str:
        # 与えられた path と同名のファイルを
        # worker_space から探し
        # suffix を付与して rename し
        # その renamed path を返す関数

        worker_space = self._get_worker_space()

        src_path = os.path.join(worker_space, os.path.basename(orig_path))
        p1_, p2_ = os.path.splitext(src_path)
        dst_path_ = p1_ + '_' + suffix + p2_

        if os.path.isfile(src_path):
            os.rename(src_path, dst_path_)

        elif not ignore_no_exist:
            raise FileNotFoundError(f'{src_path} is not found.')

        return dst_path_

    def _get_worker_space(self) -> str:
        worker = get_worker()
        if worker is None:
            assert hasattr(self, '_tmp_dir'), 'Internal Error! Run _distribute_files() first!'
            return self._tmp_dir.name
        else:
            return worker.local_directory

    def _distribute_files(self, paths: list[str], scheduler_address=None) -> None:

        # executor 向け
        self._copy_to_temp_space(paths)

        # dask worker 向け
        client = get_client(scheduler_address)
        if client is not None:
            for path in paths:
                if not os.path.exists(path):
                    raise FileNotFoundError
                client.upload_file(path, load=False)

    def _verify_tmp_dir(self):
        should_process = False

        if not hasattr(self, '_tmp_dir'):
            should_process = True

        elif self._tmp_dir is None:
            should_process = True

        if not should_process:
            return

        # dask worker space のように使える一時フォルダを作成する
        # Python プロセス終了時に（使用中のプロセスがなければ）
        # 削除されるので、重大なものでなければ後処理は不要
        tmp_dir = tempfile.TemporaryDirectory(prefix='pyfemtet-')
        self._tmp_dir = tmp_dir

    def _copy_to_temp_space(self, paths: list[str]) -> None:

        self._verify_tmp_dir()

        # client.upload_file 相当の処理を行う
        for path in paths:
            shutil.copy(path, self._tmp_dir.name)

    # ===== setup =====

    def _setup_before_parallel(self, scheduler_address=None) -> None:
        pass

    def _setup_after_parallel(self, opt: AbstractOptimizer) -> None:
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
    def _check_using_fem(self, fun: Callable) -> bool:
        return False

    # ===== postprocessing after recording =====

    def _create_postprocess_args(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def _postprocess_after_recording(
            dask_scheduler,
            trial_name: str,
            df: pd.DataFrame,
            **kwargs
    ) -> ...:  # _postprocess_after_recording
        pass

    # ===== others =====

    # noinspection PyMethodMayBeStatic
    def _get_additional_data(self) -> dict:
        return dict()


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

    def update(self) -> None:
        return None
