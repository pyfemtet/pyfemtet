from __future__ import annotations

from typing import TYPE_CHECKING

import os
import json
import subprocess

# noinspection PyUnresolvedReferences
from pywintypes import com_error

from pyfemtet._i18n import _
from pyfemtet.opt.interface._base_interface import AbstractFEMInterface
from pyfemtet.opt.interface._femtet_interface import FemtetInterface
from pyfemtet.opt.exceptions import *
from pyfemtet.opt.problem.variable_manager import *

if TYPE_CHECKING:
    from pyfemtet.opt.optimizer import AbstractOptimizer


here, me = os.path.split(__file__)
JOURNAL_PATH = os.path.abspath(os.path.join(here, 'update_model.py'))


# NX 単体で Interface 化する予定がないのでモジュール分割しない
class _NXInterface(AbstractFEMInterface):

    def __init__(
            self,
            prt_path: str,
            export_curves: bool or None = None,
            export_surfaces: bool or None = None,
            export_solids: bool or None = None,
            export_flattened_assembly: bool or None = None,
    ):
        # check NX installation
        self.run_journal_path = os.path.join(os.environ.get('UGII_BASE_DIR'), 'NXBIN', 'run_journal.exe')
        if not os.path.isfile(self.run_journal_path):
            raise FileNotFoundError(_(
                en_message='`run_journal.exe` is not found. '
                           'Please check:\n'
                           '- NX is installed.\n'
                           '- The environment variable `UGII_BASE_DIR` is set.\n'
                           '- `<UGII_BASE_DIR>\\NXBIN\\run_journal.exe` exists.\n',
                jp_message='「run_journal.exe」 が見つかりませんでした。'
                           '以下のことを確認してください。\n'
                           '- NX がインストールされている\n'
                           '- 環境変数 UGII_BASE_DIR が設定されている\n'
                           '- <UGII_BASE_DIR>\\NXBIN\\run_journal.exe が存在する'
            ))

        self.prt_path = os.path.abspath(prt_path)
        assert os.path.isfile(self.prt_path)
        self._prt_path = self.prt_path
        self.export_curves = export_curves
        self.export_surfaces = export_surfaces
        self.export_solids = export_solids
        self.export_flattened_assembly = export_flattened_assembly

    def _setup_before_parallel(self, scheduler_address=None) -> None:
        self._distribute_files([self.prt_path], scheduler_address)

    def _setup_after_parallel(self, opt: AbstractOptimizer = None) -> None:
        # get suffix
        suffix = self._get_file_suffix(opt)

        # rename and get worker path
        self.prt_path = self._rename_and_get_path_on_worker_space(
            self._prt_path,
            suffix,
        )

    def _export_xt(self, xt_path) -> None:
        """Update .x_t"""

        # 前のが存在するならば消しておく
        if os.path.isfile(xt_path):
            os.remove(xt_path)

        # 変数の json 文字列を作る
        str_json = json.dumps(
            {name: variable.value for name, variable
             in self.current_prm_values.items()})

        # create dumped json of export settings
        tmp_dict = dict(
            include_curves=self.export_curves,
            include_surfaces=self.export_surfaces,
            include_solids=self.export_solids,
            flatten_assembly=self.export_flattened_assembly,
        )
        dumped_json_export_settings = json.dumps(tmp_dict)

        # NX journal を使ってモデルを編集する
        env = os.environ.copy()
        subprocess.run(
            [
                self.run_journal_path,  # run_journal.exe
                JOURNAL_PATH,  # update_model.py
                '-args',
                self.prt_path,
                str_json,
                xt_path,
                dumped_json_export_settings,
            ],
            env=env,
            shell=True,
            cwd=os.path.dirname(self.prt_path)
        )

        # この時点で x_t ファイルがなければ NX がモデル更新に失敗しているはず
        if not os.path.isfile(xt_path):
            raise ModelError


class FemtetWithNXInterface(FemtetInterface, _NXInterface):
    """Control Femtet and NX.

    Using this class, you can import CAD files created
    in NX through the Parasolid format into a
    Femtet project. It allows you to pass design
    variables to NX, update the model, and
    perform analysis using the updated model in Femtet.

    Args:
        prt_path (str):
            The path to .prt file containing the
            CAD data from which the import is made.

        export_curves(bool or None, optional):
            Defaults to None.
        export_surfaces(bool or None, optional):
            Defaults to None.
        export_solids(bool or None, optional):
            Defaults to None.
        export_flattened_assembly(bool or None, optional):
            Defaults to None.

    Notes:
        ```export_*``` arguments sets
        parasolid export setting of NX.
        If None, PyFemtet does not change
        the current setting of NX.

        It is recommended not to change these values
        from the settings used when exporting the
        Parasolid that was imported into Femtet.

    """

    def __init__(
            self,
            prt_path: str,
            femprj_path: str = None,
            model_name: str = None,
            connect_method: str = "auto",  # dask worker では __init__ の中で 'new' にするので super() の引数にしない。（しても意味がない）
            save_pdt: str = "all",  # 'all' or None
            strictly_pid_specify: bool = True,  # dask worker では True にしたいので super() の引数にしない。
            allow_without_project: bool = False,  # main でのみ True を許容したいので super() の引数にしない。
            open_result_with_gui: bool = True,
            parametric_output_indexes_use_as_objective: dict[int, str | float] = None,
            always_open_copy=False,
            export_curves: bool or None = None,
            export_surfaces: bool or None = None,
            export_solids: bool or None = None,
            export_flattened_assembly: bool or None = None,
    ):

        FemtetInterface.__init__(
            self,
            femprj_path=femprj_path,
            model_name=model_name,
            connect_method=connect_method,
            save_pdt=save_pdt,
            strictly_pid_specify=strictly_pid_specify,
            allow_without_project=allow_without_project,
            open_result_with_gui=open_result_with_gui,
            parametric_output_indexes_use_as_objective=parametric_output_indexes_use_as_objective,
            always_open_copy=always_open_copy,
        )

        _NXInterface.__init__(
            self,
            prt_path=prt_path,
            export_curves=export_curves,
            export_surfaces=export_surfaces,
            export_solids=export_solids,
            export_flattened_assembly=export_flattened_assembly,
        )

    def _setup_before_parallel(self, scheduler_address=None):
        FemtetInterface._setup_before_parallel(self, scheduler_address)
        _NXInterface._setup_before_parallel(self, scheduler_address)

    def _setup_after_parallel(self, opt: AbstractOptimizer = None):
        FemtetInterface._setup_after_parallel(self, opt)
        _NXInterface._setup_after_parallel(self, opt)

    def update_model(self):

        # 競合しないよう保存先を temp にしておく
        worker_space = self._get_worker_space()
        xt_path = os.path.join(worker_space, 'temp.x_t')

        # export parasolid
        self._export_xt(xt_path)

        # LastXTPath を更新する
        try:
            self.Femtet.Gaudi.LastXTPath = xt_path
        except (KeyError, AttributeError, com_error):
            raise RuntimeError('This feature is available from Femtet version 2023.2. Please update Femtet.')

        # update_parameter で変数は更新されているので
        # ここでモデルを完全に再構築できる
        FemtetInterface.update_model(self)


def _debug_1():

    x = Variable()
    x.name = 'x'
    x.value = 20

    fem = _NXInterface(
        prt_path=os.path.join(os.path.dirname(__file__), 'model1.prt'),
        export_solids=True,
    )
    fem._setup_before_parallel()
    fem._setup_after_parallel()
    fem.update_parameter(dict(x=x))
    fem._export_xt(os.path.join(os.path.dirname(__file__), 'model1.x_t'))


if __name__ == '__main__':
    _debug_1()
