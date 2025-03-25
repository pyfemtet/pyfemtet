from __future__ import annotations

from typing import TYPE_CHECKING

import os
import json
import subprocess

from pyfemtet._i18n import Msg

from pyfemtet.opt.interface.interface import AbstractFEMInterface
from pyfemtet.opt.exceptions import *

if TYPE_CHECKING:
    from pyfemtet.opt.optimizer import AbstractOptimizer


here, me = os.path.split(__file__)
JOURNAL_PATH = os.path.abspath(os.path.join(here, 'update_model.py'))


class NXInterface(AbstractFEMInterface):
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
            prt_path,
            export_curves: bool or None = None,
            export_surfaces: bool or None = None,
            export_solids: bool or None = None,
            export_flattened_assembly: bool or None = None,
    ):
        # check NX installation
        self.run_journal_path = os.path.join(os.environ.get('UGII_BASE_DIR'), 'NXBIN', 'run_journal.exe')
        if not os.path.isfile(self.run_journal_path):
            raise FileNotFoundError(Msg.ERR_RUN_JOURNAL_NOT_FOUND)

        self.prt_path = os.path.abspath(prt_path)
        assert os.path.isfile(self.prt_path)
        self.export_curves = export_curves
        self.export_surfaces = export_surfaces
        self.export_solids = export_solids
        self.export_flattened_assembly = export_flattened_assembly

    def _setup_before_parallel(self):
        self._distribute_files([self.prt_path])

    def _setup_after_parallel(self, opt: AbstractOptimizer = None):

        # get suffix
        suffix = self._get_worker_index_from_optimizer(opt)

        # rename and get worker path
        self.sldprt_path = self._rename_and_get_path_on_worker_space(
            self.prt_path,
            suffix,
        )

    def _export_updated_xt(self, x_t_path) -> None:
        """Update .x_t"""

        # 前のが存在するならば消しておく
        if os.path.isfile(x_t_path):
            os.remove(x_t_path)

        # 変数の json 文字列を作る
        str_json = json.dumps(self.current_prm_values)

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
                x_t_path,
                dumped_json_export_settings,
            ],
            env=env,
            shell=True,
            cwd=os.path.dirname(self.prt_path)
        )

        # この時点で x_t ファイルがなければ NX がモデル更新に失敗しているはず
        if not os.path.isfile(x_t_path):
            raise ModelError

    def update(self) -> None:
        raise NotImplementedError


def _debug_1():

    fem = NXInterface(
        prt_path=os.path.join(os.path.dirname(__file__), 'model1.prt'),
        export_solids=True,
    )
    fem._setup_before_parallel()
    fem._setup_after_parallel()
    fem.update_parameter({'x': 20})
    fem._export_updated_xt(os.path.join(os.path.dirname(__file__), 'model1.x_t'))


if __name__ == '__main__':
    _debug_1()
