import os
import json
import subprocess

import pandas as pd
from dask.distributed import get_worker

from pyfemtet.core import ModelError
from pyfemtet.opt.interface import FemtetInterface, logger
from pyfemtet._message import Msg


here, me = os.path.split(__file__)


class FemtetWithNXInterface(FemtetInterface):
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
        **kwargs:
            For other arguments, please refer to the
            :class:`FemtetInterface` class.

    Notes:
        ```export_*``` arguments sets
        parasolid export setting of NX.
        If None, PyFemtet does not change
        the current setting of NX.

        It is recommended not to change these values
        from the settings used when exporting the
        Parasolid that was imported into Femtet.

    """

    _JOURNAL_PATH = os.path.abspath(os.path.join(here, 'update_model.py'))

    def __init__(
            self,
            prt_path,
            export_curves: bool or None = None,
            export_surfaces: bool or None = None,
            export_solids: bool or None = None,
            export_flattened_assembly: bool or None = None,
            **kwargs
    ):
        # check NX installation
        self.run_journal_path = os.path.join(os.environ.get('UGII_BASE_DIR'), 'NXBIN', 'run_journal.exe')
        if not os.path.isfile(self.run_journal_path):
            raise FileNotFoundError(Msg.ERR_RUN_JOURNAL_NOT_FOUND)

        # 引数の処理
        # dask サブプロセスのときは prt_path を worker space から取るようにする
        try:
            worker = get_worker()
            space = worker.local_directory
            self.prt_path = os.path.join(space, os.path.basename(prt_path))
        except ValueError:  # get_worker に失敗した場合
            self.prt_path = os.path.abspath(prt_path)

        self.export_curves = export_curves
        self.export_surfaces = export_surfaces
        self.export_solids = export_solids
        self.export_flattened_assembly = export_flattened_assembly

        # FemtetInterface の設定 (femprj_path, model_name の更新など)
        # + restore 情報の上書き
        super().__init__(
            prt_path=self.prt_path,
            export_curves=self.export_curves,
            export_surfaces=self.export_surfaces,
            export_solids=self.export_solids,
            export_flattened_assembly=self.export_flattened_assembly,
            **kwargs
        )

    def check_param_value(self, name):
        """Override FemtetInterface.check_param_value().

        Do nothing because the parameter can be registered
        to not only .femprj but also .prt.

        """
        pass

    def _setup_before_parallel(self, client):
        client.upload_file(
            self.kwargs['prt_path'],
            False
        )
        super()._setup_before_parallel(client)

    def update_model(self, parameters: 'pd.DataFrame', with_warning=False) -> None:
        """Update .x_t"""

        self.parameters = parameters.copy()

        # Femtet が参照している x_t パスを取得する
        x_t_path = self.Femtet.Gaudi.LastXTPath

        # 前のが存在するならば消しておく
        if os.path.isfile(x_t_path):
            os.remove(x_t_path)

        # 変数の json 文字列を作る
        tmp_dict = {}
        for i, row in parameters.iterrows():
            tmp_dict[row['name']] = row['value']
        str_json = json.dumps(tmp_dict)

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
                self._JOURNAL_PATH,  # update_model.py
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

        # モデルの再インポート
        self._call_femtet_api(
            self.Femtet.Gaudi.ReExecute,
            False,
            ModelError,  # 生きてるのに失敗した場合
            error_message=Msg.ERR_MODEL_RECONSTRUCTION_FAILED,
            is_Gaudi_method=True,
        )

        # 処理を確定
        self._call_femtet_api(
            self.Femtet.Redraw,
            False,  # 戻り値は常に None なのでこの変数に意味はなく None 以外なら何でもいい
            ModelError,  # 生きてるのに失敗した場合
            error_message=Msg.ERR_MODEL_UPDATE_FAILED,
            is_Gaudi_method=True,
        )

        # femprj モデルの変数も更新
        super().update_model(parameters, with_warning=False)
