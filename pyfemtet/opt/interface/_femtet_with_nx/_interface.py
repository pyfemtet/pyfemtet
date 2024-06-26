import os
import json
import subprocess

import pandas as pd
from dask.distributed import get_worker

from pyfemtet.core import ModelError
from pyfemtet.opt.interface import FemtetInterface, logger


here, me = os.path.split(__file__)


class FemtetWithNXInterface(FemtetInterface):
    """Femtet with NX interface class.

    Args:
        prt_path: The path to the prt file.
        export_curves(bool or None): Parasolid export setting of NX. If None, PyFemtet does not change the setting of NX. Defaults to None.
        export_surfaces(bool or None): Parasolid export setting of NX. If None, PyFemtet does not change the setting of NX. Defaults to None.
        export_solids(bool or None): Parasolid export setting of NX. If None, PyFemtet does not change the setting of NX. Defaults to None.
        export_flattened_assembly(bool or None): Parasolid export setting of NX. If None, PyFemtet does not change the setting of NX. Defaults to None.

    For details of The other arguments, see ``FemtetInterface``.

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
            raise FileNotFoundError(
                r'"%UGII_BASE_DIR%\NXBIN\run_journal.exe" が見つかりませんでした。環境変数 UGII_BASE_DIR 又は NX のインストール状態を確認してください。')

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

    def update_model(self, parameters: 'pd.DataFrame') -> None:
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
            error_message=f'モデル再構築に失敗しました.',
            is_Gaudi_method=True,
        )

        # 処理を確定
        self._call_femtet_api(
            self.Femtet.Redraw,
            False,  # 戻り値は常に None なのでこの変数に意味はなく None 以外なら何でもいい
            ModelError,  # 生きてるのに失敗した場合
            error_message=f'モデル再構築に失敗しました.',
            is_Gaudi_method=True,
        )

        # femprj モデルの変数も更新
        super().update_model(parameters)

