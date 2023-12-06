import subprocess

import os
here, me = os.path.split(__file__)
import json

from .core import Femtet

PATH_JOURNAL = os.path.abspath(os.path.join(here, 'update_model_parameter.py'))


class NX_Femtet(Femtet):
    def __init__(self, path_prt):

        # 引数の処理
        self.path_prt = os.path.abspath(path_prt)

        # 引数の保存
        super().__init__(self.path_prt)


    def update(self, df)->None:
        """Update_model_via_NX と run を実行します.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------

        """
        self.update_model_via_NX(df)
        super().run()


    def update_model_via_NX(self, df):
        # run_journal を使って prt から x_t を作る

        # Femtet が参照している x_t パスを取得する
        path_x_t = self.Femtet.Gaudi.LastXTPath

        # 先にそれを消しておく
        if os.path.isfile(path_x_t):
            os.remove(path_x_t)

        # NX Journal を使って x_t を更新する
        exe = r'%UGII_BASE_DIR%\NXBIN\run_journal.exe'
        tmpdict = {}
        for i, row in df.iterrows():
            tmpdict[row['name']] = row['value']
        strDict = json.dumps(tmpdict)
        env = os.environ.copy()
        subprocess.run(
            [exe, PATH_JOURNAL, '-args', self.path_prt, strDict, path_x_t],
            env=env,
            shell=True,
            cwd=os.path.dirname(self.path_prt))

        # この時点で x_t ファイルがなければモデルエラー
        if not os.path.isfile(path_x_t):
            from .core import ModelError
            raise ModelError

        # プロジェクトの更新
        self.Femtet.Gaudi.ReExecute()
        self.Femtet.Redraw()

