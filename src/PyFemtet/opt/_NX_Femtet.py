import subprocess

import os
here, me = os.path.split(__file__)
import json

from .core import Femtet

PATH_JOURNAL = os.path.abspath(os.path.join(here, 'update_model_parameter.py'))


class NX_Femtet(Femtet):
    def __init__(self, path_prt, strategy='catch'):
        self.path_prt = os.path.abspath(path_prt)
        super().setFemtet(strategy) # Femtet がセットされる

    def run(self, df):
        self.update_model_via_NX(df)
        super().run(df)
    
    def update_model_via_NX(self, df):
        # run_journal を使って prt から x_t を作る / prt と同じ名前の x_t ができる
        # 先にそれを消しておく
        path_x_t = os.path.splitext(self.path_prt)[0] + '.x_t'
        if os.path.isfile(path_x_t):
            os.remove(path_x_t)
        exe = r'%UGII_BASE_DIR%\NXBIN\run_journal.exe'
        tmpdict = {}
        for i, row in df.iterrows():
            tmpdict[row['name']] = row['value']
        strDict = json.dumps(tmpdict)
        env = os.environ.copy()
        subprocess.run(
            [exe, PATH_JOURNAL, '-args', self.path_prt, strDict],
            env=env,
            shell=True,
            cwd=os.path.dirname(self.path_prt))
        # この時点で x_t ファイルがなければモデルエラー
        if not os.path.isfile(path_x_t):
            from .core import ModelError
            raise ModelError
        # モデルとプロジェクトの更新
        # self.Femtet.Gaudi.NewModel(False)
        self.Femtet.Gaudi.Activate()
        self.Femtet.Gaudi.NewModel(False)
        self.Femtet.Gaudi.ImportSolidWorks(path_x_t)
        self.Femtet.Gaudi.ReExecute()
        '''条件等も反映されるという話だったが、残念ながら反映されなかった。'''

