from time import sleep
import numpy as np
from win32com.client import Dispatch
from femtetutils import util
from dask.distributed import get_worker
import os
from pythoncom import CoInitialize, CoUninitialize



class FEM:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def update(self, x):
        pass

    def setup_before_parallel(self, client):
        pass


class Femtet(FEM):

    def __init__(self, path):

        # win32com の初期化
        CoInitialize()

        # femtet 起動
        util.execute_femtet()
        sleep(15)

        # 接続
        sleep(np.random.rand())  # 簡易排他処理
        self.Femtet = Dispatch('FemtetMacro.Femtet')

        # ファイルを開く
        try:
            worker = get_worker()
            space = worker.local_directory
            self.Femtet.LoadProject(os.path.join(space, os.path.basename(path)), True)
        except ValueError:
            self.Femtet.LoadProject(path, True)

        # restore 用データの保存
        super().__init__(
            path=path,
        )

    def update(self, parameters):
        self.Femtet.Gaudi.Activate()
        for i, row in parameters.iterrows():
            self.Femtet.UpdateVariable(row['name'], row['value'])
        self.Femtet.Gaudi.ReExecute()
        self.Femtet.Gaudi.Redraw()
        self.Femtet.Gaudi.Mesh()
        self.Femtet.Solve()
        self.Femtet.OpenCurrentResult()

    def setup_before_parallel(self, client):
        client.upload_file(
            self.kwargs['path'],
            False
        )


    # def __del__(self):
    #     CoUninitialize()  # Win32 exception occurred releasing IUnknown at 0x0000022427692748
