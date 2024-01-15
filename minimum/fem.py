from time import sleep
import numpy as np
from win32com.client import Dispatch
from femtetutils import util
from dask.distributed import get_worker
import os


class FEM:

    def update(self, x):
        pass


class Femtet(FEM):

    def __init__(self, path):
        # femtet 起動
        util.execute_femtet()
        sleep(20)

        # 接続
        sleep(np.random.rand())  # 簡易排他処理
        self.Femtet = Dispatch('FemtetMacro.Femtet')

        # ファイルを開く
        worker = get_worker()
        space = worker.local_directory
        self.Femtet.LoadProject(os.path.join(space, os.path.basename(path)), True)

    def update(self, parameters):
        self.Femtet.Gaudi.Activate()
        for i, row in parameters.iterrows():
            self.Femtet.UpdateVariable(row['name'], row['value'])
        self.Femtet.Gaudi.ReExecute()
        self.Femtet.Gaudi.Redraw()
        self.Femtet.Gaudi.Mesh()
        self.Femtet.Solve()
        self.Femtet.OpenCurrentResult()

