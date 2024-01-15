from time import sleep
import numpy as np
from win32com.client import Dispatch


class FEM:

    def update(self, x):
        pass


class Femtet(FEM):

    def __init__(self):
        sleep(np.random.rand())  # 簡易排他処理
        self.Femtet = Dispatch('FemtetMacro.Femtet')

    def update(self, parameters):
        self.Femtet.Gaudi.Activate()
        for i, row in parameters.iterrows():
            self.Femtet.UpdateVariable(row['name'], row['value'])
        self.Femtet.Gaudi.ReExecute()
        self.Femtet.Gaudi.Redraw()
        self.Femtet.Gaudi.Mesh()
        self.Femtet.Solve()
        self.Femtet.OpenCurrentResult()

