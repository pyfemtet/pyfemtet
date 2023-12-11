import os
from time import time, sleep
from threading import Thread
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from .monitor import Monitor


here, me = os.path.split(__file__)


class UserInterruption(Exception):
    pass


class OptimizerBase(ABC):

    def __init__(self):
        self.history = History()
        self.parameters = dict()
        self.objectives = dict()
        self.state = 'ready'
        self.monitor = None

    def set_state(self, state):
        print(f'---{state}---')
        self.state = state

    def get_state(self):
        return self.state

    def add_parameter(self, name, init, lb, ub):
        self.parameters[name] = (init, lb, ub)

    def add_objective(self, name, fun, *args, **kwargs):
        self.objectives[name] = (fun, args, kwargs)

    def f(self, x):
        if self.get_state() == 'interrupted':
            raise UserInterruption
        x = np.array(x)
        objective_values = []
        for name, (fun, args, kwargs) in self.objectives.items():
            objective_values.append(fun(x, *args, **kwargs))
        self.history.record(x, objective_values)
        return objective_values

    @abstractmethod
    def _main(self):
        pass

    def main(self, n_trials=10, method='TPE'):

        self.set_state('preparing')

        self.history.init(
            self.parameters.keys(),
            self.objectives.keys()
        )

        # 計算スレッドとそれを止めるためのイベント
        t = Thread(target=self._main, args=(n_trials, method))
        t.start()
        self.set_state('processing')

        # モニタースレッド
        self.monitor = Monitor(self)
        tm = Thread(target=self.monitor.start_server)
        tm.start()

        start = time()
        while True:
            should_terminate = [
                not t.is_alive(),
            ]
            if any(should_terminate):
                break
            sleep(1)
            print(f'  duration:{time()-start} sec.')

        self.set_state('terminated')

        t.join()
        # tm.join()  # サーバースレッドは明示的に終了しないので待ってはいけない


class History:

    def __init__(self):
        self._data = []
        self._data_columns = []
        self.data = pd.DataFrame()
        self.path = os.path.abspath(f'{__file__.replace(".py", ".csv")}')
        self.param_names = []
        self.obj_names = []

    def init(self, param_names, obj_names):
        self.param_names = list(param_names)
        self.obj_names = list(obj_names)
        self._data_columns = []
        self._data_columns.extend(list(param_names))
        self._data_columns.extend(list(obj_names))

    def record(self, x, obj_values):
        row = []
        row.extend(x)
        row.extend(obj_values)
        self._data.append(row)
        self.data = pd.DataFrame(
            self._data,
            columns=self._data_columns,
        )
