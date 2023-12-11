import os
from time import time, sleep
from threading import Thread
from multiprocessing import Process, Manager
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

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['monitor']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

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

    def _setup_main(self, *args, **kwargs):
        pass

    def main(self, n_trials=10, n_parallel=3, method='TPE'):

        self.set_state('preparing')

        self.history.init(
            self.parameters.keys(),
            self.objectives.keys()
        )

        self._setup_main(method)

        # 計算スレッドとそれを止めるためのイベント
        t = Thread(target=self._main, args=(n_trials,))
        t.start()
        self.set_state('processing')

        # モニタースレッド
        self.monitor = Monitor(self)
        tm = Thread(target=self.monitor.start_server)
        tm.start()

        # 追加の計算プロセス
        processes = []
        for subprocess_idx in range(n_parallel-1):
            p = Process(target=self._main, args=(n_trials, subprocess_idx))
            p.start()
            processes.append(p)

        start = time()
        while True:
            should_terminate = [not t.is_alive()]
            should_terminate.extend([not p.is_alive() for p in processes])
            if all(should_terminate):  # all(not alive)
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
