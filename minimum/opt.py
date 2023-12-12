import os
from time import time, sleep
from threading import Thread
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import ray
from .monitor import Monitor


here, me = os.path.split(__file__)


# class UserInterruption(Exception):
#     pass

#
# @ray.remote
# class _ParallelVariableNamespace:
#
#     def __init__(self):
#         self.state = 'undefined'
#         self.history = []
#
#     def set_state(self, state):
#         self.state = state
#
#     def get_state(self) -> 'ObjectRef':
#         return self.state
#
#     def append_history(self, row):
#         self.history.append(row)
#
#     def get_history(self) -> 'ObjectRef':
#         return self.history
#
#
# class ParallelVariableNamespace:
#
#     def __init__(self):
#         self.ns = _ParallelVariableNamespace.remote()
#
#     def set_state(self, state):
#         self.ns.set_state.remote(state)
#
#     def get_state(self):
#         return ray.get(self.ns.get_state.remote())
#
#     def append_history(self, row):
#         self.ns.append_history.remote(row)
#
#     def get_history(self):
#         return ray.get(self.ns.get_history.remote())


class OptimizerBase(ABC):

    def __init__(self):
        ray.init()  # (ignore_reinit_error=True)
        self.parameters = dict()
        self.objectives = dict()
        self.monitor = None
        self.pdata = ParallelVariableNamespace()
        self.history = History(self.pdata)

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     del state['monitor']
    #     return state
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)

    def add_parameter(self, name, init, lb, ub):
        self.parameters[name] = (init, lb, ub)

    def add_objective(self, name, fun, *args, **kwargs):
        self.objectives[name] = (fun, args, kwargs)

    def f(self, x):
        if self.pdata.get_state() == 'interrupted':
            raise UserInterruption
        x = np.array(x)
        objective_values = []
        for name, (fun, args, kwargs) in self.objectives.items():
            objective_values.append(fun(x, *args, **kwargs))
        self.history.record(x, objective_values)
        return objective_values

    @abstractmethod
    def _main(self, *args, **kwargs):
        pass

    def _setup_main(self, *args, **kwargs):
        pass

    def main(self, n_trials=10, n_parallel=3, method='TPE'):

        self.pdata.set_state('preparing')
        self.history.init(
            self.parameters.keys(),
            self.objectives.keys()
        )
        self._setup_main(method)

        # 計算スレッドとそれを止めるためのイベント
        t = Thread(target=self._main, args=(n_trials,))
        t.start()
        self.pdata.set_state('processing')

        # モニタースレッド
        self.monitor = Monitor(self)
        tm = Thread(target=self.monitor.start_server)
        tm.start()

        # 追加の計算プロセス
        @ray.remote
        def _main_remote(*args, **kwargs):
            self._main(*args, **kwargs)
        processes = []
        for subprocess_idx in range(n_parallel-1):
            p = _main_remote.remote(n_trials, subprocess_idx)
            processes.append(p)

        start = time()
        while True:
            should_terminate = [not t.is_alive()]
            if all(should_terminate):  # all tasks are killed
                break
            sleep(1)

        self.pdata.set_state('terminated')

        t.join()
        ray.wait(processes)
        # tm.join()  # サーバースレッドは明示的に終了しないので待ってはいけない


# class History:
#
#     def __init__(self, pdata):
#         self.pdata = pdata
#         self.data = pd.DataFrame()
#         self.path = os.path.abspath(f'{__file__.replace(".py", ".csv")}')
#         self.param_names = []
#         self.obj_names = []
#         self._data_columns = []
#
#     def init(self, param_names, obj_names):
#         self.param_names = list(param_names)
#         self.obj_names = list(obj_names)
#         self._data_columns = []
#         self._data_columns.extend(list(param_names))
#         self._data_columns.extend(list(obj_names))
#
#     def record(self, x, obj_values):
#         row = []
#         row.extend(x)
#         row.extend(obj_values)
#         self.pdata.append_history(row)
#         data = self.pdata.get_history()
#         self.data = pd.DataFrame(
#             data,
#             columns=self._data_columns,
#         )
