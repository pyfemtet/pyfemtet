import os
from threading import Thread
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


here, me = os.path.split(__file__)


class OptimizerBase(ABC):

    def __init__(self):
        self.history = History()
        self.parameters = dict()
        self.objectives = dict()

    def add_parameter(self, name, init, lb, ub):
        self.parameters[name] = (init, lb, ub)

    def add_objective(self, name, fun, *args, **kwargs):
        self.objectives[name] = (fun, args, kwargs)

    def f(self, x):
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
        self.history.init(
            self.parameters.keys(),
            self.objectives.keys()
        )
        t = Thread(target=self._main, args=(n_trials, method))
        t.start()
        t.join()


class History:

    def __init__(self):
        self._data = []
        self._data_columns = []
        self.data = pd.DataFrame()
        self.path = os.path.abspath(f'{__file__.replace(".py", ".csv")}')

    def init(self, param_names, obj_names):
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


if __name__ == '__main__':

    opt = OptimizerBase()

    opt.add_parameter()
    opt.add_parameter()
    opt.add_parameter()
    opt.add_objective()
    opt.add_objective()
