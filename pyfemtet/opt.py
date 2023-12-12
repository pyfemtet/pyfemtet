from abc import ABC, abstractmethod

import os
import datetime
from time import time, sleep
from threading import Thread

import numpy as np
import pandas as pd
from optuna._hypervolume import WFG
import ray

import win32com.client
from win32com.client import Dispatch, constants, CDispatch
from pywintypes import com_error
from femtetutils import util, constant
from .tools.DispatchUtils import Dispatch_Femtet, Dispatch_Femtet_with_new_process, Dispatch_Femtet_with_specific_pid, _get_pid

from ._core import *
from .FEMIF import *
from .monitor import Monitor


def symlog(x):
    """
    定義域を負領域に拡張したlog関数です。
    多目的最適化における目的関数同士のスケール差により
    意図しない傾向が生ずることのの軽減策として
    内部でsymlog処理を行います。
    """
    if isinstance(x, np.ndarray):
        ret = np.zeros(x.shape)
        idx = np.where(x >= 0)
        ret[idx] = np.log10(x[idx] + 1)
        idx = np.where(x < 0)
        ret[idx] = -np.log10(1 - x[idx])
    else:
        if x >= 0:
            ret = np.log10(x + 1)
        else:
            ret = -np.log10(1 - x)

    return ret


def _check_direction(direction):
    message = '評価関数の direction は "minimize", "maximize", 又は数値でなければなりません.'
    message += f'与えられた値は {direction} です.'
    if isinstance(direction, float) or isinstance(direction, int):
        pass
    elif isinstance(direction, str):
        if (direction != 'minimize') and (direction != 'maximize'):
            raise Exception(message)
    else:
        raise Exception(message)


def _check_lb_ub(lb, ub, name=None):
    message = f'下限{lb} > 上限{ub} です.'
    if name is not None:
        message = f'{name}に対して' + message
    if (lb is not None) and (ub is not None):
        if lb > ub:
            raise Exception(message)


class Objective:

    default_name = 'obj'

    def __init__(self, fun, name, direction, args, kwargs):
        _check_direction(direction)
        self.fun = fun
        self.args = args
        self.kwargs = kwargs
        self.direction = direction
        self.name = name

    def _convert(self, value: float):
        # 評価関数（direction 任意）を目的関数（minimize, symlog）に変換する
        ret = value
        if isinstance(self.direction, float) or isinstance(self.direction, int):
            ret = abs(value - self.direction)
        elif self.direction == 'minimize':
            ret = value
        elif self.direction == 'maximize':
            ret = -value

        ret = symlog(ret)

        return ret

    def calc(self):
        return self._convert(self.fun(*self.args, **self.kwargs))


class Constraint:

    default_name = 'cns'

    def __init__(self, fun, name, lb, ub, strict, args, kwargs):
        _check_lb_ub(lb, ub)
        self.fun = fun
        self.args = args
        self.kwargs = kwargs

        self.name = name
        self.lb = lb
        self.ub = ub
        self.strict = strict

    def calc(self):
        return self.fun(*self.args, **self.kwargs)


class History:

    def __init__(self, history_path, interprocess_variables):
        self.iv = interprocess_variables
        self.data = pd.DataFrame()
        self.path = history_path
        # self.param_names = []
        # self.obj_names = []
        # self._data_columns = []

    def init(self, param_names, obj_names):
        self._data_columns = ...

    def record(self, x, obj_values):
        row = []
        row.extend(x)
        row.extend(obj_values)
        self.iv.append_history(row)
        data = self.iv.get_history()
        self.data = pd.DataFrame(
            data,
            columns=self._data_columns,
        )



class OptimizerBase(ABC):

    def __init__(self, history_path=None, fem: FEMIF = None):

        ray.init()  # (ignore_reinit_error=True)

        # 引数の処理
        if history_path is None:
            history_path = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M.csv')
        self.history_path = os.path.abspath(history_path)
        if fem is None:
            self.fem = Femtet()

        # メンバーの宣言
        self.iv = InterprocessVariables()
        self.parameters = pd.DataFrame()
        self.objectives: [Objective] = []
        self.constraints: [Constraint] = []
        self.history = History(history_path, self.iv)
        self.monitor: Monitor = None
        self.seed: int or None = None
        self.message = ''
        self._obj_values: [float] = []
        self._cns_values: [float] = []
        self._fem_class = type(self.fem)
        self._fem_args = self.fem.args
        self._fem_kwargs = self.fem.kwargs

        # 初期化
        self.parameters = pd.DataFrame(
            columns=['name', 'value', 'lb', 'ub', 'memo'],
            dtype=object,
        )

    # multiprocess 時に pickle できないオブジェクト参照の削除
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['fem']
        del state['monitor']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def set_fem(self, **extra_kwargs):
        if extra_kwargs is not None:
           self._fem_kwargs.update(extra_kwargs)

        self.fem = self._fem_class(
            *self._fem_args,
            **self._fem_kwargs,
        )


    def set_random_seed(self, seed: int):
        self.seed = seed

    def get_random_seed(self):
        return self.seed

    def add_parameter(
            self,
            name: str,
            initial_value: float or None = None,
            lower_bound: float or None = None,
            upper_bound: float or None = None,
            memo: str = ''
    ):


        if initial_value is None:
            initial_value = femtetValue

        d = {
            'name': name,
            'value': initial_value,
            'lbound': lower_bound,
            'ubound': upper_bound,
            'memo': memo,
        }
        df = pd.DataFrame(d, index=[0], dtype=object)

        if len(self._parameter) == 0:
            newdf = df
        else:
            newdf = pd.concat([self._parameter, df], ignore_index=True)

        if self._isDfValid(newdf):
            self._parameter = newdf
        else:
            raise Exception('パラメータの設定が不正です。')

