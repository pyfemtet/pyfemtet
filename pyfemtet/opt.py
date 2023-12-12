from abc import ABC, abstractmethod

import os
import datetime
import inspect
import ast
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


def _is_access_gaudi(fun):

    # 関数fのソースコードを取得
    source = inspect.getsource(fun)

    # ソースコードを抽象構文木（AST）に変換
    tree = ast.parse(source)

    # 関数定義を見つける
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # 関数の第一引数の名前を取得
            first_arg_name = node.args.args[0].arg

            # 関数内の全ての属性アクセスをチェック
            for sub_node in ast.walk(node):
                if isinstance(sub_node, ast.Attribute):
                    # 第一引数に対して 'Gogh' へのアクセスがあるかチェック
                    if (
                            isinstance(sub_node.value, ast.Name)
                            and sub_node.value.id == first_arg_name
                            and sub_node.attr == 'Gogh'
                    ):
                        return True
            # ここまできてもなければアクセスしてない
            return False


def _is_feasible(value, lb, ub):
    if lb is None and ub is not None:
        return value < ub
    elif lb is not None and ub is None:
        return lb < value
    elif lb is not None and ub is not None:
        return lb < value < ub
    else:
        return True

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
        return self.fun(*self.args, **self.kwargs)


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

    def __init__(self, history_path, fem_opt):
        self.path = history_path
        self.fem_opt = fem_opt
        self.iv = self.fem_opt.iv
        self.data = pd.DataFrame()
        self.param_names = []
        self.obj_names = []
        self.cns_names = []
        self._data_columns = []

    def init(self, param_names, obj_names, cns_names):
        self.param_names = param_names
        self.obj_names = obj_names
        self.cns_names = cns_names

        columns = list()
        columns.append('trial')  # index
        columns.extend(self.param_names)  # parameters
        for obj_name in self.obj_names:  # objectives, direction
            columns.extend([obj_name, f'{obj_name}_direction'])
        columns.append('non_domi')
        for cns_name in cns_names:  # cns, lb, ub
            columns.extend([cns_name, f'{cns_name}_lb', f'{cns_name}_ub'])
        columns.append('feasible')
        columns.append('hypervolume')
        columns.append('message')
        columns.append('time')
        self._data_columns = columns

    def record(self, parameters, objectives, constraints, obj_values, cns_values, message):

        # create row
        row = list()
        row.append(len(self.data))
        row.extend(parameters['value'].values)
        for obj, obj_value in zip(objectives, obj_values):  # objectives, direction
            row.extend([obj_value, obj.direction])
        row.append(False)  # dummy non_domi
        feasible_list = []
        for cns, cns_value in zip(constraints, cns_values):  # cns, lb, ub
            row.extend([cns_value, cns.lb, cns.ub])
            feasible_list.append(_is_feasible(cns_value, cns.lb, cns.ub))
        row.append(all(feasible_list))
        row.append(0)  # dummy hypervolume
        row.append(message)  # message
        row.append(datetime.datetime.now())  # time

        # append
        self.iv.append_history(row)
        data = self.iv.get_history()
        self.data = pd.DataFrame(
            data,
            columns=self._data_columns,
        )

        # calc
        self._calc_non_domi()
        self._calc_hypervolume()

    def _calc_non_domi(self):

        # 目的関数の履歴を取り出してくる
        solution_set = self.data[self.obj_names].copy()

        # 最小化問題の座標空間に変換する
        for name, objective in self.fem_opt.objectives.items():
            solution_set[name] = solution_set[name].map(objective._convert)

            # 非劣解の計算
        non_domi = []
        for i, row in solution_set.iterrows():
            non_domi.append((row > solution_set).product(axis=1).sum(axis=0) == 0)

        # 非劣解の登録
        self.data['non_domi'] = non_domi

        del solution_set

    def _calc_hypervolume(self):
        """
        hypervolume 履歴を更新する
        ※ reference point が変わるたびに hypervolume を計算しなおす必要がある
        [1]Hisao Ishibuchi et al. "Reference Point Specification in Hypercolume Calculation for Fair Comparison and Efficient Search"
        """
        #### 前準備
        # パレート集合の抽出
        idx = self.data['non_domi']
        pdf = self.data[idx]
        pareto_set = pdf[self.obj_names].values
        n = len(pareto_set)  # 集合の要素数
        m = len(pareto_set.T)  # 目的変数数
        # 長さが 2 以上でないと計算できない
        if n <= 1:
            return np.nan
        # 最小化問題に convert
        for i, (name, objective) in enumerate(self.fem_opt.objectives.items()):
            for j in range(n):
                pareto_set[j, i] = objective._convert(pareto_set[j, i])
                #### reference point の計算[1]
        # 逆正規化のための範囲計算
        maximum = pareto_set.max(axis=0)
        minimum = pareto_set.min(axis=0)
        # (H+m-1)C(m-1) <= n <= (m-1)C(H+m) になるような H を探す
        H = 0
        while True:
            left = math.comb(H + m - 1, m - 1)
            right = math.comb(H + m, m - 1)
            if left <= n <= right:
                break
            else:
                H += 1
        # H==0 なら r は最大の値
        if H == 0:
            r = 2
        else:
            # r を計算
            r = 1 + 1. / H
        r = 1.01
        # r を逆正規化
        reference_point = r * (maximum - minimum) + minimum

        #### hv 履歴の計算
        wfg = WFG()
        hvs = []
        for i in range(n):
            hv = wfg.compute(pareto_set[:i], reference_point)
            if np.isnan(hv):
                hv = 0
            hvs.append(hv)

        # 計算結果を履歴の一部に割り当て
        df = self.data
        df.loc[idx, 'hypervolume'] = np.array(hvs)

        # dominated の行に対して、上に見ていって
        # 最初に見つけた non-domi 行の hypervolume の値を割り当てます
        for i in range(len(df)):
            if df.loc[i, 'non_domi'] == False:
                try:
                    df.loc[i, 'hypervolume'] = df.loc[:i][df.loc[:i]['non_domi']].iloc[-1]['hypervolume']
                except IndexError:
                    # pass # nan のままにする
                    df.loc[i, 'hypervolume'] = 0


class OptimizerBase(ABC):

    def __init__(fem: FEMIF = None, self, history_path=None):

        print('---initialize---')

        ray.init()  # (ignore_reinit_error=True)

        # 引数の処理
        if history_path is None:
            history_path = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M.csv')
        self.history_path = os.path.abspath(history_path)
        if fem is None:
            self.fem = Femtet()

        # メンバーの宣言
        self.ipv = InterprocessVariables()
        self.parameters = pd.DataFrame()
        self.objectives = dict()
        self.constraints = dict()
        self.history = History(history_path, self.ipv)
        self.monitor: Monitor = None
        self.seed: int or None = None
        self.message = ''
        self.obj_values: [float] = []
        self.cns_values: [float] = []
        self._fem_class = type(self.fem)
        self._fem_args = self.fem.args
        self._fem_kwargs = self.fem.kwargs

        # 初期化
        self.parameters = pd.DataFrame(
            columns=['name', 'value', 'lb', 'ub', 'memo'],
            dtype=object,
        )

        self.ipv.set_state('ready')

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

        _check_lb_ub(lower_bound, upper_bound, name)
        value = self.fem.check_param_value(name)
        if initial_value is None:
            initial_value = value

        d = {
            'name': name,
            'value': initial_value,
            'lb': lower_bound,
            'ub': upper_bound,
            'memo': memo,
        }
        pdf = pd.DataFrame(d, index=[0], dtype=object)

        if len(self.parameters) == 0:
            self.parameters = pdf
        else:
            self.parameters = pd.concat([self.parameters, pdf], ignore_index=True)

    def add_objective(
            self,
            fun,
            name: str or None = None,
            direction: str or float = 'minimize',
            args: tuple or None = None,
            kwargs: dict or None = None
    ):
        # 引数の処理
        if args is None:
            args = tuple()
        elif not isinstance(args, tuple):
            args = (args,)
        if kwargs is None:
            kwargs = dict()
        if name is None:
            prefix = Objective.default_name
            i = 0
            while True:
                candidate = f'{prefix}_{str(int(i))}'
                is_existing = candidate in list(self.objectives.keys())
                if not is_existing:
                    break
            name = candidate

        # Femtet 特有の処理
        if isinstance(self.fem, Femtet):
            args = (self.fem.Femtet, *args)

        self.objectives[name] = Objective(fun, name, direction, args, kwargs)


    def add_constraint(
            self,
            fun,
            name: str or None = None,
            lower_bound: float or None = None,
            upper_bound: float or None = None,
            strict: bool = True,
            args: tuple or None = None,
            kwargs: dict or None = None,
    ):
        # 引数の処理
        if args is None:
            args = tuple()
        elif not isinstance(args, tuple):
            args = (args,)
        if kwargs is None:
            kwargs = dict()
        if name is None:
            prefix = Constraint.default_name
            i = 0
            while True:
                candidate = f'{prefix}_{str(int(i))}'
                is_existing = candidate in list(self.objectives.keys())
                if not is_existing:
                    break
            name = candidate

        # strict constraint の場合、solve 前に評価したいので Gaudi へのアクセスを禁ずる
        if strict:
            if _is_access_gaudi(fun):
                message = f'関数 {fun.__name__} に Gaudi （Femtet 解析結果）へのアクセスがあります.'
                message += 'デフォルトでは constraint は解析前に評価され, 条件を満たさない場合解析を行いません.'
                message += '拘束に解析結果を含めたい場合は, strict=False を設定してください.'
                raise Exception(message)

        # Femtet 特有の処理
        if isinstance(self.fem, Femtet):
            args = (self.fem.Femtet, *args)

        self.constraints[name] = Constraint(fun, name, lower_bound, upper_bound, strict, args, kwargs)



    def get_parameter(self, format='dict'):
        if format == 'df':
            return self.parameters
        elif format == 'values' or format == 'value':
            return self.parameters.value.values
        elif format == 'dict':
            ret = {}
            for i, row in self.parameters.iterrows():
                ret[row['name']] = row.value
            return ret
        else:
            raise Exception('get_parameter() got invalid format: {format}')

    def is_calculated(self, x):
        # 提案された x が最後に計算したものと一致していれば True
        # ただし 1 回目の計算なら False
        #    ひとつでも違う  1回目の計算  期待
        #    True            True         False
        #    False           True         False
        #    True            False        False
        #    False           False        True

        # ひとつでも違う
        param_names = self.history.param_names
        last_x = self.history.data[param_names].iloc[-1].values
        condition1 = False
        for _x, _last_x in zip(x, last_x):
            condition1 = condition1 or (float(_x) != float(_last_x))

        # 1 回目の計算
        condition2 = len(self.history.data) == 0

        return (not condition1) and condition2

    def f(self, x, message):

        x = np.array(x)

        # 中断指令の処理
        if self.ipv.get_state() == 'interrupted':
            self.fem.quit()
            raise UserInterruption

        # アルゴリズムの関係で、すでに計算しているのにもう一度計算しようとする場合は
        # FEM 解析を飛ばす
        if not self.is_calculated(x):

            # parameter の update
            self.parameters['value'] = x

            # fem のソルブ
            self.fem.update(self.parameters)

            # 計算
            self.obj_values = [obj.calc() for _, obj in self.objectives.items()]
            self.cns_values = [cns.calc() for _, cns in self.constraints.items()]

            # 記録
            self.history.record(self.parameters, self.objectives, self.constraints, self.obj_values, self.cns_values, message)

        # minimize
        return [obj._convert(v) for (_, obj), v in zip(self.objectives.items(), self.obj_values)]

    @abstractmethod
    def _main(self, *args, **kwargs):
        pass

    def _setup_main(self, *args, **kwargs):
        pass

    def main(self, n_trials, n_parallel, method):

        # setup
        self.ipv.set_state('preparing')
        self.history.init(
            list(self.parameters.keys()),
            list(self.objectives.keys()),
            list(self.constraints.keys()),
        )
        self._setup_main(method)

        # 計算スレッドとそれを止めるためのイベント
        t = Thread(target=self._main, args=(n_trials,))
        t.start()
        self.ipv.set_state('processing')

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
        end = time()
        print(f'execution time is {end - start} sec.')

        self.ipv.set_state('terminated')

        t.join()
        ray.wait(processes)

        self.ipv.set_state('ready')
