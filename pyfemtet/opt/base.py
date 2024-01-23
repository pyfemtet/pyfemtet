from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Iterable, Any

import os
import datetime
import inspect
import ast
from time import time, sleep
from threading import Thread
from subprocess import Popen
import warnings
import logging

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from optuna.exceptions import ExperimentalWarning
from optuna._hypervolume import WFG
from dask.distributed import LocalCluster, Client, Lock, get_worker

from win32com.client import constants, Constants

from ..core import ModelError, MeshError, SolveError
from .interface import FEMInterface, FemtetInterface
from .monitor import Monitor


logger = logging.getLogger('optimization-main')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(process)d] %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)  # TODO: optuna, dask の logger の書式を統一する


warnings.filterwarnings('ignore', category=ExperimentalWarning)


def generate_lhs(bounds: List[List[float]], seed: int or None = None) -> np.ndarray:
    """Latin Hypercube Sampling from given design parameter bounds.

    If the number of parameters is d,
    sampler returns (N, d) shape ndarray.
    N equals p**2, p is the minimum prime number over d.
    For example, when d=3, then p=5 and N=25.

    Args:
        bounds (list[list[float]]): List of [lower_bound, upper_bound] of parameters.
        seed (int or None, optional): Random seed. Defaults to None.

    Returns:
        np.ndarray: (N, d) shape ndarray.
    """

    d = len(bounds)

    sampler = LatinHypercube(
        d,
        scramble=False,
        strength=2,
        # optimization='lloyd',
        optimization='random-cd',
        seed=seed,
    )

    LIMIT = 100

    def is_prime(p):
        for i in range(2, p):
            if p % i == 0:
                return False
        return True

    def get_prime(minimum):
        for p in range(minimum, LIMIT):
            if is_prime(p):
                return p

    n = get_prime(d + 1) ** 2
    data = sampler.random(n)  # [0,1)

    for i, (data_range, datum) in enumerate(zip(bounds, data.T)):
        minimum, maximum = data_range
        band = maximum - minimum
        converted_datum = datum * band + minimum
        data[:, i] = converted_datum

    return data  # data.shape = (N, d)


def symlog(x: float or np.ndarray):
    """Log function whose domain is extended to the negative region.

    Symlog processing is performed internally as a measure to reduce
    unintended trends caused by scale differences
    between objective functions in multi-objective optimization.

    Args:
        x (float or np.ndarray)

    Returns:
        float
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
            raise ValueError(message)
    else:
        raise ValueError(message)


def _check_lb_ub(lb, ub, name=None):
    message = f'下限{lb} > 上限{ub} です.'
    if name is not None:
        message = f'{name}に対して' + message
    if (lb is not None) and (ub is not None):
        if lb > ub:
            raise ValueError(message)


def _is_access_gogh(fun):

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


class _Scapegoat:
    """Helper class for parallelize Femtet."""
    # constants を含む関数を並列化するために
    # メイン処理で一時的に constants への参照を
    # このオブジェクトにして、後で restore する
    def __init__(self, ignore=False):
        self._ignore_when_restore_constants = ignore


class Function:
    """Base class for Objective and Constraint."""

    def __init__(self, fun, name, args, kwargs):

        # serializable でない COM 定数を parallelize するため
        # COM 定数を一度 _Scapegoat 型のオブジェクトにする
        for varname in fun.__globals__:
            if isinstance(fun.__globals__[varname], Constants):
                fun.__globals__[varname] = _Scapegoat()

        self.fun = fun
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def calc(self, fem: FEMInterface):
        """Execute user-defined fun.

        If fem is a FemtetInterface,
        the 1st argument of fun is set to fem automatically.

        Args:
            fem (FEMInterface)

        Returns:
            float
        """        
        args = self.args
        # Femtet 特有の処理
        if isinstance(fem, FemtetInterface):
            args = (fem.Femtet, *args)
        return self.fun(*args, **self.kwargs)

    def _restore_constants(self):
        """Helper function for parallelize Femtet."""
        fun = self.fun
        for varname in fun.__globals__:
            if isinstance(fun.__globals__[varname], _Scapegoat):
                if not fun.__globals__[varname]._ignore_when_restore_constants:
                    fun.__globals__[varname] = constants


class Objective(Function):
    """Class for registering user-defined objective function."""

    default_name = 'obj'

    def __init__(self, fun, name, direction, args, kwargs):
        """Initializes an Objective instance.

        Args:
            fun: The user-defined objective function.
            name (str): The name of the objective function.
            direction (str or float or int): The direction of optimization.
            args: Additional arguments for the objective function.
            kwargs: Additional keyword arguments for the objective function.

        Raises:
            ValueError: If the direction is not valid.

        """
        _check_direction(direction)
        self.direction = direction
        super().__init__(fun, name, args, kwargs)

    def convert(self, value: float):
        """Converts an evaluation value to the value of user-defined objective function based on the specified direction.

        When direction is `'minimize'`, ``value`` is calculated.
        When direction is `'maximize'`, ``-value`` is calculated.
        When direction is float, ``abs(value - direction)`` is calculated.
        Finally, the calculated value is passed to the symlog function.

        ``value`` is the return value of the user-defined function.

        Args:
            value (float): The evaluation value to be converted.

        Returns:
            float: The converted objective value.

        """

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


class Constraint(Function):
    """Class for registering user-defined constraint function."""

    default_name = 'cns'

    def __init__(self, fun, name, lb, ub, strict, args, kwargs):
        """Initializes a Constraint instance.

        Args:
            fun: The user-defined constraint function.
            name (str): The name of the constraint function.
            lb: The lower bound of the constraint.
            ub: The upper bound of the constraint.
            strict (bool): Whether to enforce strict inequality for the bounds.
            args: Additional arguments for the constraint function.
            kwargs: Additional keyword arguments for the constraint function.

        Raises:
            ValueError: If the lower bound is greater than or equal to the upper bound.

        """

        _check_lb_ub(lb, ub)
        self.lb = lb
        self.ub = ub
        self.strict = strict
        super().__init__(fun, name, args, kwargs)


class HistoryDfCore:
    """Class for managing a DataFrame object in a distributed manner."""

    def __init__(self):
        self.df = pd.DataFrame()

    def set_df(self, df):
        self.df = df

    def get_df(self):
        return self.df


class History:
    """Class for managing the history of optimization results.

    Attributes:
        path (str): The path to the history file.
        _actor_data (HistoryDfCore): The distributed DataFrame object for storing actor data.
        data (pd.DataFrame): The DataFrame object for storing the entire history.
        param_names (list): The names of the parameters in the study.
        obj_names (list): The names of the objectives in the study.
        cns_names (list): The names of the constraints in the study.

    """
    def __init__(self, history_path, client):
        """Initializes a History instance.

        Args:
            history_path (str): The path to the history file.

        """

        # 引数の処理
        self.path = history_path  # .csv
        self._future = client.submit(HistoryDfCore, actor=True)
        self._actor_data = self._future.result()
        self.param_names = []
        self.obj_names = []
        self.cns_names = []

        # path が存在すれば dataframe を読み込む
        if os.path.isfile(self.path):
            self.actor_data = pd.read_csv(self.path, encoding='shift-jis')

    @property
    def actor_data(self):
        return self._actor_data.get_df().result()

    @actor_data.setter
    def actor_data(self, df):
        self._actor_data.set_df(df).result()

    def init(self, param_names, obj_names, cns_names):
        """Initializes the parameter, objective, and constraint names in the History instance.

        Args:
            param_names (list): The names of parameters in optimization.
            obj_names (list): The names of objectives in optimization.
            cns_names (list): The names of constraints in optimization.

        """
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

        # restart ならば前のデータとの整合を確認
        if len(self.actor_data.columns) > 0:
            # 読み込んだ columns が生成した columns と違っていればエラー
            try:
                if list(self.actor_data.columns) != columns:
                    raise Exception(f'読み込んだ history と問題の設定が異なります. \n\n読み込まれた設定:\n{list(self.actor_data.columns)}\n\n現在の設定:\n{columns}')
                else:
                    # 同じであっても目的と拘束の上下限や direction が違えばエラー
                    pass
            except ValueError:
                raise Exception(f'読み込んだ history と問題の設定が異なります. \n\n読み込まれた設定:\n{list(self.actor_data.columns)}\n\n現在の設定:\n{columns}')

        else:
            # actor_data は actor 経由の getter property なので self.data[column] = ... とやっても
            # actor には変更が反映されない. 以下同様
            tmp = self.actor_data
            for column in columns:
                tmp[column] = None
            self.actor_data = tmp

    def record(self, parameters, objectives, constraints, obj_values, cns_values, message):
        """Records the optimization results in the history.

        Args:
            parameters (pd.DataFrame): The parameter values.
            objectives (dict): The objective functions.
            constraints (dict): The constraint functions.
            obj_values (list): The objective values.
            cns_values (list): The constraint values.
            message (str): Additional information or messages related to the optimization results.

        """

        # create row
        row = list()
        row.append(-1)  # dummy trial index
        row.extend(parameters['value'].values)
        for (name, obj), obj_value in zip(objectives.items(), obj_values):  # objectives, direction
            row.extend([obj_value, obj.direction])
        row.append(False)  # dummy non_domi
        feasible_list = []
        for (name, cns), cns_value in zip(constraints.items(), cns_values):  # cns, lb, ub
            row.extend([cns_value, cns.lb, cns.ub])
            feasible_list.append(_is_feasible(cns_value, cns.lb, cns.ub))
        row.append(all(feasible_list))
        row.append(-1.)  # dummy hypervolume
        row.append(message)  # message
        row.append(datetime.datetime.now())  # time

        with Lock('calc-history'):
            # append
            if len(self.actor_data) == 0:
                self.actor_data = pd.DataFrame([row], columns=self.actor_data.columns)
            else:
                tmp = self.actor_data
                tmp.loc[len(tmp)] = row
                self.actor_data = tmp

            # calc
            tmp = self.actor_data
            tmp['trial'] = np.arange(len(tmp)) + 1  # 1 始まり
            self.actor_data = tmp
            self._calc_non_domi(objectives)
            self._calc_hypervolume(objectives)

    def _calc_non_domi(self, objectives):

        # 目的関数の履歴を取り出してくる
        solution_set = self.actor_data[self.obj_names].copy()

        # 最小化問題の座標空間に変換する
        for name, objective in objectives.items():
            solution_set[name] = solution_set[name].map(objective.convert)

        # 非劣解の計算
        non_domi = []
        for i, row in solution_set.iterrows():
            non_domi.append((row > solution_set).product(axis=1).sum(axis=0) == 0)

        # 非劣解の登録
        tmp = self.actor_data
        tmp['non_domi'] = non_domi
        self.actor_data = tmp

        del solution_set

    def _calc_hypervolume(self, objectives):
        # プロセス間の競合を避けるためローカルにコピーする
        df = self.actor_data.copy()

        # パレート集合の抽出
        idx = df['non_domi'].values
        pdf = df[idx]
        pareto_set = pdf[self.obj_names].values
        n = len(pareto_set)  # 集合の要素数
        m = len(pareto_set.T)  # 目的変数数
        # 多目的でないと計算できない
        if m <= 1:
            return None
        # 長さが 2 以上でないと計算できない
        if n <= 1:
            return None
        # 最小化問題に convert
        for i, (name, objective) in enumerate(objectives.items()):
            for j in range(n):
                pareto_set[j, i] = objective.convert(pareto_set[j, i])
                #### reference point の計算[1]
        # 逆正規化のための範囲計算
        maximum = pareto_set.max(axis=0)
        minimum = pareto_set.min(axis=0)

        # # [1]Hisao Ishibuchi et al. "Reference Point Specification in Hypercolume Calculation for Fair Comparison and Efficient Search"
        # # (H+m-1)C(m-1) <= n <= (m-1)C(H+m) になるような H を探す[1]
        # H = 0
        # while True:
        #     left = math.comb(H + m - 1, m - 1)
        #     right = math.comb(H + m, m - 1)
        #     if left <= n <= right:
        #         break
        #     else:
        #         H += 1
        # # H==0 なら r は最大の値
        # if H == 0:
        #     r = 2
        # else:
        #     # r を計算
        #     r = 1 + 1. / H
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
        df.loc[idx, 'hypervolume'] = np.array(hvs)

        # dominated の行に対して、上に見ていって
        # 最初に見つけた non-domi 行の hypervolume の値を割り当てます
        for i in range(len(df)):
            if df.loc[i, 'non_domi'] == False:
                try:
                    df.loc[i, 'hypervolume'] = df.loc[:i][df.loc[:i]['non_domi']].iloc[-1]['hypervolume']
                except IndexError:
                    df.loc[i, 'hypervolume'] = 0

        # ローカルの df を worker に戻す
        self.actor_data = df


class OptimizationStatusActor:
    status_int = -1
    status = 'undefined'

    def set(self, value, text):
        self.status_int = value
        self.status = text


class OptimizationStatus:
    """Optimization status.

    undefined -> setting up -> launching -> running (-> interrupting) -> terminated

    """
    UNDEFINED = -1
    INITIALIZING = 0
    SETTING_UP = 10
    LAUNCHING = 20
    WAIT_1ST = 25
    RUNNING = 30
    INTERRUPTING = 40
    TERMINATED = 50
    TERMINATE_ALL = 60

    def __init__(self, client):
        self._future = client.submit(OptimizationStatusActor, actor=True)
        self._actor = self._future.result()
        self.set(self.INITIALIZING)

    @classmethod
    def const_to_str(self, status_const):
        if status_const == self.UNDEFINED: return 'Undefined'
        if status_const == self.INITIALIZING: return 'Initializing'
        if status_const == self.SETTING_UP: return 'Setting up'
        if status_const == self.LAUNCHING: return 'Launching FEM processes'
        if status_const == self.WAIT_1ST: return 'Running and waiting for 1st FEM result.'
        if status_const == self.RUNNING: return 'Running'
        if status_const == self.INTERRUPTING: return 'Interrupting'
        if status_const == self.TERMINATED: return 'Terminated'
        if status_const == self.TERMINATE_ALL: return 'Terminate_all'

    def set(self, status_const):
        # self._actor.status_int = status_const
        # self._actor.status = self.const_to_str(status_const)
        self._actor.set(status_const, self.const_to_str(status_const)).result()
        print(f'---{self.const_to_str(status_const)}---')

    def get(self):
        return self._actor.status_int

    def get_text(self):
        return self._actor.status


class AbstractOptimizer(ABC):

    def __init__(self):
        self.fem = None
        self.fem_class = None
        self.fem_kwargs = dict()
        self.parameters = pd.DataFrame()
        self.objectives = dict()
        self.constraints = dict()
        self.status = None  # shared
        self.history = None  # shared
        self.message = ''
        self.seed = None
        self.timeout = None
        self.n_trials = None
        self.is_cluster = False

    def f(self, x):

        if self.status.get() == OptimizationStatus.LAUNCHING:
            self.status.set(OptimizationStatus.WAIT_1ST)

        # x の更新
        self.parameters['value'] = x

        # FEM の更新
        logger.debug('fem.update() start')
        self.fem.update(self.parameters)

        # y, _y, c の更新
        logger.debug('calculate y start')
        y = [obj.calc(self.fem) for obj in self.objectives.values()]

        logger.debug('calculate _y start')
        _y = [obj.convert(value) for obj, value in zip(self.objectives.values(), y)]

        logger.debug('calculate c start')
        c = [cns.calc(self.fem) for cns in self.constraints.values()]

        logger.debug('history.record start')
        self.history.record(
            self.parameters,
            self.objectives,
            self.constraints,
            y,
            c,
            self.message
        )

        if self.status.get() == OptimizationStatus.WAIT_1ST:
            self.status.set(OptimizationStatus.RUNNING)

        logger.debug('history.record end')
        return np.array(y), np.array(_y), np.array(c)

    def set_fem(self):
        self.fem = self.fem_class(**self.fem_kwargs)

        # COM 定数の restore
        for obj in self.objectives.values():
            obj._restore_constants()
        for cns in self.constraints.values():
            cns._restore_constants()

    def get_parameter(self, format='dict'):
        """Returns the parameters in the specified format.

        Args:
            format (str, optional): The desired format of the parameters. Can be 'df' (DataFrame), 'values', or 'dict'. Defaults to 'dict'.

        Returns:
            object: The parameters in the specified format.

        Raises:
            ValueError: If an invalid format is provided.

        """
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
            raise ValueError('get_parameter() got invalid format: {format}')

    def _main(self, subprocess_idx):
        self.set_fem()
        self.main(subprocess_idx)

    @abstractmethod
    def main(self, subprocess_idx: int):
        """Called from dask worker."""
        pass

    @abstractmethod
    def setup_before_parallel(self, *args, **kwargs):
        """Preprocessing for dask parallel workers."""
        pass


class OptunaOptimizer(AbstractOptimizer):

    def __init__(
            self,
            sampler_class: optuna.samplers.BaseSampler or None = None,
            sampler_kwargs: dict or None = None,
            add_init_method: str or Iterable[str] or None = None
    ):
        super().__init__()
        self.study_name = None
        self.storage = None
        self.study = None
        self.sampler_class = optuna.samplers.TPESampler if sampler_class is None else sampler_class
        self.sampler_kwargs = dict() if sampler_kwargs is None else sampler_kwargs
        self.additional_initial_parameter = []
        self.additional_initial_methods = add_init_method if hasattr(add_init_method, '__iter__') else [add_init_method]


    def _objective(self, trial):

        # 中断の確認 (FAIL loop に陥る対策)
        if self.status.get() == OptimizationStatus.INTERRUPTING:
            trial.study.stop()  # 現在実行中の trial を最後にする
            return None  # set TrialState FAIL

        # candidate x
        x = []
        for i, row in self.parameters.iterrows():
            v = trial.suggest_float(row['name'], row['lb'], row['ub'])
            x.append(v)
        x = np.array(x).astype(float)

        # message の設定
        self.message = trial.user_attrs['message'] if 'message' in trial.user_attrs.keys() else ''

        # fem や opt 経由で変数を取得して constraint を計算する時のためにアップデート
        self.parameters['value'] = x
        self.fem.update_parameter(self.parameters)

        # strict 拘束
        strict_constraints = [cns for cns in self.constraints.values() if cns.strict]
        for cns in strict_constraints:
            feasible = True
            cns_value = cns.calc(self.fem)
            if cns.lb is not None:
                feasible = feasible and (cns_value >= cns.lb)
            if cns.ub is not None:
                feasible = feasible and (cns.ub >= cns_value)
            if not feasible:
                print(f'以下の変数で拘束 {cns.name} が満たされませんでした。')
                print(self.get_parameter('dict'))
                raise optuna.TrialPruned()  # set TrialState PRUNED because FAIL causes similar candidate loop.

        # 計算
        try:
            _, _y, c = self.f(x)
        except (ModelError, MeshError, SolveError) as e:
            print(str(type(e)).split("'")[1].split('.')[-1], e)
            print('FEM 解析に失敗しました.')

            # 中断の確認 (解析中に interrupt されている場合対策)
            if self.status.get() == OptimizationStatus.INTERRUPTING:
                trial.study.stop()  # 現在実行中の trial を最後にする
                return None  # set TrialState FAIL

            raise optuna.TrialPruned()  # set TrialState PRUNED because FAIL causes similar candidate loop.

        # 拘束 attr の更新
        _c = []  # 非正なら OK
        for (name, cns), c_value in zip(self.constraints.items(), c):
            lb, ub = cns.lb, cns.ub
            if lb is not None:  # fun >= lb  <=>  lb - fun <= 0
                _c.append(lb - c_value)
            if ub is not None:  # ub >= fun  <=>  fun - ub <= 0
                _c.append(c_value - ub)
        trial.set_user_attr('constraint', _c)

        # 中断の確認 (解析中に interrupt されている場合対策)
        if self.status.get() == OptimizationStatus.INTERRUPTING:
            trial.study.stop()  # 現在実行中の trial を最後にする
            return None  # set TrialState FAIL

        # 結果
        return tuple(_y)

    def _constraint(self, trial):
        return trial.user_attrs['constraint'] if 'constraint' in trial.user_attrs.keys() else (1,)  # infeasible

    def setup_before_parallel(self):
        """Main process から呼ばれる関数"""

        # create storage
        self.study_name = os.path.basename(self.history.path)
        storage_path = self.history.path.replace('.csv', '.db')  # history と同じところに保存
        if self.is_cluster:  # remote cluster なら scheduler の working dir に保存
            storage_path = os.path.basename(self.history.path).replace('.csv', '.db')

        # create study if storage is not exists
        if not os.path.exists(storage_path):

            self.storage = optuna.integration.dask.DaskStorage(
                f'sqlite:///{storage_path}',
            )

            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                load_if_exists=True,
                directions=['minimize'] * len(self.objectives),
            )

            # 初期値の設定
            if len(self.study.trials) == 0:  # リスタートでなければ
                # ユーザーの指定した初期値
                params = self.get_parameter('dict')
                self.study.enqueue_trial(params, user_attrs={"message": "initial"})

                # add_initial_parameter で追加された初期値
                for prm, prm_set_name in self.additional_initial_parameter:
                    self.study.enqueue_trial(
                        prm,
                        user_attrs={"message": prm_set_name}
                    )

                # add_init で指定された方法による初期値
                if 'LHS' in self.additional_initial_methods:
                    names = []
                    bounds = []
                    for i, row in self.parameters.iterrows():
                        names.append(row['name'])
                        lb = row['lb']
                        ub = row['ub']
                        bounds.append([lb, ub])
                    data = generate_lhs(bounds, seed=self.seed)
                    for datum in data:
                        d = {}
                        for name, v in zip(names, datum):
                            d[name] = v
                        self.study.enqueue_trial(
                            d, user_attrs={"message": "additional initial (Latin Hypercube Sampling)"}
                        )

        else:

            self.storage = optuna.integration.dask.DaskStorage(
                f'sqlite:///{storage_path}',
            )

    def add_init_parameter(
            self,
            parameter: dict or Iterable,
            name: str or None = None,
    ):
        """Add additional initial parameter for evaluate.

        The parameter set is ignored if the main() is continued.

        Args:
            parameter (dict or Iterable): Parameter to evaluate before run optimization algorithm.
            name (str or None): Optional. If specified, the name is saved in the history row. Default to None.

        """
        if name is None:
            name = 'additional initial'
        else:
            name = f'additional initial ({name})'
        self.additional_initial_parameter.append([parameter, name])

    def main(self, subprocess_idx):
        """Dask worker から呼ばれる関数"""

        # (re)set random seed
        seed = self.seed
        if seed is not None:
            if subprocess_idx is not None:
                seed += subprocess_idx

        # restore sampler
        sampler = self.sampler_class(
            seed=seed,
            constraints_func=self._constraint,
            **self.sampler_kwargs
        )

        # load study
        study = optuna.load_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=sampler,
        )

        # callback to terminate
        callbacks = []
        n_existing_trials = len(self.history.actor_data)
        if self.n_trials is not None:
            n_trials = n_existing_trials + self.n_trials
            callbacks.append(MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE,)))

        # run
        study.optimize(
            self._objective,
            timeout=self.timeout,
            callbacks=callbacks,
        )

        # finalize
        del self.fem


class FEMOpt:
    """Base class to control FEM interface and optimizer.

    Attributes:
        fem (FEMInterface): The finite element method interface.
        history_path (str): The path to the history (.csv) file .
        ipv (InterprocessVariables): The interprocess variables.
        parameters (pd.DataFrame): The DataFrame object for storing the parameters.
        objectives (dict): The dictionary of objective functions.
        constraints (dict): The dictionary of constraint functions.
        history (History): The history of optimization results.
        monitor (Monitor): The monitor object for visualization and monitoring.
        monitor_thread: Thread object for monitor server.
        seed (int or None): The random seed for reproducibility.
        message(str) : Additional information or messages related to the optimization process
        obj_values ([float]): A list to store objective values during optimization
        cns_values ([float]): A list to store constraint values during optimization

    """

    def __init__(
            self,
            fem: FEMInterface = None,
            opt: AbstractOptimizer = None,
            history_path: str = None,
            scheduler_address: str = None
    ):
        """Initializes an FEMOpt instance.

        Args:
            fem (FEMInterface, optional): The finite element method interface. Defaults to None. If None, automattically set to FemtetInterface.
            opt (AbstractOptimizer):
            history_path (str, optional): The path to the history file. Defaults to None. If None, '%Y_%m_%d_%H_%M_%S.csv' is created in current directory.
            scheduler_address (str or None): If cluster processing, set this parameter like ``"tcp://xxx.xxx.xxx.xxx:xxxx"``.

        """

        logger.info('Initialize FEMOpt')

        # 引数の処理
        if history_path is None:
            history_path = datetime.datetime.now().strftime('%Y%m%d_%H%M%S.csv')
        self.history_path = os.path.abspath(history_path)
        self.scheduler_address = scheduler_address

        if fem is None:
            self.fem = FemtetInterface()
        else:
            self.fem = fem

        if opt is None:
            self.opt = OptunaOptimizer()
        else:
            self.opt = opt

        # メンバーの宣言
        self.client = None
        self.status = None
        self.history = None
        self.monitor_process_future = None
        self.monitor_server_kwargs = dict()

    # multiprocess 時に pickle できないオブジェクト参照の削除
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['fem']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def set_random_seed(self, seed: int):
        """Sets the random seed for reproducibility.

        Args:
            seed (int): The random seed value to be set.

        """
        self.opt.seed = seed

    def add_parameter(
            self,
            name: str,
            initial_value: float or None = None,
            lower_bound: float or None = None,
            upper_bound: float or None = None,
            memo: str = ''
    ):
        """Adds a parameter to the optimization problem.
        
        Args:
            name (str): The name of the parameter.
            initial_value (float or None, optional): The initial value of the parameter. Defaults to None. If None, try to get inittial value from FEMInterface.
            lower_bound (float or None, optional): The lower bound of the parameter. Defaults to None. However, this argument is required for some algorithms.
            upper_bound (float or None, optional): The upper bound of the parameter. Defaults to None. However, this argument is required for some algorithms.
            memo (str, optional): Additional information about the parameter. Defaults to ''.
        Raises:
            ValueError: If initial_value is not specified and the value for the given name is also not specified.

        """

        _check_lb_ub(lower_bound, upper_bound, name)
        value = self.fem.check_param_value(name)
        if initial_value is None:
            if value is not None:
                initial_value = value
            else:
                raise ValueError('initial_value を指定してください.')

        d = {
            'name': name,
            'value': float(initial_value),
            'lb': float(lower_bound),
            'ub': float(upper_bound),
            'memo': memo,
        }
        pdf = pd.DataFrame(d, index=[0], dtype=object)

        if len(self.opt.parameters) == 0:
            self.opt.parameters = pdf
        else:
            self.opt.parameters = pd.concat([self.opt.parameters, pdf], ignore_index=True)

    def add_objective(
            self,
            fun,
            name: str or None = None,
            direction: str or float = 'minimize',
            args: tuple or None = None,
            kwargs: dict or None = None
    ):
        """Adds an objective to the optimization problem.

        Args:
            fun (callable): The objective function.
            name (str or None, optional): The name of the objective. Defaults to None.
            direction (str or float, optional): The optimization direction. Defaults to 'minimize'.
            args (tuple or None, optional): Additional arguments for the objective function. Defaults to None.
            kwargs (dict or None, optional): Additional keyword arguments for the objective function. Defaults to None.
        
        Note:
            If the FEMInterface is FemtetInterface, the 1st argument of fun should be Femtet (IPyDispatch) object.
        
        Tip:
            If name is None, name is a string with the prefix `"obj_"` followed by a sequential number.

        """

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
                is_existing = candidate in list(self.opt.objectives.keys())
                if not is_existing:
                    break
                else:
                    i += 1
            name = candidate

        self.opt.objectives[name] = Objective(fun, name, direction, args, kwargs)


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
        """Adds a constraint to the optimization problem.

        Args：
            fun (callable): The constraint function.
            name (str or None, optional): The name of the constraint. Defaults to None.
            lower_bound (float or Non, optional): The lower bound of the constraint. Defaults to None.
            upper_bound (float or Non, optional): The upper bound of the constraint. Defaults to None.
            strict (bool, optional): Flag indicating if it is a strict constraint. Defaults to True.
            args (tuple or None, optional): Additional arguments for the constraint function. Defaults toNone.

        Note:
            If the FEMInterface is FemtetInterface, the 1st argument of fun should be Femtet (IPyDispatch) object.

        Tip:
            If name is None, name is a string with the prefix `"cns_"` followed by a sequential number.

        """

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
                is_existing = candidate in list(self.opt.constraints.keys())
                if not is_existing:
                    break
                else:
                    i += 1
            name = candidate

        # strict constraint の場合、solve 前に評価したいので Gogh へのアクセスを禁ずる
        if strict:
            if _is_access_gogh(fun):
                message = f'関数 {fun.__name__} に Gogh （Femtet 解析結果）へのアクセスがあります.'
                message += 'デフォルトでは constraint は解析前に評価され, 条件を満たさない場合解析を行いません.'
                message += '拘束に解析結果を含めたい場合は, strict=False を設定してください.'
                raise Exception(message)

        self.opt.constraints[name] = Constraint(fun, name, lower_bound, upper_bound, strict, args, kwargs)

    def get_parameter(self, format='dict'):
        """Returns the parameters in the specified format.

        Args:
            format (str, optional): The desired format of the parameters. Can be 'df' (DataFrame), 'values', or 'dict'. Defaults to 'dict'.

        Returns:
            object: The parameters in the specified format.

        Raises:
            ValueError: If an invalid format is provided.

        """
        return self.opt.get_parameter(format)

    def set_monitor_host(self, host='localhost', port=None):
        """Sets up the monitor server with the specified host and port.

        Args:
            host (str): The hostname or IP address of the monitor server.
            port (int or None, optional): The port number of the monitor server. If None, ``8080`` will be used. Defaults to None.

        Tip:
            If host is ``0.0.0.0``, a server will be set up
            that is visible from the local network.
            Start a browser on another machine
            and type ``<ip_address_or_hostname>:<port>`` in the address bar.

            However, please note that in this case,
            it will be visible to all users on the local network.

        """
        self.monitor_server_kwargs = dict(
            host=host,
            port=port
        )

    def main(
            self,
            n_trials=None,
            n_parallel=1,
            timeout=None,
    ):
        """Runs the main optimization process.

        Args:
            n_trials (int or None): The number of trials. Defaults to None.
            n_parallel (int): The number of parallel processes. Defaults to 1.
            timeout (float or None): The maximum amount of time in seconds that each trial can run. Defaults to None.
            method (str): The optimization method to use. Defaults to 'TPE'.
            **setup_kwargs: Additional keyword arguments for setting up the optimization process.
        
        Tip:
            If setup_monitor_server() is not executed, a local server for monitoring will be started at localhost:8080.

        Note:
            If ``n_trials`` and ``timeout`` are both None, it will calculate repeatedly until interrupting by the user.
        
        Note:
            If ``n_parallel`` >= 2, depending on the end timing, ``n_trials`` may be exceeded by up to ``n_parallel-1`` times.

        Note:
            Currently, supported methods are 'TPE' and 'botorch'.
            For detail, see https://optuna.readthedocs.io/en/stable/reference/samplers/index.html.


        """


        # dask メンバーの設定
        self.opt.is_cluster = self.scheduler_address is not None
        if self.opt.is_cluster:
            # 既存のクラスターに接続
            logger.info('Connecting to existing cluster.')
            self.client = Client(self.scheduler_address)

        else:
            logger.info('Launching single machine cluster. This may take tens of seconds.')
            cluster = LocalCluster(processes=True, threads_per_worker=1)
            self.client = Client(cluster, direct_to_workers=False)
            self.scheduler_address = self.client.scheduler.address

        # monitor 用 worker を起動
        logger.info('Launching monitor server. This may take a few seconds.')
        self.monitor_process_worker_name = datetime.datetime.now().strftime("monitor-server-process-%Y%m%d%H%M%S")
        cmd = f'dask worker {self.client.scheduler.address} --name {self.monitor_process_worker_name} --no-nanny'
        worker_addresses = list(self.client.has_what().keys())
        Popen(cmd, shell=True)  #, stdout=PIPE) --> cause stream error

        # monitor 用 worker が増えるまで待つ
        while not (len(self.client.has_what()) > len(worker_addresses)):
            sleep(1)
        new_worker_addresses = list(self.client.has_what().keys())
        launched_worker_addresses = [w_adrs for w_adrs in new_worker_addresses if not w_adrs in worker_addresses]
        assert len(launched_worker_addresses) == 1
        launched_worker_address = launched_worker_addresses[0]

        # actor の設定
        self.status = OptimizationStatus(self.client)
        self.status.set(OptimizationStatus.SETTING_UP)
        self.history = History(self.history_path, self.client)
        self.history.init(
            self.opt.parameters['name'].to_list(),
            list(self.opt.objectives.keys()),
            list(self.opt.constraints.keys()),
        )

        # 共通引数
        self.opt.n_trials = n_trials
        self.opt.timeout = timeout

        # fem
        self.fem.setup_before_parallel(self.client)

        # opt
        self.opt.fem_class = type(self.fem)
        self.opt.fem_kwargs = self.fem.kwargs
        self.opt.status = self.status
        self.opt.history = self.history
        self.opt.setup_before_parallel()

        # launch monitor
        self.monitor_process_future = self.client.submit(
            start_monitor_server_forever,
            self.history, self.status,  # args
            **self.monitor_server_kwargs,  # kwargs
            workers=launched_worker_address,  # if invalid arg,
            allow_other_workers=False
        )

        # クラスターでの計算開始
        self.status.set(OptimizationStatus.LAUNCHING)
        start = time()
        subprocess_indices = list(range(n_parallel))

        if self.opt.is_cluster:
            # FIXME: monitor-server-process には割り当てないようにする
            worker_addresses = list(self.client.nthreads().keys())
            worker_addresses.remove(launched_worker_address)
            calc_futures = self.client.map(
                self.opt._main,
                subprocess_indices,
                workers=worker_addresses,
                allow_other_workers=False,
            )

        else:
            # ローカルなら既存の fem をひとつ使う
            calc_futures = self.client.map(self.opt._main, subprocess_indices[1:])

            # ローカルプロセスでの計算開始
            # constants の restore （map の後なら大丈夫）
            # TODO: 最適化中に cluster を scale する場合はここに気を付ける
            for obj in self.opt.objectives.values():
                obj._restore_constants()
            for cns in self.opt.constraints.values():
                cns._restore_constants()
            self.opt.fem = self.fem
            t_main = Thread(target=self.opt.main, args=(subprocess_indices[0],))
            t_main.start()

        # save history
        def save_history():
            while True:
                sleep(2)
                try:
                    self.history.actor_data.to_csv(self.history.path, index=None, encoding='shift-jis')
                except PermissionError:
                    pass
                if self.status.get() == OptimizationStatus.TERMINATED:
                    break
        t_save_history = Thread(target=save_history)
        t_save_history.start()


        # 終了を待つ
        self.client.gather(calc_futures)
        if not self.opt.is_cluster:  # 既存の fem を使っているならそれも待つ
            t_main.join()
        self.status.set(OptimizationStatus.TERMINATED)
        end = time()

        # 一応
        t_save_history.join()

        print(f'Optimization finished. Elapsed time is {end - start} sec.')
        print('計算が終了しました. ウィンドウを閉じると終了します.')
        print(f'結果は{self.history.path}を確認してください.')

    def terminate_all(self):

        # monitor が terminated 状態で少なくとも一度更新されなければ running のまま固まる
        sleep(1)

        # terminate monitor process
        self.status.set(OptimizationStatus.TERMINATE_ALL)
        print(self.monitor_process_future.result())
        sleep(1)

        # terminate actors
        self.client.cancel(self.history._future, force=True)
        self.client.cancel(self.status._future, force=True)
        print('Terminate actors.')
        sleep(1)

        # terminate monitor worker
        n_workers = len(self.client.nthreads())
        self.client.retire_workers(
            names=[self.monitor_process_worker_name],
            close_workers=True,
            remove=True,
        )
        while n_workers == len(self.client.nthreads()):
            sleep(1)
        print('Terminate monitor processes worker.')
        sleep(1)

        # close scheduler, other workers(, cluster)
        self.client.close()
        while self.client.scheduler is not None:
            sleep(1)
        print('Terminate client.')

        # terminate dask relative processes.
        if not self.opt.is_cluster:
            self.client.shutdown()
            print('Terminate all relative processes.')
        sleep(3)


def start_monitor_server_forever(history, status, host='localhost', port=8080):
    monitor = Monitor(history, status)
    monitor.start_server(host, port)
    return 'Exit monitor server process gracefully'

