# typing
from typing import List

# built-in
import os
import datetime
import inspect
import ast
import csv
import ctypes

# 3rd-party
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube
from optuna._hypervolume import WFG
from dask.distributed import Lock, get_client

# win32com
from win32com.client import constants, Constants

# pyfemtet relative
from pyfemtet.opt.interface import FEMInterface, FemtetInterface

# logger
import logging
from pyfemtet.logger import get_logger
logger = get_logger('femopt')
logger.setLevel(logging.INFO)


__all__ = [
    'generate_lhs',
    '_check_bound',
    '_is_access_gogh',
    'is_feasible',
    'Objective',
    'Constraint',
    'History',
    'OptimizationStatus',
    'logger',
]


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
        for j in range(2, p):
            if p % j == 0:
                return False
        return True

    def get_prime(_minimum):
        for p in range(_minimum, LIMIT):
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


def _check_bound(lb, ub, name=None):
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


def is_feasible(value, lb, ub):
    """
    Check if a value is within the specified lower bound and upper bound.

    Args:
        value (numeric): The value to check.
        lb (optional, numeric): The lower bound. If not specified, there is no lower bound.
        ub (optional, numeric): The upper bound. If not specified, there is no upper bound.

    Returns:
        bool: True if the value satisfies the bounds; False otherwise.
    """
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
        # ParametricIF で使う dll 関数は _FuncPtr 型であって __globals__ を持たないが、
        # これは絶対に constants を持たないので単に無視すればよい。
        if not isinstance(fun, ctypes._CFuncPtr):
            for varname in fun.__globals__:
                if isinstance(fun.__globals__[varname], Constants):
                    fun.__globals__[varname] = _Scapegoat()

        self.fun = fun
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def calc(self, fem: FEMInterface):
        """Execute user-defined fun.

        Args:
            fem (FEMInterface)

        Returns:
            float
        """
        args = self.args
        # Femtet 特有の処理
        if isinstance(fem, FemtetInterface):
            args = (fem.Femtet, *args)
        return float(self.fun(*args, **self.kwargs))

    def _restore_constants(self):
        """Helper function for parallelize Femtet."""
        fun = self.fun
        if not isinstance(fun, ctypes._CFuncPtr):
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

        Note:
            If FEMOpt.fem is a instance of FemtetInterface or its subclass,
            the 1st argument of fun is set to fem automatically.


        """
        self._check_direction(direction)
        self.direction = direction
        super().__init__(fun, name, args, kwargs)

    def convert(self, value: float):
        """Converts an evaluation value to the value of objective function based on the specified direction.

        When direction is `'minimize'`, ``value`` is calculated.
        When direction is `'maximize'`, ``-value`` is calculated.
        When direction is float, ``abs(value - direction)`` is calculated.
        Finally, the calculated value is passed to the symlog function and returns it.

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

        return float(ret)

    def _check_direction(self, direction):
        message = '評価関数の direction は "minimize", "maximize", 又は数値でなければなりません.'
        message += f'与えられた値は {direction} です.'
        if isinstance(direction, float) or isinstance(direction, int):
            pass
        elif isinstance(direction, str):
            if (direction != 'minimize') and (direction != 'maximize'):
                raise ValueError(message)
        else:
            raise ValueError(message)


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

        _check_bound(lb, ub)
        self.lb = lb
        self.ub = ub
        self.strict = strict
        super().__init__(fun, name, args, kwargs)


class _HistoryDfCore:
    """Class for managing a DataFrame object in a distributed manner."""

    def __init__(self):
        self.df = pd.DataFrame()

    def set_df(self, df):
        self.df = df

    def get_df(self):
        return self.df


class History:
    """Class for managing the history of optimization results.

    Args:
        history_path (str): The path to the csv file.
        prm_names (List[str], optional): The names of parameters. Defaults to None.
        obj_names (List[str], optional): The names of objectives. Defaults to None.
        cns_names (List[str], optional): The names of constraints. Defaults to None.
        client (dask.distributed.Client): Dask client.
        additional_metadata (str, optional): metadata of optimization process.

    Raises:
        FileNotFoundError: If the csv file is not found.

    Attributes:
        HEADER_ROW (int): Header row number of csv file. Must be grater than 0. Default to 2.
        ENCODING (str): Encoding of csv file. Default to 'cp932'.
        prm_names (str): User defined names of parameters.
        obj_names (str): User defined names of objectives.
        cns_names (str): User defined names of constraints.
        local_data (pd.DataFrame): Local copy (on memory) of optimization history. 
        is_restart (bool): If the optimization process is a continuation of another process or not.
        is_processing (bool): The optimization is running or not.

    """

    HEADER_ROW = 2
    ENCODING = 'cp932'
    prm_names = []
    obj_names = []
    cns_names = []
    is_restart = False
    is_processing = False
    _future = None
    _actor_data = None

    def __init__(
            self,
            history_path,
            prm_names=None,
            obj_names=None,
            cns_names=None,
            client=None,
            additional_metadata=None,
    ):

        # 引数の処理
        self.path = history_path  # .csv
        self.prm_names = prm_names
        self.obj_names = obj_names
        self.cns_names = cns_names
        self.additional_metadata = additional_metadata or ''

        # 初期化
        self.local_data = pd.DataFrame()

        # 最適化実行中かどうか
        self.is_processing = client is not None

        # 最適化実行中の process monitor である場合
        if self.is_processing:

            # actor の生成
            self._future = client.submit(_HistoryDfCore, actor=True)
            self._actor_data = self._future.result()

            # csv が存在すれば続きからモード
            self.is_restart = os.path.isfile(self.path)

            # 続きからなら df を読み込んで df にコピー
            if self.is_restart:
                self.load()

            # そうでなければ df を初期化
            else:
                columns, metadata = self.create_df_columns()
                for c in columns:
                    self.local_data[c] = None
                self.metadata = metadata

            # actor_data の初期化
            self.actor_data = self.local_data

            # 一時ファイルに書き込みを試み、UnicodeEncodeError が出ないかチェック
            import tempfile
            try:
                with tempfile.TemporaryFile() as f:
                    self.save(_f=f)
            except UnicodeEncodeError:
                raise ValueError('変数名、目的名または拘束名にエンコードできない文字が含まれています。環境依存文字は使用しないでください。')

        # visualization only の場合
        else:
            # csv が存在しなければおかしい
            if not os.path.isfile(self.path):
                raise FileNotFoundError(f'{self.path} が見つかりません。')

            # csv の local_data へと、names への読み込み
            self.load()

    def load(self):
        """Load existing result csv."""

        # df を読み込む
        self.local_data = pd.read_csv(self.path, encoding=self.ENCODING, header=self.HEADER_ROW)

        # metadata を読み込む
        with open(self.path, mode='r', encoding=self.ENCODING, newline='\n') as f:
            reader = csv.reader(f, delimiter=',')
            self.metadata = reader.__next__()

        # 最適化問題を読み込む
        columns = self.local_data.columns
        prm_names = [column for i, column in enumerate(columns) if self.metadata[i] == 'prm']
        obj_names = [column for i, column in enumerate(columns) if self.metadata[i] == 'obj']
        cns_names = [column for i, column in enumerate(columns) if self.metadata[i] == 'cns']

        # is_restart の場合、読み込んだ names と引数の names が一致するか確認しておく
        if self.is_restart:
            if prm_names != self.prm_names: raise ValueError(f'実行中の設定が csv ファイルの設定と一致しません。')
            if obj_names != self.obj_names: raise ValueError(f'実行中の設定が csv ファイルの設定と一致しません。')
            if cns_names != self.cns_names: raise ValueError(f'実行中の設定が csv ファイルの設定と一致しません。')

        # visualization only の場合、読み込んだ names をも load する
        if not self.is_processing:
            self.prm_names = prm_names
            self.obj_names = obj_names
            self.cns_names = cns_names

    @property
    def actor_data(self):
        return self._actor_data.get_df().result()

    @actor_data.setter
    def actor_data(self, df):
        self._actor_data.set_df(df).result()

    def create_df_columns(self):
        """Create columns of history."""

        # df として保有するカラムを生成
        columns = list()

        # columns のメタデータを作成
        metadata = list()

        # trial
        columns.append('trial')  # index
        metadata.append(self.additional_metadata)

        # parameter
        columns.extend(self.prm_names)
        metadata.extend(['prm'] * len(self.prm_names))

        # objective relative
        for name in self.obj_names:
            columns.append(name)
            metadata.append('obj')
            columns.append(name + '_direction')
            metadata.append('obj_direction')
        columns.append('non_domi')
        metadata.append('')

        # constraint relative
        for name in self.cns_names:
            columns.append(name)
            metadata.append('cns')
            columns.append(name + '_lower_bound')
            metadata.append('cns_lb')
            columns.append(name + '_upper_bound')
            metadata.append('cns_ub')
        columns.append('feasible')
        metadata.append('')

        # the others
        columns.append('hypervolume')
        metadata.append('')
        columns.append('message')
        metadata.append('')
        columns.append('time')
        metadata.append('')

        return columns, metadata

    def record(
            self,
            parameters,
            objectives,
            constraints,
            obj_values,
            cns_values,
            message,
            postprocess_func,
            postprocess_args,
    ):
        """Records the optimization results in the history.

        Record only. NOT save.

        Args:
            parameters (pd.DataFrame): The parameter values.
            objectives (dict): The objective functions.
            constraints (dict): The constraint functions.
            obj_values (list): The objective values.
            cns_values (list): The constraint values.
            message (str): Additional information or messages related to the optimization results.
            postprocess_func (Callable): fem method to call after solving. i.e. save result file. Must take trial(int) for 1st argument.
            postprocess_args (dict): arguments for `postprocess_func`. i.e. create binary data of result file in the worker process.
        """

        # create row
        row = list()

        # trial(dummy)
        row.append(-1)

        # parameters
        row.extend(parameters['value'].values)

        # objectives and their direction
        for (_, obj), obj_value in zip(objectives.items(), obj_values):  # objectives, direction
            row.extend([obj_value, obj.direction])

        # non_domi (dummy)
        row.append(False)

        # constraints and their lb, ub and calculate each feasibility
        feasible_list = []
        for (_, cns), cns_value in zip(constraints.items(), cns_values):  # cns, lb, ub
            row.extend([cns_value, cns.lb, cns.ub])
            feasible_list.append(is_feasible(cns_value, cns.lb, cns.ub))

        # feasibility
        row.append(all(feasible_list))

        # the others
        row.append(-1.)  # dummy hypervolume
        row.append(message)  # message
        row.append(datetime.datetime.now())  # time

        with Lock('calc-history'):
            # append
            if len(self.actor_data) == 0:
                self.local_data = pd.DataFrame([row], columns=self.actor_data.columns)
            else:
                self.local_data = self.actor_data
                self.local_data.loc[len(self.local_data)] = row

            # calc
            self.local_data['trial'] = np.arange(len(self.local_data)) + 1  # 1 始まり
            self._calc_non_domi(objectives)  # update self.local_data
            self._calc_hypervolume(objectives)  # update self.local_data
            self.actor_data = self.local_data

            # save file
            if postprocess_args is not None:
                trial = self.local_data['trial'].values[-1]
                client = get_client()  # always returns valid client
                client.run_on_scheduler(postprocess_func, trial, **postprocess_args)

    def _calc_non_domi(self, objectives):

        # 目的関数の履歴を取り出してくる
        solution_set = self.local_data[self.obj_names]

        # 最小化問題の座標空間に変換する
        for obj_column, (_, objective) in zip(self.obj_names, objectives.items()):
            solution_set.loc[:, obj_column] = solution_set[obj_column].map(objective.convert)

        # 非劣解の計算
        non_domi = []
        for i, row in solution_set.iterrows():
            non_domi.append((row > solution_set).product(axis=1).sum(axis=0) == 0)

        # 非劣解の登録
        self.local_data['non_domi'] = non_domi

    def _calc_hypervolume(self, objectives):

        # タイピングが面倒
        df = self.local_data

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
        for i, (_, objective) in enumerate(objectives.items()):
            for j in range(n):
                pareto_set[j, i] = objective.convert(pareto_set[j, i])
                #### reference point の計算[1]
        # 逆正規化のための範囲計算
        maximum = pareto_set.max(axis=0)
        minimum = pareto_set.min(axis=0)

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
            if not df.loc[i, 'non_domi']:
                try:
                    df.loc[i, 'hypervolume'] = df.loc[:i][df.loc[:i]['non_domi']].iloc[-1]['hypervolume']
                except IndexError:
                    df.loc[i, 'hypervolume'] = 0

    def save(self, _f=None):
        """Save csv file."""

        if _f is None:
            # save df with columns with prefix
            with open(self.path, 'w', encoding=self.ENCODING) as f:
                writer = csv.writer(f, delimiter=',', lineterminator="\n")
                writer.writerow(self.metadata)
                for i in range(self.HEADER_ROW-1):
                    writer.writerow([''] * len(self.metadata))
                self.actor_data.to_csv(f, index=None, encoding=self.ENCODING, lineterminator='\n')
        else:  # test
            self.actor_data.to_csv(_f, index=None, encoding=self.ENCODING, lineterminator='\n')


class _OptimizationStatusActor:
    status_int = -1
    status = 'undefined'

    def set(self, value, text):
        self.status_int = value
        self.status = text


class OptimizationStatus:
    """Optimization status."""

    UNDEFINED = -1
    INITIALIZING = 0
    SETTING_UP = 10
    LAUNCHING_FEM = 20
    WAIT_OTHER_WORKERS = 22
    # WAIT_1ST = 25
    RUNNING = 30
    INTERRUPTING = 40
    TERMINATED = 50
    TERMINATE_ALL = 60
    CRASHED = 70

    def __init__(self, client, name='entire'):
        self._future = client.submit(_OptimizationStatusActor, actor=True)
        self._actor = self._future.result()
        self.name = name
        self.set(self.INITIALIZING)

    @classmethod
    def const_to_str(cls, status_const):
        """Convert optimization status integer to message."""
        if status_const == cls.UNDEFINED: return 'Undefined'
        if status_const == cls.INITIALIZING: return 'Initializing'
        if status_const == cls.SETTING_UP: return 'Setting up'
        if status_const == cls.LAUNCHING_FEM: return 'Launching FEM processes'
        if status_const == cls.WAIT_OTHER_WORKERS: return 'Waiting for launching other processes'
        # if status_const == cls.WAIT_1ST: return 'Running and waiting for 1st FEM result.'
        if status_const == cls.RUNNING: return 'Running'
        if status_const == cls.INTERRUPTING: return 'Interrupting'
        if status_const == cls.TERMINATED: return 'Terminated'
        if status_const == cls.TERMINATE_ALL: return 'Terminate_all'
        if status_const == cls.CRASHED: return 'Crashed'

    def set(self, status_const):
        """Set optimization status."""
        self._actor.set(status_const, self.const_to_str(status_const)).result()
        msg = f'---{self.const_to_str(status_const)}---'
        if (status_const == self.INITIALIZING) and (self.name != 'entire'):
            msg += f' (for Worker {self.name})'
        if self.name == 'entire':
            msg = '(entire) ' + msg
        logger.info(msg)

    def get(self) -> int:
        """Get optimization status."""
        return self._actor.status_int

    def get_text(self) -> str:
        """Get optimization status message."""
        return self._actor.status
