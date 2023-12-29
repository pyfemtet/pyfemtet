from abc import ABC, abstractmethod

import os
import datetime
import inspect
import ast
import math
from time import time, sleep
from threading import Thread

import numpy as np
import pandas as pd
from optuna._hypervolume import WFG
import ray

from win32com.client import Constants

from .core import InterprocessVariables, UserInterruption, TerminatableThread, Scapegoat, restore_constants_from_scapegoat
from .interface import FEMInterface, FemtetInterface
from .monitor import Monitor


def symlog(x: float | np.ndarray):
    """Log function whose domain is extended to the negative region.

    Symlog processing is performed internally as a measure to reduce
    unintended trends caused by scale differences
    between objective functions in multi-objective optimization.

    Args:
        x (float | np.ndarray)

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


def _ray_are_alive(refs):
    ready_refs, remaining_refs = ray.wait(refs, num_returns=len(refs), timeout=0)
    return len(remaining_refs) > 0


class Function:
    """Base class for Objective and Constraint."""

    def __init__(self, fun, name, args, kwargs):
        # unserializable な COM 定数を parallelize するための処理
        for varname in fun.__globals__:
            if isinstance(fun.__globals__[varname], Constants):
                fun.__globals__[varname] = Scapegoat()
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


@ray.remote
class HistoryDfCore:
    """Class for managing a DataFrame object in a distributed manner."""

    def __init__(self, df):
        self.df = df

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
    def __init__(self, history_path):
        """Initializes a History instance.

        Args:
            history_path (str): The path to the history file.

        """

        # 引数の処理
        self.path = history_path  # .csv
        self._actor_data = HistoryDfCore.remote(pd.DataFrame())
        self.data = pd.DataFrame()
        self.param_names = []
        self.obj_names = []
        self.cns_names = []

        # path が存在すれば dataframe を読み込む
        if os.path.isfile(self.path):
            self.actor_data = pd.read_csv(self.path)
            self.data = pd.read_csv(self.path)

    @property
    def actor_data(self):
        return ray.get(self._actor_data.get_df.remote())

    @actor_data.setter
    def actor_data(self, df):
        self._actor_data.set_df.remote(df)

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
            self.data = self.actor_data.copy()

    def record(self, parameters, objectives, constraints, obj_values, cns_values, message):
        """Records the optimization results in the history.

        Args:
            parameters (dict): The parameter values.
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

        # append
        if len(self.actor_data) == 0:
            self.actor_data = pd.DataFrame([row], columns=self.actor_data.columns)
        else:
            tmp = self.actor_data
            tmp.loc[len(tmp)] = row
            self.actor_data = tmp

        # calc
        try:
            tmp = self.actor_data
            tmp['trial'] = np.arange(len(tmp)) + 1  # 1 始まり
            self.actor_data = tmp
            self._calc_non_domi(objectives)
            self._calc_hypervolume(objectives)
        except (ValueError, pd.errors.IndexingError):  # 計算中に別のプロセスが append した場合、そちらに処理を任せる
            pass

        # serialize
        try:
            self.actor_data.to_csv(self.path, index=None)
        except PermissionError:
            print(f'warning: {self.path} がロックされています。データはロック解除後に保存されます。')

        # unparallelize
        self.data = self.actor_data.copy()

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
        #### 前準備
        # パレート集合の抽出
        idx = self.actor_data['non_domi']
        pdf = self.actor_data[idx]
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
        df = pd.DataFrame(self.actor_data.to_dict())  # read-only error 回避
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
        self.actor_data = df


class OptimizerBase(ABC):
    """Base class for optimization algorithms.

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

    def __init__(self, fem: FEMInterface = None, history_path=None):
        """Initializes an OptimizerBase instance.

        Args:
            fem (FEMInterface, optional): The finite element method interface. Defaults to None. If None, automattically set to FemtetInterface.
            history_path (str, optional): The path to the history file. Defaults to None. If None, '%Y_%m_%d_%H_%M_%S.csv' is created in current directory.

        """
        print('---initialize---')

        ray.init(ignore_reinit_error=True)

        # 引数の処理
        if history_path is None:
            history_path = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S.csv')
        self.history_path = os.path.abspath(history_path)
        if fem is None:
            self.fem = FemtetInterface()
        else:
            self.fem = fem

        # メンバーの宣言
        self.ipv = InterprocessVariables()
        self.parameters = pd.DataFrame()
        self.objectives = dict()
        self.constraints = dict()
        self.history = History(self.history_path)
        self.monitor: Monitor = None
        self.monitor_thread = None
        self.monitor_server_kwargs = dict()
        self.seed: int or None = None
        self.message = ''
        self.obj_values: [float] = []
        self.cns_values: [float] = []
        self._fem_class = type(self.fem)
        self._fem_kwargs = self.fem.kwargs.copy()

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
        del state['monitor_thread']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def set_fem(self, **extra_kwargs):
        """Sets or resets the finite element method interface.

        Args:
            **extra_kwargs: Additional keyword arguments to be passed to the FEMInterface constructor.

        """
        fem_kwargs = self._fem_kwargs.copy()
        fem_kwargs.update(extra_kwargs)
        self.fem = self._fem_class(
            **fem_kwargs,
        )

    def set_random_seed(self, seed: int):
        """Sets the random seed for reproducibility.

        Args:
            seed (int): The random seed value to be set.

        """
        self.seed = seed

    def get_random_seed(self):
        """Returns the current random seed value.

        Returns:
            int: The current random seed value.

        """
        return self.seed

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
                is_existing = candidate in list(self.objectives.keys())
                if not is_existing:
                    break
                else:
                    i += 1
            name = candidate

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
                is_existing = candidate in list(self.objectives.keys())
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

        self.constraints[name] = Constraint(fun, name, lower_bound, upper_bound, strict, args, kwargs)

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

    def is_calculated(self, x):
        """Checks if the proposed x is the last calculated value.

        Args:
            x (iterable): The proposed x value.

        Returns:
            bool: True if the proposed x is the last calculated value, False otherwise.

        """

        # 提案された x が最後に計算したものと一致していれば True
        # ただし 1 回目の計算なら False
        #    ひとつでも違う  1回目の計算  期待
        #    True            True         False
        #    False           True         False
        #    True            False        False
        #    False           False        True

        # 1 回目の計算
        if len(self.history.data) == 0:
            return False

        # ひとつでも違う
        param_names = self.history.param_names
        last_x = self.history.data[param_names].iloc[-1].values
        condition = False
        for _x, _last_x in zip(x, last_x):
            condition = condition or (float(_x) != float(_last_x))

        return not condition


    def f(self, x, message=''):
        """Calculates the objective function values for the given parameter values.

        Args:
            x (iterable): The parameter values.
            message (str, optional): Additional information about the calculation. Defaults to ''.

        Raises:
            UserInterruption: If the calculation is interrupted.

        Returns:
            list: The converted objective function values.
        
        Note:
            The return value is not the return value of a user-defined function,
            but the converted value when reconsidering the optimization problem to be minimize problem.
            See :func:`Objective.convert` for detail.

        """

        x = np.array(x)

        # 中断指令の処理
        if self.ipv.get_state() == 'interrupted':
            raise UserInterruption

        # アルゴリズムの関係で、すでに計算しているのにもう一度計算しようとする場合は
        # FEM 解析を飛ばす
        if not self.is_calculated(x):

            # parameter の update
            self.parameters['value'] = x

            # fem のソルブ
            self.fem.update(self.parameters)

            # constants への参照を復帰させる
            # parallel_process の中でこれを実行するとメインプロセスで restore されなくなるし、
            # main の中 parallel_process の前にこれを実行すると unserializability に引っかかる
            # メンバー変数の列挙
            for attr_name in dir(self):
                if attr_name.startswith('__'):
                    continue
                # メンバー変数の取得
                attr_value = getattr(self, attr_name)
                # メンバー変数が辞書なら
                if isinstance(attr_value, dict):
                    for _, value in attr_value.items():
                        # 辞書の value が Function なら
                        if isinstance(value, Function):
                            restore_constants_from_scapegoat(value)

            # 計算
            self.obj_values = [float(obj.calc(self.fem)) for _, obj in self.objectives.items()]
            self.cns_values = [float(cns.calc(self.fem)) for _, cns in self.constraints.items()]

            # 記録
            if self.fem.subprocess_idx is not None:
                message = message + f'; by subprocess{self.fem.subprocess_idx}'
            self.history.record(self.parameters, self.objectives, self.constraints, self.obj_values, self.cns_values, message)

        # minimize
        return [obj.convert(v) for (_, obj), v in zip(self.objectives.items(), self.obj_values)]


    @abstractmethod
    def concrete_main(self, *args, **kwargs):
        """The main function for the concrete class to implement.

        if ``n_trials`` >= 2, this method will be called by a parallel process.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        pass

    def setup_concrete_main(self, *args, **kwargs):
        """Performs the setup for the concrete class.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        pass

    def setup_monitor_server(self, host, port=None):
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

    def main(self, n_trials=None, n_parallel=1, timeout=None, method='TPE', **setup_kwargs):
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
            If ``n_trials`` and ``timeout`` are both None, it will calculate repeatedly until interrupted by the user.
        
        Note:
            If ``n_parallel`` >= 2, depending on the end timing, ``n_trials`` may be exceeded by up to ``n_parallel-1`` times.

        Note:
            Currently, supported methods are 'TPE' and 'botorch'.
            For detail, see https://optuna.readthedocs.io/en/stable/reference/samplers/index.html.


        """

        # 共通引数
        self.n_trials = n_trials
        self.n_parallel = n_parallel
        self.timeout = timeout
        self.method = method

        # setup
        self.ipv.set_state('preparing')
        self.history.init(
            self.parameters['name'].to_list(),
            list(self.objectives.keys()),
            list(self.constraints.keys()),
        )
        self.setup_concrete_main(**setup_kwargs)  # 具象クラス固有のメソッド

        # 計算スレッドとそれを止めるためのイベント
        t = Thread(target=self.concrete_main)
        t.start()  # Exception が起きてもここでは検出できないし、メインスレッドは落ちない

        # 計算開始
        self.ipv.set_state('processing')

        # モニタースレッド
        self.monitor = Monitor(self)
        self.monitor_thread = TerminatableThread(
            target=self.monitor.start_server,
            kwargs=self.monitor_server_kwargs
        )
        self.monitor_thread.start()

        # 追加の計算プロセスが行う処理の定義
        @ray.remote
        def parallel_process(_subprocess_idx, _subprocess_settings):
            print('Start to re-initialize fem object.')
            # プロセス化されたときに del した fem を restore する
            self.set_fem(
                subprocess_idx=_subprocess_idx,
                subprocess_settings=_subprocess_settings
            )
            print('Start to setup parallel process.')
            self.fem.parallel_setup()
            print('Start parallel optimization.')
            try:
                self.concrete_main(_subprocess_idx)
            except UserInterruption:
                pass
            print('Finish parallel optimization.')
            self.fem.parallel_terminate()
            print('Finish parallel process.')

        # 追加の計算プロセスを立てる前の前処理
        subprocess_settings = self.fem.settings_before_parallel(self)

        # 追加の計算プロセス
        obj_refs = []
        for subprocess_idx in range(self.n_parallel-1):
            obj_ref = parallel_process.remote(subprocess_idx, subprocess_settings)
            obj_refs.append(obj_ref)

        start = time()
        while True:
            should_terminate = [not t.is_alive(), not _ray_are_alive(obj_refs)]
            if all(should_terminate):  # all tasks are killed
                break
            sleep(1)
        end = time()

        # 一応
        t.join()
        ray.wait(obj_refs)

        print(f'Optimization finished. Elapsed time is {end - start} sec.')
        self.ipv.set_state('terminated')
        print('計算が終了しました. ウィンドウを閉じると終了します.')
        print(f'結果は{self.history.path}を確認してください.')

        # shutdown 前に ray remote actor を消しておく
        ray.kill(self.ipv.ns)
        del self.ipv.ns
        for obj in obj_refs:
            del obj

        ray.shutdown()

    def terminate_monitor(self):
        """Forcefully terminates the monitor thread."""
        self.monitor_thread.force_terminate()
