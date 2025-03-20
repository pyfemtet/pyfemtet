# typing
import json
from typing import List, TYPE_CHECKING

# built-in
import os
import datetime
import inspect
import ast
import csv
import ctypes
from packaging import version
import platform
import enum
import warnings

# 3rd-party
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube
import optuna
if version.parse(optuna.version.__version__) < version.parse('4.0.0'):
    from optuna._hypervolume import WFG
    wfg = WFG()
    compute_hypervolume = wfg.compute
else:
    from optuna._hypervolume import wfg
    compute_hypervolume = wfg.compute_hypervolume
from dask.distributed import Lock, get_client, Client

# win32com
if (platform.system() == 'Windows') or TYPE_CHECKING:
    from win32com.client import constants, Constants
else:
    class Constants:
        pass
    constants = None

# pyfemtet relative
from pyfemtet.opt.interface import FEMInterface, FemtetInterface
from pyfemtet._message import encoding, Msg

# logger
from pyfemtet.logger import get_module_logger

logger = get_module_logger('opt.core', __name__)


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
    message = Msg.ERR_CHECK_MINMAX
    if (lb is not None) and (ub is not None):
        if lb > ub:
            raise ValueError(message)


def _get_scope_indent(source: str) -> int:
    SPACES = [' ', '\t']
    indent = 0
    while True:
        if source[indent] not in SPACES:
            break
        else:
            indent += 1
    return indent


def _remove_indent(source: str, indent: int) -> str:  # returns source
    lines = source.splitlines()
    edited_lines = [l[indent:] for l in lines]
    edited_source = '\n'.join(edited_lines)
    return edited_source


def _check_access_femtet_objects(fun, target: str = 'Femtet'):

    # 関数fのソースコードを取得
    source = inspect.getsource(fun)

    # ソースコードを抽象構文木（AST）に変換
    try:
        # instanceメソッドなどの場合を想定してインデントを削除
        source = _remove_indent(source, _get_scope_indent(source))
        tree = ast.parse(source)

    except Exception:
        return False  # パースに失敗するからと言ってエラーにするまででもない

    # if function or staticmethod, 1st argument is Femtet. Find the name.
    varname_contains_femtet = ''  # invalid variable name
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            all_arguments: ast.arguments = node.args

            args: list[ast.arg] = all_arguments.args
            # args.extend(all_arguments.posonlyargs)  # 先にこっちを入れるべきかも

            target_arg = args[0]

            # if class method or instance method, 2nd argument is it.
            # In this implementation, we cannot detect the FunctionDef is
            # method or not because the part of source code is unindented and parsed.
            if target_arg.arg == 'self' or target_arg.arg == 'cls':
                if len(args) > 1:
                    target_arg = args[1]
                else:
                    target_arg = None

            if target_arg is not None:
                varname_contains_femtet = target_arg.arg

    # check Femtet access
    if target == 'Femtet':
        for node in ast.walk(tree):

            # by accessing argument directory
            if isinstance(node, ast.Name):
                # found local variables
                node: ast.Name
                if node.id == varname_contains_femtet:
                    # found Femtet
                    return True

            # by accessing inside method
            elif isinstance(node, ast.Attribute):
                # found attribute of something
                node: ast.Attribute
                if node.attr == 'Femtet':
                    # found **.Femtet.**
                    return True

    # check Gogh access
    elif target == 'Gogh':
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr == 'Gogh':
                    # found **.Gogh.**
                    node: ast.Attribute
                    parent = node.value

                    # by accessing argument directory
                    if isinstance(parent, ast.Name):
                        # found *.Gogh.**
                        parent: ast.Name
                        if parent.id == varname_contains_femtet:
                            # found Femtet.Gogh.**
                            return True

                    # by accessing inside method
                    if isinstance(parent, ast.Attribute):
                        # found **.*.Gogh.**
                        parent: ast.Attribute
                        if parent.attr == 'Femtet':
                            # found **.Femtet.Gogh.**
                            return True

    # ここまで来たならば target へのアクセスはおそらくない
    return False


def _is_access_gogh(fun):
    return _check_access_femtet_objects(fun, target='Gogh')


def _is_access_femtet(fun):
    return _check_access_femtet_objects(fun, target='Femtet')


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
    if np.isnan(value):
        return False
    if lb is None and ub is not None:
        return value <= ub
    elif lb is not None and ub is None:
        return lb <= value
    elif lb is not None and ub is not None:
        return lb <= value <= ub
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
        if fun is not None:
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
        if self.fun is None:
            RuntimeError(f'`fun` of {self.name} is not specified.')

        args = self.args

        # Femtet 特有の処理
        if isinstance(fem, FemtetInterface):
            args = (fem.object_passed_to_functions, *args)

        else:

            # 仕様変更までの猶予処理:
            #   fun の引数と与えられている引数を比較し
            #   与えられている positional 引数が 1 足りなければ
            #   fem を加える

            # fun の引数を調べる
            req_pos_params = dict()
            parameters = inspect.signature(self.fun).parameters
            for k, v in parameters.items():
                if v.default == v.empty:
                    req_pos_params.update({k: v})

            # fun の positional 引数の内
            # 与えられていないものを探す
            
            # まず kwargs を引く
            # これで req_pos_params のうち
            # kw 指定されていないものが残る
            [req_pos_params.pop(k) for k in self.kwargs.keys() if k in req_pos_params.keys()]
            
            # 数を比較し、足りなければ追加する
            if len(req_pos_params) - len(args) == 1:
                args = (fem.object_passed_to_functions, *args)

            # 数が同じならば FutureWarning
            elif len(req_pos_params) == len(args):

                shown = False
                message = ("When using an interface other than Femtet, "
                           "it is planned that the first argument will "
                           "be automatically passed as FEMInterface in "
                           "future versions. To accommodate this change,"
                           " please ensure that your function takes an "
                           "argument of type FEMInterface as its first "
                           "parameter.")

                for fil in warnings.filters:
                    if fil[1] is not None:  # re.compile(message)
                        if fil[1].pattern == message:
                            shown = True

                if not shown:
                    warnings.filterwarnings(
                        'once',
                        message,
                        FutureWarning,
                    )
                    warnings.warn(
                        message,
                        FutureWarning,
                    )

            # 数が合わないが、次でエラーになるほうが
            # 分かりやすいので何もしない
            else:
                pass

        return float(self.fun(*args, **self.kwargs))

    def _restore_constants(self):
        """Helper function for parallelize Femtet."""
        fun = self.fun
        if fun is not None:
            if not isinstance(fun, ctypes._CFuncPtr):
                for varname in fun.__globals__:
                    if isinstance(fun.__globals__[varname], _Scapegoat):
                        if not fun.__globals__[varname]._ignore_when_restore_constants:
                            fun.__globals__[varname] = constants


class Objective(Function):
    """Class for registering user-defined objective function."""

    default_name = 'obj'

    def __init__(self, fun, name, direction, args, kwargs, with_symlog=False):
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
        self.with_symlog = with_symlog
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

        if self.with_symlog:
            ret = symlog(ret)

        return float(ret)

    @staticmethod
    def _check_direction(direction):
        message = Msg.ERR_CHECK_DIRECTION
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

    def __init__(self, fun, name, lb, ub, strict, args, kwargs, using_fem):
        """Initializes a Constraint instance.

        Args:
            fun: The user-defined constraint function.
            name (str): The name of the constraint function.
            lb: The lower bound of the constraint.
            ub: The upper bound of the constraint.
            strict (bool): Whether to enforce strict inequality for the bounds.
            args: Additional arguments for the constraint function.
            kwargs: Additional keyword arguments for the constraint function.
            using_fem: Update fem or not before run calc().

        Raises:
            ValueError: If the lower bound is greater than or equal to the upper bound.

        """

        _check_bound(lb, ub)
        self.lb = lb
        self.ub = ub
        self.strict = strict
        self.using_fem = using_fem
        super().__init__(fun, name, args, kwargs)


class ObjectivesFunc:
    """複数の値を返す関数を objective として扱うためのクラス

    複数の値を返す関数を受け取る。

    最初に評価されたときに計算を実行し
    そうでない場合は保持した値を返す
    callable のリストを提供する。

    """

    def __init__(self, fun, n_return):
        self._evaluated = [False for _ in range(n_return)]
        self._values = [None for _ in range(n_return)]
        self._i = 0
        self.fun = fun
        self.n_return = n_return

    def __iter__(self):
        return self

    def __next__(self):

        # iter の長さ
        if self._i == self.n_return:
            self._i = 0
            raise StopIteration

        # iter として提供する callable オブジェクト
        # self の情報にもアクセスする必要があり
        # それぞれが iter された時点での i 番目という
        # 情報も必要なのでこのスコープで定義する必要がある
        class NthFunc:
            def __init__(self_, i):
                # 何番目の要素であるかを保持
                # self._i を直接参照すると
                # 実行時点での ObjectiveFunc の
                # 値を参照してしまう
                self_.i = i

            def __call__(self_, *args, **kwargs):
                # 何番目の要素であるか
                i = self_.i

                # 一度も評価されていなければ評価する
                if not any(self._evaluated):
                    self._values = tuple(self.fun(*args, **kwargs))
                    assert len(self._values) == self.n_return, '予期しない戻り値の数'

                # 評価したらフラグを立てる
                self._evaluated[i] = True

                # すべてのフラグが立ったらクリアする
                if all(self._evaluated):
                    self._evaluated = [False for _ in range(self.n_return)]

                # 値を返す
                return self._values[i]

            @property
            def __globals__(self_):
                # ScapeGoat 実装への対処
                return self.fun.__globals__

        # callable を作成
        f = NthFunc(self._i)

        # index を更新
        self._i += 1

        return f


class _HistoryColumnProcessor:

    def __init__(
            self,
            history,
            prm_names,
            obj_names,
            cns_names,
            sub_fidelity_names,
    ):
        self.history: 'History' = history
        self.prm_names: list[str] | None = prm_names
        self.obj_names: list[str] | None = obj_names
        self.cns_names: list[str] | None = cns_names
        self.sub_fidelity_names: list[str] | None = sub_fidelity_names

    @staticmethod
    def extract_prm_names(columns, meta_columns):
        return [c for c, m in zip(columns, meta_columns) if m == 'prm']

    @staticmethod
    def extract_obj_names(columns, meta_columns):
        return [c for c, m in zip(columns, meta_columns) if m == 'obj']

    @staticmethod
    def extract_cns_names(columns, meta_columns):
        return [c for c, m in zip(columns, meta_columns) if m == 'cns']

    @staticmethod
    def extract_fidelity_names(columns, meta_columns):
        out = []
        for c, m in zip(columns, meta_columns):
            if m.startswith('fidelity') and ('obj' not in m):
                name = c
                out.append(name)
        return out

    @staticmethod
    def extract_fidelity_obj_columns(columns, meta_columns, sub_fidelity_name):
        out = []
        for c, m in zip(columns, meta_columns):
            if m.startswith('fidelity') and ('_obj' in m):  # all
                name = c.split(' of ')[-1]
                if name == sub_fidelity_name:  # given name only
                    out.append(c)
        return out

    @staticmethod
    def extract_fidelity_column_name(columns, meta_columns, sub_fidelity_name):
        for c, m in zip(columns, meta_columns):
            if m.startswith('fidelity') and ('obj' not in m):
                name = c
                if name == sub_fidelity_name:
                    return c

    def parse_csv(self, path) -> tuple[
        pd.DataFrame,
        list,  # columns
        list,  # meta_columns
    ]:

        # df を読み込む
        df = pd.read_csv(path, encoding=self.history.ENCODING, header=self.history.HEADER_ROW)

        # meta_columns を読み込む
        with open(path, mode='r', encoding=self.history.ENCODING, newline='\n') as f:
            reader = csv.reader(f, delimiter=',')
            meta_columns = reader.__next__()

        # 最適化問題を読み込む
        columns = df.columns

        return df, columns, meta_columns


class History:
    """Class for managing the history of optimization results.

    Args:
        history_path (str): The path to the csv file.
        prm_names (List[str], optional): The names of parameters. Defaults to None.
        obj_names (List[str], optional): The names of objectives. Defaults to None.
        cns_names (List[str], optional): The names of constraints. Defaults to None.
        client (dask.distributed.Client): Dask client.
        hv_reference (str or list[float or np.ndarray, optional):
            The method to calculate hypervolume or
            the reference point itself.
            Valid values are 'dynamic-pareto' or
            'dynamic-nadir' or 'nadir' or 'pareto'
            or fixed point (in objective function space).

    """

    HEADER_ROW = 2
    ENCODING = encoding
    prm_names = []
    obj_names = []
    cns_names = []
    is_restart = False
    is_processing = False
    _df = None  # in case without client

    class OptTrialState(enum.Enum):
        succeeded = 'Succeeded'
        hidden_constraint_violation = 'Hidden Constraint Violation'
        strict_constraint_violation = 'Strict Constraint Violation'
        skipped = 'Skipped'

    def __init__(
            self,
            history_path,
            prm_names=None,
            obj_names=None,
            cns_names=None,
            sub_fidelity_names = None,
            client=None,
            hv_reference=None,
    ):
        # hypervolume 計算メソッド
        self._hv_reference = 'dynamic-pareto' if hv_reference is None else hv_reference

        # 引数の処理
        self.path = history_path  # .csv
        self.prm_names = prm_names
        self.obj_names = obj_names
        self.cns_names = cns_names
        self.sub_fidelity_names = sub_fidelity_names
        self.extra_data = dict()
        self.meta_columns = None
        self.__scheduler_address = client.scheduler.address if client is not None else None
        self._column_mgr = _HistoryColumnProcessor(
            self,
            self.prm_names,
            self.obj_names,
            self.cns_names,
            self.sub_fidelity_names,
        )

        # 最適化実行中かどうか
        self.is_processing = client is not None

        # 最適化実行中の process monitor である場合
        if self.is_processing:

            # csv が存在すれば続きからモード
            self.is_restart = os.path.isfile(self.path)

            # 続きからなら df を読み込んで df にコピー
            if self.is_restart:
                self.load()  # 中で meta_columns を読む

            # そうでなければ df を初期化
            else:
                columns, meta_columns = self.create_df_columns()
                df = pd.DataFrame(columns=columns)
                self.meta_columns = meta_columns
                self.set_df(df)

            # 一時ファイルに書き込みを試み、UnicodeEncodeError が出ないかチェック
            import tempfile
            try:
                with tempfile.TemporaryFile() as f:
                    self.save(_f=f)
            except UnicodeEncodeError:
                raise ValueError(Msg.ERR_CANNOT_ENCODING)

        # visualization only の場合
        else:
            # csv が存在しなければおかしい
            if not os.path.isfile(self.path):
                raise FileNotFoundError(self.path)

            # csv の読み込み
            self.load()

    def get_fidelity_column_name(self, sub_fidelity_name):
        columns, meta_columns = self.create_df_columns()
        return self._column_mgr.extract_fidelity_column_name(
            columns,
            meta_columns,
            sub_fidelity_name,
        )

    def get_obj_names_of_sub_fidelity(self, sub_fidelity_name):
        columns, meta_columns = self.create_df_columns()
        return self._column_mgr.extract_fidelity_obj_columns(
            columns,
            meta_columns,
            sub_fidelity_name
        )

    def load(self):
        """Load existing result csv."""

        # df を読み込む
        df, old_columns, old_meta_columns = self._column_mgr.parse_csv(self.path)

        # 最適化問題を読み込む
        prm_names = self._column_mgr.extract_prm_names(old_columns, old_meta_columns)
        obj_names = self._column_mgr.extract_obj_names(old_columns, old_meta_columns)
        cns_names = self._column_mgr.extract_cns_names(old_columns, old_meta_columns)
        sub_fidelity_names = self._column_mgr.extract_fidelity_names(old_columns, old_meta_columns)

        # is_restart の場合、読み込んだ names と引数の names が一致するか確認しておく
        if self.is_restart:
            if prm_names != self.prm_names: raise ValueError(Msg.ERR_PROBLEM_MISMATCH)
            if obj_names != self.obj_names: raise ValueError(Msg.ERR_PROBLEM_MISMATCH)
            if cns_names != self.cns_names: raise ValueError(Msg.ERR_PROBLEM_MISMATCH)

        # visualization only の場合、読み込んだ names をも load する
        if not self.is_processing:
            self.prm_names = prm_names
            self.obj_names = obj_names
            self.cns_names = cns_names
            self.sub_fidelity_names = sub_fidelity_names

        self.meta_columns = old_meta_columns

        self.set_df(df)

    def create_df_columns(self):
        """Create columns of history."""

        # df として保有するカラムを生成
        columns = list()

        # columns のメタデータを作成
        meta_columns = list()

        # trial
        columns.append('trial')  # index
        meta_columns.append('')  # extra_data. save 時に中身を記入する。

        # parameter
        for prm_name in self.prm_names:
            columns.extend([prm_name, prm_name + '_lower_bound', prm_name + '_upper_bound'])
            meta_columns.extend(['prm', 'prm_lb', 'prm_ub'])

        # objective relative
        for name in self.obj_names:
            columns.append(name)
            meta_columns.append('obj')
            columns.append(name + '_direction')
            meta_columns.append('obj_direction')
        columns.append('non_domi')
        meta_columns.append('')

        # constraint relative
        for name in self.cns_names:
            columns.append(name)
            meta_columns.append('cns')
            columns.append(name + '_lower_bound')
            meta_columns.append('cns_lb')
            columns.append(name + '_upper_bound')
            meta_columns.append('cns_ub')
        columns.append('feasible')
        meta_columns.append('')

        # sub-fidelity relative
        # add n_sub_fidelity * (n_fidelity(1) + n_obj columns)
        if self.sub_fidelity_names is not None:
            for i, sub_fidelity_name in enumerate(self.sub_fidelity_names):
                columns.append(sub_fidelity_name)
                meta_columns.append(f'fidelity{i}')
                for j, obj_name in enumerate(self.obj_names):
                    columns.append(f'{obj_name} of {sub_fidelity_name}')
                    meta_columns.append(f'fidelity{i}_obj{j}')

        # the others
        columns.append('hypervolume')
        meta_columns.append('')
        columns.append('message')
        meta_columns.append('')
        columns.append('state')
        meta_columns.append('')
        columns.append('time_start')
        meta_columns.append('')
        columns.append('time_end')
        meta_columns.append('')

        return columns, meta_columns

    def _record(
            self,
            parameters,
            objectives,
            constraints,
            obj_values,
            cns_values,
            sub_fidelity_obj_values: dict[str, tuple['Fidelity', list[float]]],
            message,
            state,
            time_start,
            time_end,
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
            state (str): The state messages.
            time_start (datetime.datetime | None): The time to start updating.
            time_end (datetime.datetime | None): The time to record this.
            postprocess_func (Callable): fem method to call after solving. i.e. save result file. Must take trial(int) for 1st argument.
            postprocess_args (dict): arguments for `postprocess_func`. i.e. create binary data of result file in the worker process.
        """

        # create row
        row = list()

        # trial(dummy)
        row.append(-1)

        # parameters
        for i, _row in parameters.iterrows():
            row.append(_row['value'])
            row.append(_row['lower_bound'])
            row.append(_row['upper_bound'])

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
        row.append(all(feasible_list) and not self.is_hidden_infeasible_result(obj_values))

        # sub-fidelity
        for sub_fidelity_name in self.sub_fidelity_names:

            # dict[str, tuple['Fidelity', list[float]]]
            if sub_fidelity_name in sub_fidelity_obj_values:
                fidelity, sub_y = sub_fidelity_obj_values[sub_fidelity_name]
                row.append(fidelity)
                row.extend(sub_y)

            else:
                fidelity = np.nan
                sub_y, _ = self.generate_hidden_infeasible_result()
                row.append(fidelity)
                row.extend(sub_y)

        # the others
        row.append(-1.)  # dummy hypervolume
        row.append(message)  # message
        row.append(state)  # state
        row.append(time_start)  # time_start
        row.append(time_end)  # time_complete

        with Lock('calc-history'):

            df = self.get_df()

            # append
            if len(df) == 0:
                df = pd.DataFrame([row], columns=df.columns)
            else:
                df.loc[len(df)] = row

            # calc
            df['trial'] = np.arange(len(df)) + 1  # 1 始まり
            self._calc_non_domi(objectives, df)  # update df
            self._calc_hypervolume(objectives, df)  # update df

            self.set_df(df)

            # save file
            if postprocess_args is not None:
                df = self.get_df()
                trial = df['trial'].values[-1]
                client = get_client()  # always returns valid client
                client.run_on_scheduler(postprocess_func, trial, **postprocess_args)

    def get_filtered_df(self, states: list[OptTrialState]) -> pd.DataFrame:
        df = self.get_df()
        indices = [s in set([state.value for state in states]) for s in df['state']]
        return df[indices]

    def filter_valid(self, df_, keep_trial_num=False, include_skip=False):
        buff = df_[self.obj_names].notna()
        idx = buff.prod(axis=1).astype(bool)
        filtered_df = df_[idx]
        if not keep_trial_num:
            filtered_df.loc[:, 'trial'] = np.arange(len(filtered_df)) + 1
        return filtered_df

    def get_df(self, valid_only=False, include_skip=False) -> pd.DataFrame:
        if self.__scheduler_address is None:
            if valid_only:
                return self.filter_valid(self._df)
            else:
                return self._df
        else:
            # scheduler がまだ存命か確認する
            try:
                with Lock('access-df'):
                    client_: 'Client' = get_client(self.__scheduler_address)

                    if 'df' in client_.list_datasets():
                        df = client_.get_dataset('df')

                        if valid_only:
                            df = self.filter_valid(df, include_skip)

                        return df

                    else:
                        logger.error('Access df of History before it is initialized.')
                        return pd.DataFrame()
            except OSError:
                logger.error('Scheduler is already dead. Most frequent reason to show this message is that the pyfemtet monitor UI is not refreshed even if the main optimization process is terminated.')
                return pd.DataFrame()

    def set_df(self, df: pd.DataFrame):
        if self.__scheduler_address is None:
            self._df = df
        else:
            try:
                with Lock('access-df'):
                    client_: 'Client' = get_client(self.__scheduler_address)
                    if 'df' in client_.list_datasets():
                        client_.unpublish_dataset('df')  # 更新する場合は前もって削除が必要、本来は dask collection をここに入れる使い方をする。
                    client_.publish_dataset(**dict(
                        df=df
                    ))
            except OSError:
                logger.error('Scheduler is already dead. Most frequent reasen to show this message is that the pyfemtet monitor UI is not refreshed even if the main optimization process is terminated.')

    def generate_hidden_infeasible_result(self):
        y = np.full_like(np.zeros(len(self.obj_names)), np.nan)
        c = np.full_like(np.zeros(len(self.cns_names)), np.nan)
        return y, c

    def is_hidden_infeasible_result(self, y):
        return np.all(np.isnan(y))

    def _calc_non_domi(self, objectives, df):

        # feasible のもののみ取り出してくる
        idx = df['feasible'].values
        pdf = df[idx]

        # 目的関数の履歴を取り出してくる
        solution_set = pdf[self.obj_names]

        # 最小化問題の座標空間に変換する
        for obj_column, (_, objective) in zip(self.obj_names, objectives.items()):
            solution_set.loc[:, obj_column] = solution_set[obj_column].map(objective.convert)

        # 非劣解の計算
        non_domi: list[bool] = []
        for i, row in solution_set.iterrows():
            non_domi.append((row > solution_set).product(axis=1).sum(axis=0) == 0)

        # feasible も infeasible も一旦劣解にする
        df['non_domi'] = False

        # feasible のものに non_domi の評価結果を代入する
        if len(non_domi) > 0:
            df.loc[idx, 'non_domi'] = non_domi

    def _calc_hypervolume(self, objectives, df):

        # 単目的最適化ならば 0 埋めして終了
        if len(objectives) < 2:
            df.loc[len(df) - 1, 'hypervolume'] = 0.
            return

        # 最小化問題に変換された objective values を取得
        raw_objective_values = df[self.obj_names].values
        objective_values = np.full_like(raw_objective_values, np.nan)
        for n_trial in range(len(raw_objective_values)):
            for obj_idx, (_, objective) in enumerate(objectives.items()):
                objective_values[n_trial, obj_idx] = objective.convert(raw_objective_values[n_trial, obj_idx])

        # pareto front を取得
        def get_pareto(objective_values_, with_partial=False):
            ret = None
            if with_partial:
                ret = []

            pareto_set_ = np.empty((0, len(self.obj_names)))
            for i in range(len(objective_values_)):
                target = objective_values_[i]

                if any(np.isnan(target)):
                    # infeasible な場合 pareto_set の計算に含めない
                    dominated = True

                else:
                    dominated = False
                    # TODO: Array の計算に直して高速化する
                    for j in range(len(objective_values_)):
                        compare = objective_values_[j]
                        if all(target > compare):
                            dominated = True
                            break

                if not dominated:
                    pareto_set_ = np.concatenate([pareto_set_, [target]], axis=0)

                if ret is not None:
                    ret.append(np.array(pareto_set_))

            if ret is not None:
                return pareto_set_, ret
            else:
                return pareto_set_

        def get_valid_worst_converted_objective_values(objective_values_: np.ndarray) -> np.ndarray:
            # objective_values.max(axis=0)
            ret = []
            for row in objective_values_:
                if not any(np.isnan(row)):
                    ret.append(row)
            return np.array(ret).max(axis=0)

        if (
                isinstance(self._hv_reference, np.ndarray)
                or isinstance(self._hv_reference, list)
        ):
            _buff = np.array(self._hv_reference)
            assert _buff.shape == (len(self.obj_names),)

            ref_point = np.array(
                [obj.convert(raw_value) for obj, raw_value in zip(objectives.values(), _buff)]
            )

            _buff = get_pareto(objective_values)

            pareto_set = np.empty((0, len(objectives)))
            for pareto_sol in _buff:
                if all(pareto_sol < ref_point):
                    pareto_set = np.concatenate([pareto_set, [pareto_sol]], axis=0)

            hv = compute_hypervolume(pareto_set, ref_point)
            df.loc[len(df) - 1, 'hypervolume'] = hv
            return

        elif self._hv_reference == 'dynamic-pareto':
            pareto_set, pareto_set_list = get_pareto(objective_values, with_partial=True)
            for i, partial_pareto_set in enumerate(pareto_set_list):
                # 並列計算時など Valid な解がまだ一つもない場合は pareto_set が長さ 0 になる
                # その場合 max() を取るとエラーになる
                if len(pareto_set) == 0:
                    df.loc[i, 'hypervolume'] = 0
                else:
                    ref_point = pareto_set.max(axis=0) + 1e-8
                    hv = compute_hypervolume(partial_pareto_set, ref_point)
                    df.loc[i, 'hypervolume'] = hv
            return

        elif self._hv_reference == 'dynamic-nadir':
            _, pareto_set_list = get_pareto(objective_values, with_partial=True)
            for i, partial_pareto_set in enumerate(pareto_set_list):
                # filter valid objective values only
                values = get_valid_worst_converted_objective_values(objective_values)

                # 並列計算時など Valid な解がまだ一つもない場合は長さ 0 になる
                # その場合 max() を取るとエラーになる
                if len(values) == 0:
                    df.loc[i, 'hypervolume'] = 0

                else:
                    ref_point = values.max(axis=0) + 1e-8
                    hv = compute_hypervolume(partial_pareto_set, ref_point)
                    df.loc[i, 'hypervolume'] = hv
            return

        elif self._hv_reference == 'nadir':
            pareto_set = get_pareto(objective_values)
            values = get_valid_worst_converted_objective_values(objective_values)
            if len(values) == 0:
                df.loc[len(df) - 1, 'hypervolume'] = 0
            else:
                ref_point = values.max(axis=0) + 1e-8
                hv = compute_hypervolume(pareto_set, ref_point)
                df.loc[len(df) - 1, 'hypervolume'] = hv
            return

        elif self._hv_reference == 'pareto':
            pareto_set = get_pareto(objective_values)
            if len(pareto_set) == 0:
                df.loc[len(df) - 1, 'hypervolume'] = 0
            else:
                ref_point = pareto_set.max(axis=0) + 1e-8
                hv = compute_hypervolume(pareto_set, ref_point)
                df.loc[len(df) - 1, 'hypervolume'] = hv
            return

        else:
            raise NotImplementedError(f'Invalid Hypervolume reference point calculation method: {self._hv_reference}')

    def save(self, _f=None):
        """Save csv file."""

        df = self.get_df()

        # extra_data の更新
        self.meta_columns[0] = json.dumps(self.extra_data)

        if _f is None:
            # save df with columns with prefix
            with open(self.path, 'w', encoding=self.ENCODING) as f:
                writer = csv.writer(f, delimiter=',', lineterminator="\n")
                writer.writerow(self.meta_columns)
                for i in range(self.HEADER_ROW-1):
                    writer.writerow([''] * len(self.meta_columns))
                df.to_csv(f, index=None, encoding=self.ENCODING, lineterminator='\n')
        else:  # test
            df.to_csv(_f, index=None, encoding=self.ENCODING, lineterminator='\n')

    def create_optuna_study(self):
        # create study
        kwargs = dict(
            # storage='sqlite:///' + os.path.basename(self.path) + '_dummy.db',
            sampler=None, pruner=None, study_name='dummy',
            load_if_exists=True,
        )

        if len(self.obj_names) == 1:
            kwargs.update(dict(direction='minimize'))
        else:
            kwargs.update(dict(directions=['minimize']*len(self.obj_names)))

        study = optuna.create_study(**kwargs)

        # add trial to study
        df: pd.DataFrame = self.get_df(valid_only=True)
        for i, row in df.iterrows():
            FD = optuna.distributions.FloatDistribution
            kwargs = dict(
                state=optuna.trial.TrialState.COMPLETE,
                params={k: v for k, v in zip(self.prm_names, row[self.prm_names])},
                distributions={k: FD(row[f'{k}_lower_bound'], row[f'{k}_upper_bound']) for k in self.prm_names},
                user_attrs=None,  # TODO: add constraint information by row['feasible']
            )

            # objective or objectives
            if len(self.obj_names) == 1:
                kwargs.update(dict(value=row[self.obj_names].values[0]))
            else:
                kwargs.update(dict(values=row[self.obj_names].values))
            trial = optuna.create_trial(**kwargs)
            study.add_trial(trial)

        return study


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

    def __init__(self, client, worker_address, name='entire'):
        self._future = client.submit(
            _OptimizationStatusActor,
            actor=True,
            workers=[worker_address],
            allow_other_workers=False,
        )
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


class _MonitorHostRecordActor:
    host = None
    port = None

    def set(self, host, port):
        self.host = host
        self.port = port


class MonitorHostRecord:

    def __init__(self, client, worker_name):
        self._future = client.submit(
            _MonitorHostRecordActor,
            actor=True,
            workers=(worker_name,),
            allow_other_workers=False,
        )
        self._actor = self._future.result()

    def set(self, host, port):
        self._actor.set(host, port).result()

    def get(self):
        host = self._actor.host
        port = self._actor.port
        if host is None and port is None:
            return dict()
        else:
            return dict(host=host, port=port)
