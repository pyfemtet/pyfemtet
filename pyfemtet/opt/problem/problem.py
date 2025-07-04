from __future__ import annotations

from typing import Callable, TYPE_CHECKING, TypeAlias, Sequence

try:
    # noinspection PyUnresolvedReferences
    from pythoncom import CoInitialize, CoUninitialize
    from win32com.client import Dispatch, Constants, constants
except ModuleNotFoundError:
    # noinspection PyPep8Naming
    def CoInitialize(): ...
    # noinspection PyPep8Naming
    def CoUninitialize(): ...
    Dispatch = type('NoDispatch', (object,), {})
    Constants = type('NoConstants', (object,), {})
    constants = Constants()

from pyfemtet._i18n import _
from pyfemtet._util.helper import *

from .variable_manager import *

if TYPE_CHECKING:
    from pyfemtet.opt.optimizer import AbstractOptimizer
    from pyfemtet.opt.interface import AbstractFEMInterface

__all__ = [
    'TrialInput',
    'TrialOutput',
    'TrialConstraintOutput',
    'TrialFunctionOutput',
    'Function',
    'FunctionResult',
    'Functions',
    'Objective',
    'ObjectiveResult',
    'Objectives',
    'ObjectivesFunc',
    'Constraint',
    'ConstraintResult',
    'Constraints',
    'Fidelity',
    'SubSampling',
    'MAIN_FIDELITY_NAME',
]

MAIN_FIDELITY_NAME = ''


class Function:
    _fun: Callable[..., float]
    args: tuple
    kwargs: dict

    @property
    def fun(self) -> Callable[..., float]:
        self._ScapeGoat.restore_constants(self._fun)
        return self._fun

    @fun.setter
    def fun(self, f: Callable[..., float]):
        self._fun = f

    def __getstate__(self):
        """Pickle 時に _fun が参照する constants を _ScapeGoat にする"""
        state = self.__dict__
        if '_fun' in state:
            self._ScapeGoat.remove_constants(state['_fun'])
        return state

    def __setstate__(self, state):
        """Pickle 時に _fun が参照する _ScapeGoat を constants にする"""
        CoInitialize()
        if '_fun' in state:
            self._ScapeGoat.restore_constants(state['_fun'])
        self.__dict__.update(state)

    class _ScapeGoat:

        @classmethod
        def restore_constants(cls, f: ...):
            """f の存在する global スコープの _Scapegoat 変数を constants に変更"""
            if not hasattr(f, '__globals__'):
                return

            for name, var in f.__globals__.items():
                if isinstance(var, cls):
                    # try 不要
                    # fun の定義がこのファイル上にある場合、つまりデバッグ時のみ
                    # remove_constants がこのスコープの constants を消すので
                    # constants を再インポートする必要がある
                    from win32com.client import constants
                    f.__globals__[name] = constants

        @classmethod
        def remove_constants(cls, f: ...):
            """f の存在する global スコープの Constants 変数を _Scapegoat に変更"""

            if not hasattr(f, '__globals__'):
                return

            for name, var in f.__globals__.items():
                if isinstance(var, Constants):
                    f.__globals__[name] = cls()

    def eval(self, fem: AbstractFEMInterface) -> float:
        return float(self.fun(fem.object_pass_to_fun, *self.args, **self.kwargs))


class Functions(dict[str, Function]):
    pass


class Objective(Function):
    direction: str | float

    @staticmethod
    def _convert(value, direction) -> float:

        direction: float | str | None = float_(direction)

        if value is None or direction is None:
            value_as_minimize = float('nan')

        elif isinstance(direction, str):
            if direction.lower() == 'minimize':
                value_as_minimize = value
            elif direction.lower() == 'maximize':
                value_as_minimize = -value
            else:
                raise NotImplementedError

        else:
            # if value is nan, return nan
            value_as_minimize = (value - direction) ** 2

        return value_as_minimize

    def convert(self, value) -> float:
        return self._convert(value, self.direction)


class FunctionResult:

    def __init__(self, func: Function, fem: AbstractFEMInterface):
        self.value: float = func.eval(fem)


class ObjectiveResult:

    def __init__(self, obj: Objective, fem: AbstractFEMInterface, obj_value: float = None):

        self.value: float = obj_value if obj_value is not None else obj.eval(fem)
        self.direction: str | float = obj.direction

    def __repr__(self):
        return str(self.value)


class ObjectivesFunc:
    """複数の値を返す関数を単一の float を返す関数に分割する。"""

    def __init__(self, fun: Callable[..., Sequence[float]], n_return: int):
        # Optimizer に追加される数と一致することを保証したいので
        # n_returns が必要
        self._called: list[bool] | None = None
        self._values: list[bool] | None = None
        self.fun: Callable[..., Sequence[float]] = fun
        self.n_return: int = n_return

    def get_fun_that_returns_ith_value(self, i):

        if i not in range(self.n_return):
            raise IndexError(
                _(
                    en_message='Index {i} is over n_return={n_return}.',
                    jp_message='インデックス {i} は n_return={n_return} を超えています。',
                    i=i, n_return=self.n_return
                )
            )

        # iter として提供する callable オブジェクト
        # self の情報にもアクセスする必要があり
        # それぞれが iter された時点での i 番目という
        # 情報も必要なのでこのスコープで定義する必要がある
        # noinspection PyMethodParameters
        class NthFunc:

            def __init__(self_, i_):
                # 何番目の要素であるかを保持
                self_.i = i_

            def __call__(self_, *args, **kwargs) -> float:
                # 何番目の要素であるか
                i_ = self_.i

                # 一度も呼ばれていなければ評価する
                if self._called is None:
                    self._values = self.fun(*args, **kwargs)
                    self._called = [False for __ in self._values]

                    assert len(self._values) == self.n_return, _(
                        en_message='The number of return values of {fun_name} is {n_values}. '
                                   'This is inconsistent with the specified n_return; {n_return}.',
                        jp_message='{fun_name} の実行結果の値の数は {n_values} でした。'
                                   'これは指定された n_return={n_return} と一致しません。',
                        fun_name=self.fun.__name__,
                        n_values=len(self._values),
                        n_return=self.n_return,
                    )

                # i_ が呼ばれたのでフラグを立てる
                self._called[i_] = True
                value = self._values[i_]

                # すべてのフラグが立ったならクリアする
                if all(self._called):
                    self._called = None
                    self._values = None

                # 値を返す
                return value

            # noinspection PyPropertyDefinition
            @property
            def __globals__(self_):
                # ScapeGoat 実装への対処
                if hasattr(self.fun, '__globals__'):
                    return self.fun.__globals__
                else:
                    return {}

        # N 番目の値を返す関数を返す
        f = NthFunc(i)

        return f


class Objectives(dict[str, Objective]):
    pass


class Constraint(Function):
    lower_bound: float | None
    upper_bound: float | None
    hard: bool
    _using_fem: bool | None = None
    _opt: AbstractOptimizer

    @property
    def using_fem(self) -> bool:
        if self._using_fem is None:
            return self._opt.fem._check_using_fem(self.fun)
        else:
            return self._using_fem

    @using_fem.setter
    def using_fem(self, value: bool | None):
        self._using_fem = value


class ConstraintResult:

    def __init__(
            self,
            cns: Constraint,
            fem: AbstractFEMInterface,
            cns_value: float = None,
            constraint_enhancement: float = None,  # offset により scipy.minimize が拘束違反の解を返す問題を回避する
            constraint_scaling: float = None,  # scaling により scipy.minimize が拘束違反の解を返す問題を回避する
    ):

        self.value: float = cns_value if cns_value is not None else cns.eval(fem)
        self.lower_bound: float | None = cns.lower_bound
        self.upper_bound: float | None = cns.upper_bound
        self.hard: bool = cns.hard
        self.ce = constraint_enhancement or 0.
        self.cs = constraint_scaling or 1.

    def __repr__(self):
        return str(self.value)

    def calc_violation(self) -> dict[str, float]:
        value = self.value
        out = {}
        if self.lower_bound is not None:
            out.update({'lower_bound': self.cs * (self.lower_bound - value) + self.ce})
        if self.upper_bound is not None:
            out.update({'upper_bound': self.cs * (value - self.upper_bound) + self.ce})
        return out

    def check_violation(self) -> str | None:
        violation = self.calc_violation()
        for l_or_u, value in violation.items():
            if value > 0:
                return l_or_u
        return None


class Constraints(dict[str, Constraint]):
    pass


Fidelity: TypeAlias = float | str | None


SubSampling: TypeAlias = int


TrialInput: TypeAlias = dict[str, Variable]
TrialOutput: TypeAlias = dict[str, ObjectiveResult]
TrialConstraintOutput: TypeAlias = dict[str, ConstraintResult]
TrialFunctionOutput: TypeAlias = dict[str, FunctionResult]
