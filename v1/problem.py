from __future__ import annotations

from typing import Callable, TYPE_CHECKING

try:
    # noinspection PyUnresolvedReferences
    from pythoncom import CoInitialize, CoUninitialize
    from win32com.client import Dispatch, Constants, constants
except ModuleNotFoundError:
    CoInitialize = lambda: None
    CoUninitialize = lambda: None
    Dispatch = type('NoDispatch', (object,), {})
    Constants = type('NoConstants', (object,), {})
    constants = Constants()

from v1.variable_manager import *
from v1.helper import *
if TYPE_CHECKING:
    from v1.optimizer import AbstractOptimizer

__all__ = [
    'TrialInput',
    'TrialOutput',
    'TrialConstraintOutput',
    'Function',
    'Functions',
    'Objective',
    'ObjectiveResult',
    'Objectives',
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

    def eval(self) -> float:
        return float(self.fun(*self.args, **self.kwargs))


class Functions(dict[str, Function]): ...


class Objective(Function):
    direction: str | float

    @staticmethod
    def _convert(value, direction):

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


class ObjectiveResult:

    def __init__(self, obj: Objective, obj_value: float = None):

        self.value: float = obj_value if obj_value is not None else obj.eval()
        self.direction: str | float = obj.direction

    def __repr__(self):
        return str(self.value)


class Objectives(dict[str, Objective]): ...


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

    def __init__(self, cns: Constraint, cns_value: float = None):

        self.value: float = cns_value if cns_value is not None else cns.eval()
        self.lower_bound: float | None = cns.lower_bound
        self.upper_bound: float | None = cns.upper_bound
        self.hard: bool = cns.hard

    def __repr__(self):
        return str(self.value)

    def calc_violation(self) -> dict[str, float]:
        value = self.value
        out = {}
        if self.lower_bound is not None:
            out.update({'lower_bound': self.lower_bound - value})
        if self.upper_bound is not None:
            out.update({'upper_bound': value - self.upper_bound})
        return out

    def check_violation(self) -> str | None:
        violation = self.calc_violation()
        for l_or_u, value in violation.items():
            if value > 0:
                return l_or_u
        return None


class Constraints(dict[str, Constraint]): ...


class Fidelity: ...


SubSampling = int


TrialInput = dict[str, Variable]
TrialOutput = dict[str, ObjectiveResult]
TrialConstraintOutput = dict[str, ConstraintResult]
