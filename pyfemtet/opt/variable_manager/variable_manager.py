from typing import Callable

import inspect
from graphlib import TopologicalSorter

import numpy as np

from .string_as_expression import _ExpressionFromString

from pyfemtet._i18n import _

__all__ = [
    'SupportedVariableTypes',
    'Parameter',
    'Variable',
    'NumericVariable',
    'NumericParameter',
    'CategoricalVariable',
    'CategoricalParameter',
    'Expression',
    'ExpressionFromFunction',
    # 'NumericExpressionFromFunction',
    # 'CategoricalExpressionFromFunction',
    'ExpressionFromString',
    'VariableManager',
]


SupportedVariableTypes = str | float


class Variable:
    name: str
    value: ...
    pass_to_fem: bool
    properties: dict[str, ...]

    def __init__(self):
        self.value = None
        self.properties = {}

    def __repr__(self):
        return str(self.value)


class Parameter(Variable):
    pass


class NumericVariable(Variable):
    value: float


class NumericParameter(NumericVariable, Parameter):
    value: float
    lower_bound: float | None
    upper_bound: float | None
    step: float | None


class CategoricalVariable(Variable):
    value: str


class CategoricalParameter(CategoricalVariable, Parameter):
    choices: list[str]


class Expression(Variable):
    pass


class ExpressionFromFunction(Expression):
    fun: Callable
    args: tuple | None
    kwargs: dict | None


class NumericExpressionFromFunction(ExpressionFromFunction, NumericVariable):
    fun: Callable[..., float]


class CategoricalExpressionFromFunction(ExpressionFromFunction, CategoricalVariable):
    fun: Callable[..., str]


class ExpressionFromString(Expression, NumericVariable):
    _expr: _ExpressionFromString
    InternalClass: type = _ExpressionFromString


class VariableManager:

    variables: dict[str, Variable]
    dependencies: dict[str, set[str]]
    evaluation_order: list[str]  # 評価する順番

    def __init__(self):
        super().__init__()
        self.variables = dict()
        self.dependencies = dict()

    @staticmethod
    def _calc_dependencies(variables: dict[str, Variable]) -> dict[str, set[str]]:

        dependencies: dict[str, set[str]] = dict()
        all_var_names: set[str] = set(variables.keys())

        var_name: str
        for var_name, var in variables.items():

            # Expression ならば、fun の引数のうち
            # variables の名前であるものを dependency とする
            if isinstance(var, ExpressionFromFunction):
                arg_names: set[str] = set(inspect.signature(var.fun).parameters)
                dep_var_names: set[str] = all_var_names & arg_names
                dependency = dep_var_names

            # ExpressionString であれば、
            # 使われている変数を解析する
            elif isinstance(var, ExpressionFromString):
                dependency = var._expr.dependency

            else:
                assert isinstance(var, Variable)
                dependency = {}

            dependencies.update({var_name: dependency})

        return dependencies

    @staticmethod
    def _calc_evaluation_order(dependencies: dict[str, set[str]]) -> list[str]:
        ts = TopologicalSorter(dependencies)
        return list(ts.static_order())

    def resolve(self):
        self.dependencies = self._calc_dependencies(self.variables)
        self.evaluation_order = self._calc_evaluation_order(self.dependencies)

    def eval_expressions(self):

        for var_name in self.evaluation_order:

            var = self.variables[var_name]

            # fun を持つ場合
            if isinstance(var, ExpressionFromFunction):

                # pop するので list をコピーしておく
                user_def_args = list(var.args)

                # 位置引数を揃え、残りをキーワード引数にする
                pos_args = []
                kw_args = var.kwargs or {}

                # 引数順に調べる
                required_arg_names = inspect.signature(var.fun).parameters.values()
                for p in required_arg_names:

                    # 位置引数であ（りう）る
                    if p.kind <= inspect.Parameter.POSITIONAL_OR_KEYWORD:

                        # 変数である
                        if p.name in self.variables:
                            # order 順に見ているのですでに value が正しいはず
                            pos_args.append(self.variables[p.name].value)

                        # ユーザー定義位置引数である
                        else:
                            try:
                                pos_args.append(user_def_args.pop(0))
                            except IndexError as e:  # pop from empty list
                                msg = []
                                for p_ in required_arg_names:
                                    if p_.kind == inspect.Parameter.VAR_POSITIONAL:
                                        msg.append(f'*{p_.name}')
                                    elif p_.kind == inspect.Parameter.VAR_KEYWORD:
                                        msg.append(f'**{p_.name}')
                                    else:
                                        msg.append(p_.name)
                                raise type(e)(
                                    *e.args,
                                    _(
                                        'Missing arguments! '
                                        'The arguments specified by `args`: {var_args} / '
                                        'The arguments specified by `kwargs`: {var_kwargs} / '
                                        'Required arguments: {msg}',
                                        var_args=var.args,
                                        var_kwargs=var.kwargs,
                                        msg=msg
                                    ),
                                ) from None

                    # *args である
                    elif p.kind == inspect.Parameter.VAR_POSITIONAL:

                        # ユーザー定義位置引数でないとおかしい
                        assert p.name not in self.variables, _('Extra positional argument name cannot be duplicated with a variable name.')

                        # *args なので残り全部を pos_args に入れる
                        pos_args.extend(user_def_args)

                    # キーワード引数である
                    elif p.kind == inspect.Parameter.KEYWORD_ONLY:

                        # 変数である
                        if p.name in self.variables:
                            # order 順に見ているのですでに value が正しいはず
                            kw_args.update({p.name: self.variables[p.name].value})

                    # **kwargs である
                    else:

                        # kw_args にユーザー定義キーワード引数を入れているので何もしなくてよい
                        assert p.name not in self.variables, _('Extra keyword argument name cannot be duplicated with a variable name.')

                # fun を実行する
                var.value = var.fun(*pos_args, **kw_args)

            # string expression の場合
            elif isinstance(var, ExpressionFromString):

                # dependency_values を作る
                dependency: set[str] = var._expr.dependency

                # order 順に見ているので value は正しいはず
                dependency_values: dict[str, float] = {
                    dep_name: self.variables[dep_name].value
                    for dep_name in dependency
                }

                # 計算
                var.value = var._expr.eval(dependency_values)

            # その他 = Expression ではない場合
            else:
                # optimizer によって直接 update されている
                # はずなので何もしなくてよい
                pass

    # noinspection PyShadowingBuiltins
    def get_variables(
            self,
            *,
            filter: str | tuple | None = None,  # 'pass_to_fem' and 'parameter' (OR filter)
            format: str = None,  # None, 'dict' and 'values'
    ) -> (
        dict[str, Variable]
        | dict[str, Parameter]
        | dict[str, SupportedVariableTypes]
        | np.ndarray
    ):

        raw = {}

        for name, var in self.variables.items():

            if filter is not None:
                if 'pass_to_fem' in filter:
                    if var.pass_to_fem:
                        raw.update({name: var})

                if 'parameter' in filter:
                    if isinstance(var, Parameter):
                        raw.update({name: var})

            else:
                raw.update({name: var})

        if format is None:
            return raw

        elif format == 'raw':
            return raw

        elif format == 'dict':
            return {name: var.value for name, var in raw.items()}

        elif format == 'values':
            return np.array([var.value for var in raw.values()])

        else:
            raise NotImplementedError(
                _(
                    'invalid format {format} is passed to '
                    'VariableManager.get_variables(). '
                    'Valid formats are one of (`raw`, `dict`, `values`).',
                    format=format,
                )
            )
