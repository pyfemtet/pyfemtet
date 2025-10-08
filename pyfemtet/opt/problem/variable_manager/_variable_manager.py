from typing import Callable, TypeAlias, Literal

import inspect
from numbers import Real  # マイナーなので型ヒントには使わず isinstance で使う
from graphlib import TopologicalSorter
import unicodedata

import numpy as np

from ._string_as_expression import _ExpressionFromString

from pyfemtet._i18n import _
from pyfemtet._util.symbol_support_for_param_name import convert_symbols

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


SupportedVariableTypes: TypeAlias = str | Real  # isinstance で使いうるので Real


class Variable:
    name: str
    value: SupportedVariableTypes
    pass_to_fem: bool
    properties: dict[str, ...]

    def __init__(self):
        # noinspection PyTypeChecker
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

                # 関数に渡す引数の初期化
                final_args = []
                final_kwargs = {}

                # 関数のシグネチャを取得
                params = inspect.signature(var.fun).parameters

                # 用意された引数をコピー
                user_args = list(var.args)
                user_kwargs = var.kwargs.copy()

                # 関数に定義されている順に組み立て
                for p in params.values():

                    # def sample(a, /, b, *c, d=None, **e):
                    #     ...
                    # a: POSITIONAL_OR_KEYWORD
                    # b: POSITIONAL_OR_KEYWORD
                    # c: VAR_POSITIONAL
                    # d: KEYWORD_ONLY
                    # e: VAR_KEYWORD

                    # 位置引数で定義されているもの
                    if p.kind <= inspect.Parameter.POSITIONAL_OR_KEYWORD:

                        # 変数である
                        if p.name in self.variables:
                            # order 順に見ているのですでに value が正しいはず
                            final_args.append(self.variables[p.name].value)

                        # ユーザーが供給する変数である
                        else:
                            # user kwargs にあるかまず確認する
                            if p.name in user_kwargs:
                                # 該当物を抜き出して args に追加
                                final_args.append(user_kwargs.pop(p.name))

                            # user args がまだある
                            elif len(user_args) > 0:
                                # 先頭を抜き出して args に追加
                                final_args.append(user_args.pop(0))

                            # pos args が空であり、kwargs の中にもない
                            else:
                                msg = []
                                for p_ in params.values():
                                    if p_.kind == inspect.Parameter.VAR_POSITIONAL:
                                        msg.append(f'*{p_.name}')
                                    elif p_.kind == inspect.Parameter.VAR_KEYWORD:
                                        msg.append(f'**{p_.name}')
                                    else:
                                        msg.append(p_.name)
                                raise RuntimeError(_(
                                        'Missing arguments! '
                                        'The arguments specified by `args`: {var_args} / '
                                        'The arguments specified by `kwargs`: {var_kwargs} / '
                                        'Required arguments: {msg}',
                                        var_args=var.args,
                                        var_kwargs=var.kwargs,
                                        msg=msg
                                ))

                    # *args である
                    elif p.kind == inspect.Parameter.VAR_POSITIONAL:

                        # ユーザー定義位置引数でないとおかしい
                        assert p.name not in self.variables, _('Extra positional argument name cannot be duplicated with a variable name.')

                        # *args なので残り全部を抜いて pos_args に入れる
                        final_args.extend(user_args)
                        user_args = []

                    # キーワード引数で定義されているもの
                    elif p.kind == inspect.Parameter.KEYWORD_ONLY:

                        # 変数である
                        if p.name in self.variables:
                            # order 順に見ているのですでに value が正しいはず
                            final_kwargs.update({p.name: self.variables[p.name].value})

                        # ユーザーが供給する変数である
                        else:
                            # user kwargs にあるかまず確認する
                            if p.name in user_kwargs:
                                # 該当物を抜き出して kwargs に追加
                                final_kwargs.update({p.name: user_kwargs.pop(p.name)})

                            # user args がまだある
                            elif len(user_args) > 0:
                                # 先頭を抜き出して args に追加
                                final_kwargs.update({p.name: user_args.pop(0)})

                            # pos args が空であり、kwargs の中にもない
                            else:
                                msg = []
                                for p_ in params.values():
                                    if p_.kind == inspect.Parameter.VAR_POSITIONAL:
                                        msg.append(f'*{p_.name}')
                                    elif p_.kind == inspect.Parameter.VAR_KEYWORD:
                                        msg.append(f'**{p_.name}')
                                    else:
                                        msg.append(p_.name)
                                raise RuntimeError(_(
                                        'Missing arguments! '
                                        'The arguments specified by `args`: {var_args} / '
                                        'The arguments specified by `kwargs`: {var_kwargs} / '
                                        'Required arguments: {msg}',
                                        var_args=var.args,
                                        var_kwargs=var.kwargs,
                                        msg=msg
                                ))

                    # **kwargs である
                    elif p.kind == inspect.Parameter.VAR_KEYWORD:
                        # 変数であってはいけない
                        assert p.name not in self.variables, _('Extra keyword argument name cannot be duplicated with a variable name.')

                        # 残りの kwargs を移す
                        final_kwargs.update(user_kwargs)
                        user_kwargs = {}

                    else:
                        raise NotImplementedError(f'Unknown argument type: {p.kind=}')

                # fun を実行する
                var.value = var.fun(*final_args, **final_kwargs)

            # string expression の場合
            elif isinstance(var, ExpressionFromString):

                # dependency_values を作る
                dependency: set[str] = var._expr.dependency

                # order 順に見ているので value は正しいはず
                dependency_values: dict[str, SupportedVariableTypes] = {
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
            filter: (Literal['pass_to_fem', 'parameter']
                     | tuple[Literal['pass_to_fem', 'parameter']]
                     | None) = None,  # 'pass_to_fem' and 'parameter' (OR filter)
            format: Literal['dict', 'values', 'raw'] | None = None,  # Defaults to 'raw'
    ) -> (
        dict[str, Variable]
        | dict[str, Parameter]
        | dict[str, SupportedVariableTypes]
        | np.ndarray
    ):
        # 参照を返す仕様を dynamic bounds で利用

        raw = {}

        for name, var in self.variables.items():

            if filter is not None:
                if 'pass_to_fem' in filter:
                    if var.pass_to_fem:
                        name = var.properties.get('original_name', name)
                        raw.update({name: var})

                if 'parameter' in filter:
                    if isinstance(var, Parameter):
                        name = var.properties.get('original_name', name)
                        raw.update({name: var})

            else:
                name = var.properties.get('original_name', name)
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

    def set_variable(self, variable: Variable):
        original_name = variable.name
        variable.properties.update(
            {'original_name': original_name}
        )
        variable.name = convert_symbols(variable.name)
        if not unicodedata.is_normalized('NFKC', variable.name):
            variable.name = unicodedata.normalize('NFKC', variable.name)
        self.variables.update({variable.name: variable})
