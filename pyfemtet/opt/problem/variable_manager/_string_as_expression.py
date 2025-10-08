import re
from numbers import Number

from sympy import sympify
from sympy.core.sympify import SympifyError
from sympy import Min, Max, Add, Symbol, Expr, Basic  # TODO: Add sqrt, pow

from pyfemtet._util.symbol_support_for_param_name import convert_symbols

__all__ = [
    '_ExpressionFromString', 'InvalidExpression', 'SympifyError'
]


def _convert(expr_str):
    """ \" で囲まれた部分があれば記号を置き換える """

    def repl(m: re.Match) -> str:
        inner = m.group(1)
        return f'"{convert_symbols(inner)}"'

    # 非貪欲で " 内だけを拾う
    expr_str = re.sub(r'"(.+?)"', repl, expr_str)
    # " を消す
    expr_str = expr_str.replace('"', '')

    return expr_str


class InvalidExpression(Exception):
    pass


def get_valid_functions(values_may_be_used_in_mean: dict[str, Number] = None):
    v = values_may_be_used_in_mean or {}
    return {
        'mean': lambda *args: Add(*args).subs(v) / len(args),
        'max': Max,
        'min': Min,
        'S': Symbol('S')
    }


class _ExpressionFromString:
    _expr_str: str
    _sympy_expr: Expr

    def __init__(
            self,
            expression_string: str | Number = None,
            sympy_expr: Expr = None,
    ):
        """
        Raises:
            SympifyError: Sympy が認識できない場合
            InvalidExpression: Sympy は認識できるが PyFemtet で想定する型ではない場合（tuple など）

        Examples:
            e = Expression('1')
            e.expr  # '1'
            e.value  # 1.0

            e = Expression(1)
            e.expr  # '1'
            e.value  # 1.0

            e = Expression('a')
            e.expr  # 'a'
            e.value  # ValueError

            e = Expression('1/2')
            e.expr  # '1/2'
            e.value  # 0.5

            e = Expression('1.0000')
            e.expr  # '1.0'
            e.value  # 1.0

            # To use "-", "@" and "." for name
            e = Expression('"Sample-1.0@Part1" * 2')
            e.expr  # 'Sample_hyphen_1_dot_0_at_Part1 * 2'

        """

        # check
        assert not (expression_string is None and sympy_expr is None)

        if sympy_expr is not None:
            assert expression_string is None
            self._sympy_expr = sympy_expr
            self._expr_str = str(sympy_expr)

        else:
            assert expression_string is not None
            self._expr_str: str = _convert(str(expression_string))

            # max(name1, name2) など関数を入れる際に問題になるので
            # 下記の仕様は廃止、使い方として数値桁区切り , を入れてはいけない
            # # sympify 時に tuple 扱いになるので , を置き換える
            # # 日本人が数値に , を使うとき Python では _ を意味する
            # # expression に _ が入っていても構わない
            # tmp_expr = str(self._expr_str).replace(',', '_')
            self._sympy_expr = sympify(self._expr_str, locals=get_valid_functions())  # noqa

        if not isinstance(self._sympy_expr, Basic):
            raise InvalidExpression(f'{self._expr_str} は数式ではありません。')

    @property
    def dependency(self) -> set[str]:
        s: Symbol
        return {s.name for s in self._sympy_expr.free_symbols}

    def is_number(self) -> bool:
        return self._sympy_expr.is_number

    def is_expression(self) -> bool:
        return not self.is_number()

    @property
    def expression_string(self) -> str:
        return self._expr_str

    def eval(self, dependency_values: dict[str, Number]):

        # 型チェック
        assert all([isinstance(value, Number) for value
                    in dependency_values.values()]), \
            'ExpressionFromString では数値変数のみをサポートしています。'

        re_sympy_expr = sympify(  # noqa
            self.expression_string,
            locals=get_valid_functions(dependency_values),
        )

        evaluated_sympy_obj = re_sympy_expr.subs(dependency_values)  # noqa
        try:
            evaluated_value = float(evaluated_sympy_obj)
        except (ValueError, TypeError) as e:
            raise type(e)(*e.args, f'{evaluated_sympy_obj=} cannot convert to float.') from None

        return evaluated_value
