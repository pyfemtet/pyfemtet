from numbers import Number

from sympy import sympify
from sympy.core.sympify import SympifyError
from sympy import Min, Max, Add, Symbol, Expr, Basic  # TODO: Add sqrt, pow

__all__ = [
    '_ExpressionFromString', 'InvalidExpression', 'SympifyError'
]


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

    def __init__(self, expression_string: str | float = None, sympy_expr: Expr = None):
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

        """

        # check
        assert not (expression_string is None and sympy_expr is None)

        if sympy_expr is not None:
            assert expression_string is None
            self._sympy_expr = sympy_expr
            self._expr_str = str(sympy_expr)

        else:
            assert expression_string is not None
            self._expr_str: str = str(expression_string)

            # max(name1, name2) など関数を入れる際に問題になるので
            # 下記の仕様は廃止、使い方として数値桁区切り , を入れてはいけない
            # # sympify 時に tuple 扱いになるので , を置き換える
            # # 日本人が数値に , を使うとき Python では _ を意味する
            # # expression に _ が入っていても構わない
            # tmp_expr = str(self._expr_str).replace(',', '_')
            self._sympy_expr = sympify(self._expr_str, locals=get_valid_functions())

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

        re_sympy_expr = sympify(
            self.expression_string,
            locals=get_valid_functions(dependency_values),
        )

        evaluated_sympy_obj = re_sympy_expr.subs(dependency_values)
        try:
            evaluated_value = float(evaluated_sympy_obj)
        except (ValueError, TypeError) as e:
            raise type(e)(*e.args, f'{evaluated_sympy_obj=} cannot convert to float.') from None

        return evaluated_value


def debug():
    import sympy
    from pyfemtet.opt.variable_manager.variable_manager import Variable

    a = Variable(); a.name = 'a'; a.value = 1
    b = Variable(); b.name = 'b'; b.value = 3
    c = _ExpressionFromString('a + b')
    print(c.dependency)
    assert c.dependency == {a.name, b.name}
    print(c.eval({a.name: a.value, b.name: b.value}))
    assert c.eval({a.name: a.value, b.name: b.value}) == 4

    x = sympy.Symbol('x')
    a_sympy = sympy.Symbol(a.name)
    b_sympy = sympy.Symbol(b.name)
    f = x ** 3
    d = _ExpressionFromString(
        sympy_expr=sympy.integrate(
            f,
            (x, a_sympy, b_sympy))
    )
    print(d.dependency)
    assert d.dependency == {a.name, b.name}
    print(d.eval({a.name: a.value, b.name: b.value}))
    assert d.eval({a.name: a.value, b.name: b.value}) == 20

    e = _ExpressionFromString('sqrt(a) + pow(a, 2) + max(a, b) + mean(b, d) + d')
    print(e.dependency)
    assert e.dependency == {a.name, b.name, 'd'}
    print(e.eval({a.name: a.value, b.name: b.value, 'd': d.eval({a.name: a.value, b.name: b.value})}))
    assert e.eval({a.name: a.value, b.name: b.value, 'd': d.eval({a.name: a.value, b.name: b.value})}) == 36.5


if __name__ == '__main__':
    debug()
