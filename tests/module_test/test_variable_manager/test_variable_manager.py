from pyfemtet.opt.problem.variable_manager._string_as_expression import _ExpressionFromString
from pyfemtet.opt.problem.variable_manager import *
import pytest
from sympy import sympify
# from numbers import Number
# from yourmodule._variable_manager import VariableManager, ExpressionFromString, NumericVariable


def test_internal_ExpressionFromString():
    import sympy
    from pyfemtet.opt.problem.variable_manager._variable_manager import Variable

    a = Variable();
    a.name = 'a';
    a.value = 1
    b = Variable();
    b.name = 'b';
    b.value = 3
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


def d_fun(a, b, arg1, *args, c='dog', kwarg=None, **kwargs) -> float:
    print(f'{a=}')
    print(f'{b=}')
    print(f'{arg1=}')
    print(f'{args=}')
    print(f'{c=}')
    print(f'{kwarg=}')
    print(f'{kwargs=}')
    if c == 'cat':
        return a + b + arg1 + args[0] + kwarg + kwargs['sample']
    else:
        assert False


def test_VariableManager():
    vm = VariableManager()

    a = NumericVariable()
    a.name = 'a'
    a.value = 1.
    b = NumericVariable()
    b.name = 'b'
    b.value = 2.
    c = CategoricalVariable()
    c.name = 'c'
    c.value = 'cat'

    d = ExpressionFromFunction()
    d.name = 'd'
    d.fun = d_fun
    d.args = (3, 4, 5)
    d.kwargs = dict(sample=6, kwarg=7)

    e = ExpressionFromString()
    e.name = 'e'
    e._expr = ExpressionFromString.InternalClass(expression_string='mean(a, b) + sqrt(3 * (a + b))')

    sympy_expr = sympify('a + b + d')
    f = ExpressionFromString()
    f.name = 'f'
    f._expr = ExpressionFromString.InternalClass(sympy_expr=sympy_expr)

    vm.variables.update(dict(
        a=a, b=b, c=c, d=d, e=e, f=f
    ))

    vm.resolve()
    vm.eval_expressions()

    print(vm.get_variables(format='dict'))
    assert vm.get_variables(format='dict') == {'a': 1.0, 'b': 2.0, 'c': 'cat', 'd': 23.0, 'e': 4.5, 'f': 26}


def test_variable_manager_expression():
    x1 = Variable()
    x1.name = 'x1'
    x1.value = 1.

    x2 = ExpressionFromFunction()
    x2.name = 'x2'
    x2.fun = lambda value_=None: value_
    x2.args = tuple()
    x2.kwargs = dict(value_=2.)

    vm = VariableManager()
    vm.variables = dict(x1=x1, x2=x2)
    vm.dependencies = dict()
    vm.resolve()
    vm.eval_expressions()


def test_eval_expressions_simple_number():
    vm = VariableManager()
    v = NumericVariable()
    v.value = 10
    vm.variables['a'] = v
    vm.resolve()
    vm.eval_expressions()
    assert vm.variables['a'].value == 10


def test_eval_expressions_simple_expression():
    vm = VariableManager()
    v = NumericVariable()
    v.value = 3
    vm.variables['a'] = v

    e = ExpressionFromString()
    e._expr = e.InternalClass('a + 2')
    vm.variables['b'] = e

    vm.resolve()
    vm.eval_expressions()
    assert vm.variables['b'].value == 5


def test_eval_expressions_min_max_mean():
    vm = VariableManager()
    v1 = NumericVariable()
    v1.value = 1
    v2 = NumericVariable()
    v2.value = 10
    v3 = NumericVariable()
    v3.value = 5

    vm.variables['a'] = v1
    vm.variables['b'] = v2
    vm.variables['c'] = v3

    # min
    e_min = ExpressionFromString()
    e_min._expr = e_min.InternalClass('min(a, b, c)')
    vm.variables['min_val'] = e_min

    # max
    e_max = ExpressionFromString()
    e_max._expr = e_max.InternalClass('max(a, b, c)')
    vm.variables['max_val'] = e_max

    # mean
    e_mean = ExpressionFromString()
    e_mean._expr = e_mean.InternalClass('mean(a, b, c)')
    vm.variables['mean_val'] = e_mean

    vm.resolve()
    vm.eval_expressions()
    assert vm.variables['min_val'].value == 1
    assert vm.variables['max_val'].value == 10
    assert vm.variables['mean_val'].value == pytest.approx(5.333333, rel=1e-5)


def test_eval_expressions_dependency_chain():
    vm = VariableManager()
    v = NumericVariable()
    v.value = 2
    vm.variables['x'] = v

    e1 = ExpressionFromString()
    e1._expr = e1.InternalClass('x + 3')
    vm.variables['y'] = e1

    e2 = ExpressionFromString()
    e2._expr = e2.InternalClass('y * 2')
    vm.variables['z'] = e2

    vm.resolve()
    vm.eval_expressions()
    assert vm.variables['y'].value == 5
    assert vm.variables['z'].value == 10


def test_eval_expressions_invalid_expression():
    vm = VariableManager()
    v = NumericVariable()
    v.value = 1
    vm.variables['a'] = v

    e = ExpressionFromString()
    # 存在しない変数を使う
    e._expr = e.InternalClass('a + b')
    vm.variables['err'] = e

    vm.resolve()
    with pytest.raises(Exception):
        vm.eval_expressions()


def test_eval_expressions_decimal_string():
    vm = VariableManager()
    v = NumericVariable()
    v.value = 1.0
    vm.variables['a'] = v

    e = ExpressionFromString()
    e._expr = e.InternalClass('a + 2.0000')
    vm.variables['b'] = e

    vm.resolve()
    vm.eval_expressions()
    assert vm.variables['b'].value == 3.0


def test_eval_expressions_expression_is_number():
    # '1' など定数の場合も正しく評価
    vm = VariableManager()
    e = ExpressionFromString()
    e._expr = e.InternalClass('1')
    vm.variables['one'] = e

    vm.resolve()
    vm.eval_expressions()
    assert vm.variables['one'].value == 1.0


def test_eval_expressions_chained_mean():
    vm = VariableManager()
    v1 = NumericVariable()
    v1.value = 2
    v2 = NumericVariable()
    v2.value = 6

    vm.variables['a'] = v1
    vm.variables['b'] = v2

    e1 = ExpressionFromString()
    e1._expr = e1.InternalClass('mean(a, b)')
    vm.variables['avg'] = e1

    e2 = ExpressionFromString()
    e2._expr = e2.InternalClass('avg * 3')
    vm.variables['triple_avg'] = e2

    vm.resolve()
    vm.eval_expressions()
    assert vm.variables['avg'].value == 4.0
    assert vm.variables['triple_avg'].value == 12.0


if __name__ == '__main__':
    # test_internal_ExpressionFromString()
    # test_VariableManager()
    test_variable_manager_expression()
