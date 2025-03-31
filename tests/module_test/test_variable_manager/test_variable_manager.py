from pyfemtet.opt.variable_manager.string_as_expression import _ExpressionFromString
from pyfemtet.opt.variable_manager import *


def test_internal_ExpressionFromString():
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


def d_fun(a, b, arg1, *args, c='dog', kwarg=None, **kwargs) -> float:
    if c == 'cat':
        return a + b + arg1 + args[0] + kwarg + kwargs['sample']
    else:
        assert False


def test_VariableManager():

    vm = VariableManager()

    a = NumericVariable(); a.name = 'a'; a.value = 1.
    b = NumericVariable(); b.name = 'b'; b.value = 2.
    c = CategoricalVariable(); c.name = 'c'; c.value = 'cat'

    d = NumericExpressionFromFunction()
    d.name = 'd'
    d.fun = d_fun
    d.args = (3, 4, 5)
    d.kwargs = dict(sample=6, kwarg=7)

    e = ExpressionFromString()
    e.name = 'e'
    e._expr = _ExpressionFromString('mean(a, b) + sqrt(3 * (a + b))')

    vm.variables.update(dict(
        a=a, b=b, c=c, d=d, e=e
    ))

    vm.resolve()
    vm.eval_expressions()

    print(vm.get_variables(format='dict'))
    assert vm.get_variables(format='dict') == {'a': 1.0, 'b': 2.0, 'c': 'cat', 'd': 23.0, 'e': 4.5}


if __name__ == '__main__':
    # test_internal_ExpressionFromString()
    test_VariableManager()
