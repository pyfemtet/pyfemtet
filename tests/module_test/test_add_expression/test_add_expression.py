import sympy
from pyfemtet.opt.problem.variable_manager import *
from pyfemtet.opt.optimizer import AbstractOptimizer


# noinspection PyUnusedLocal
def expression1(prm1, prm2, arg1, *args, kwarg2=None, **kwargs):
    # print(prm1, prm2)
    # print(arg1, args, kwarg2, kwargs)
    return prm1 + 1


def test_add_expression():
    opt = AbstractOptimizer()

    x = sympy.Symbol('x')
    opt.add_expression_sympy(
        name='exp3',
        sympy_expr=sympy.integrate(
            x**3,
            (x, sympy.Symbol('prm1'), sympy.Symbol('exp1'))
        )
    )

    opt.add_parameter(
        name='prm1',
        initial_value=0,
        upper_bound=0,
        lower_bound=1,
    )

    opt.add_categorical_parameter(
        name='prm2',
        initial_value='a',
        choices=['a', 'b']
    )

    opt.add_expression_string('exp2', 'prm1 + exp1')

    opt.add_expression(
        name='exp1',
        fun=expression1,
        args=('this is arg1', 'this is other argument'),
        kwargs=dict(kwarg2='this is kwarg2', other_kwarg='this is remaining kwarg.'),
    )

    opt.variable_manager.resolve()
    opt.variable_manager.eval_expressions()

    variables: dict[str, Variable] = opt.get_variables()

    print(variables)
    assert variables == {'exp3': 0.25, 'prm1': 0, 'prm2': 'a', 'exp2': 1.0, 'exp1': 1}


if __name__ == '__main__':
    test_add_expression()
