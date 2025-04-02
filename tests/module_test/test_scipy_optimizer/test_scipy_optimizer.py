import numpy as np
from pyfemtet.opt.interface import NoFEM
from pyfemtet.opt.optimizer.scipy_optimizer.scipy_optimizer import ScipyOptimizer


def objective(_, opt: ScipyOptimizer):
    x = opt.get_variables('values')
    return (x ** 2).sum()


def constraint1(_, opt: ScipyOptimizer):
    x = opt.get_variables('values')
    return np.sum(np.abs(x))


def constraint2(_, opt: ScipyOptimizer):
    x = opt.get_variables('values')
    return x[0]


def test_scipy_optimizer():

    opt = ScipyOptimizer()
    opt.fem = NoFEM()

    opt.add_parameter('x1', -1, -1, 1)
    opt.add_parameter('x2', -1, -1, 1)
    opt.add_objective('y', objective, args=(opt,))
    opt.add_constraint('cns1', constraint1, lower_bound=0.2, strict=False, args=(opt,))
    opt.add_constraint('cns2', constraint2, lower_bound=0, strict=False, args=(opt,))

    opt.method = 'SLSQP'
    opt.options = dict(
        ftol=0.1,
        eps=0.1,
    )

    opt.run()


def scipy_optimizer_var(
        method='SLSQP',
        options=None,
        with_constraint=None,
        hard_constraint=None,
):

    options = options or {
        'slsqp': dict(ftol=0.1, eps=0.1),
        'nelder-mead': dict(fatol=0.05, xatol=0.1),
    }.get(method.lower(), None)

    with_constraint = with_constraint or {
        'slsqp': True,
    }.get(method.lower(), False)

    hard_constraint = hard_constraint or {
        'slsqp': True
    }.get(method, False)

    opt = ScipyOptimizer()
    opt.fem = NoFEM()

    opt.add_parameter('x1', -1, -1, 1)
    opt.add_parameter('x2', -1, -1, 1)
    opt.add_objective('y', objective, args=(opt,))
    if with_constraint:
        opt.add_constraint('cns1', constraint1, lower_bound=0.2, strict=hard_constraint, args=(opt,))
        opt.add_constraint('cns2', constraint2, lower_bound=0, strict=hard_constraint, args=(opt,))

    opt.method = method
    opt.options = options

    opt.run()


def test_scipy_optimizer_1():
    scipy_optimizer_var('Nelder-Mead')
    scipy_optimizer_var('BFGS')


if __name__ == '__main__':
    test_scipy_optimizer()
    test_scipy_optimizer_1()
