import numpy as np
from pyfemtet.opt.interface import NoFEM
from pyfemtet.opt.optimizer.scipy_optimizer.scipy_optimizer import ScipyOptimizer


def objective(_, opt: ScipyOptimizer):
    x = opt.get_variables('values')
    print("obj", (x ** 2).sum())
    return (x ** 2).sum()


def constraint(_, opt: ScipyOptimizer):
    x = opt.get_variables('values')
    print("cns-x", x)
    print("cns", np.sum(np.abs(x)))
    return np.sum(np.abs(x))


def test_scipy_optimizer():

    opt = ScipyOptimizer()
    opt.fem = NoFEM()

    opt.add_parameter('x1', -1, -1, 1)
    opt.add_parameter('x2', -1, -1, 1)
    opt.add_objective('y', objective, args=(opt,))
    # opt.add_constraint('cns', constraint, lower_bound=0.2, args=(opt,))
    opt.add_constraint('cns', constraint, lower_bound=0.2, strict=False, args=(opt,))

    opt.method = 'SLSQP'
    opt.tol = 0.1

    opt.run()


def test_scipy_optimizer_var(
        method,
        tol,
        options,
):
    pass


if __name__ == '__main__':
    test_scipy_optimizer()
