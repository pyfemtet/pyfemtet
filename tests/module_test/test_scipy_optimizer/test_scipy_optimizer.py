import os
from time import sleep
import numpy as np
from pyfemtet.opt.interface import NoFEM
from pyfemtet.opt.optimizer.scipy_optimizer._scipy_optimizer import ScipyOptimizer


def objective(_, opt: ScipyOptimizer):
    x = opt.get_variables('values')
    return (x ** 2).sum()


def constraint1(_, opt: ScipyOptimizer):
    x = opt.get_variables('values')
    return np.sum(np.abs(x)) * 1000


def constraint2(_, opt: ScipyOptimizer):
    x = opt.get_variables('values')
    return x[0]


def test_scipy_optimizer():
    opt = ScipyOptimizer()
    opt.fem = NoFEM()
    if os.path.isfile('pyfemtet_tmp_scipy.csv'):
        os.remove('pyfemtet_tmp_scipy.csv')
    opt.history.path = 'pyfemtet_tmp_scipy.csv'

    opt.add_parameter('x1', 1, -1, 1)
    opt.add_parameter('x2', -1, -1, 1)
    opt.add_objective('y', objective, args=(opt,))
    opt.add_constraint('cns1', constraint1, lower_bound=200, strict=True, args=(opt,))
    opt.add_constraint('cns2', constraint2, lower_bound=0, strict=True, args=(opt,))

    opt.method = 'SLSQP'
    opt.options = dict(
        ftol=0.1,
        eps=0.001,
    )
    opt.constraint_enhancement += opt.options.get('eps', 0.) * 1000

    # optimize_acqf と違って solve ごとに violation を見ているので
    # 例えば SLSQP で jac の計算のために eps の範囲内で x が変動することで
    # 拘束違反の値が提案されることを scaling によって避けることはできない
    # optimize_acqf の内部の minimize の場合は対象が FEM ではなく
    # ACQF なので、サブ最適化問題の中で拘束違反をチェックする機構は
    # scipy.optimize に任せており、そこでは jac の計算のための
    # x の変動の範囲内で拘束違反することを許容している。
    # 最適化対象が FEM ではなく ACQF なので最適化結果が厳密に拘束違反でさえ
    # なければよく、そういう場合は scaling で対処が可能である。
    # opt.constraint_enhancement = 0
    # opt.constraint_scaling = 1e6

    opt.run()

    sleep(1)


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
    if os.path.isfile('pyfemtet_tmp_scipy.csv'):
        os.remove('pyfemtet_tmp_scipy.csv')
    opt.history.path = 'pyfemtet_tmp_scipy.csv'

    opt.add_parameter('x1', -1, -1, 1)
    opt.add_parameter('x2', -1, -1, 1)
    opt.add_objective('y', objective, args=(opt,))
    if with_constraint:
        opt.add_constraint('cns1', constraint1, lower_bound=200, strict=hard_constraint, args=(opt,))
        opt.add_constraint('cns2', constraint2, lower_bound=0, strict=hard_constraint, args=(opt,))

    opt.method = method
    opt.options = options

    opt.run()

    sleep(1)


def test_scipy_optimizer_1():
    scipy_optimizer_var('Nelder-Mead')
    scipy_optimizer_var('BFGS')


if __name__ == '__main__':
    test_scipy_optimizer()
    # test_scipy_optimizer_1()
