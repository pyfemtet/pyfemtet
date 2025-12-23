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


def test_scipy_n_trials():
    opt = ScipyOptimizer('SLSQP')
    opt.fem = NoFEM()
    opt.add_parameter('x1', -1, -1, 1)
    opt.add_parameter('x2', -1, -1, 1)
    opt.add_objective('norm', lambda _, opt_: np.linalg.norm(opt_.get_variables(format='values')), args=(opt,))
    opt.n_trials = 3
    opt.run()
    df = opt.history.get_df()
    assert len(df) == 3


def test_scipy_timeout():
    from time import sleep

    sleep(1)  # 他のテストと時間が被らないように少し待つ

    def obj(_, opt_):
        sleep(1)
        return np.linalg.norm(opt_.get_variables(format='values'))

    opt = ScipyOptimizer('SLSQP')
    opt.fem = NoFEM()
    opt.add_parameter('x1', -1, -1, 1)
    opt.add_parameter('x2', -1, -1, 1)
    opt.add_objective('norm', obj, args=(opt,))
    opt.timeout = 5
    opt.run()
    df = opt.history.get_df()
    assert len(df) <= 5


def test_scipy_restart():
    csv_path = 'pyfemtet.opt.test_scipy_restart.csv'
    if os.path.isfile(csv_path):
        os.remove(csv_path)

    # 1st
    opt = ScipyOptimizer('SLSQP')
    opt.options = {'eps': 0.2}
    opt.fem = NoFEM()
    opt.add_parameter('x1', -1, -1, 1)
    opt.add_parameter('x2', -1, -1, 1)
    opt.add_objective('norm', lambda _, opt_: np.linalg.norm(opt_.get_variables(format='values')), args=(opt,))
    opt.n_trials = 3
    opt.history.path = csv_path
    opt.run()
    df = opt.history.get_df()
    optimal = df[df['optimality']].iloc[-1]
    assert len(df) == 3, len(df)

    # 2nd
    opt.run()
    df = opt.history.get_df()
    assert len(df) == 6, len(df)
    optimal = optimal[opt.history.prm_names]
    df = df[opt.history.prm_names]
    print('optimal is:')
    print(optimal)
    print('df.iloc[3] is:')
    print(df.iloc[3])
    assert (df.iloc[3] == optimal).all(), df.iloc[3] == optimal  # noqa

    # 3rd
    opt = ScipyOptimizer('SLSQP')
    opt.options = {'eps': 0.2}
    opt.fem = NoFEM()
    opt.add_parameter('x1', -1, -1, 1)
    opt.add_parameter('x2', -1, -1, 1)
    opt.add_objective(
        "norm",
        lambda _, opt_: np.linalg.norm(opt_.get_variables(format="values")),
        args=(opt,),
    )
    opt.n_trials = 3
    opt.history.path = csv_path
    opt.run()
    df = opt.history.get_df()
    assert len(df) == 9, len(df)


if __name__ == '__main__':
    # test_scipy_optimizer()
    # test_scipy_optimizer_1()
    test_scipy_n_trials()
    test_scipy_timeout()
    # test_scipy_restart()
