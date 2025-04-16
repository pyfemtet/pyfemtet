import numpy as np

from pyfemtet.opt.interface import NoFEM
from pyfemtet.opt.optimizer import OptunaOptimizer
from pyfemtet.opt.optimizer.optuna_optimizer._pof_botorch.pof_botorch_sampler import (
    PoFBoTorchSampler,
    logei_candidates_func,
    qei_candidates_func,
    qehvi_candidates_func,
    qparego_candidates_func,
    ehvi_candidates_func,
)
from pyfemtet.opt.exceptions import SolveError


def constraint(_: NoFEM, opt_: OptunaOptimizer):
    x: np.array = opt_.get_variables('values')
    return np.abs(x).sum()


def eval_hidden_constraint(x: np.ndarray):
    if np.all(x > 0):
        raise SolveError


def objective(_, opt_: OptunaOptimizer, i_: int):
    x: np.array = opt_.get_variables('values')
    eval_hidden_constraint(x)
    return x[i_] ** 2


def main(can_fun, d, m, explicit_constraint):

    fem = NoFEM()

    opt = OptunaOptimizer()
    opt.fem = fem
    opt.sampler_class = PoFBoTorchSampler
    opt.sampler_kwargs = dict(
        candidates_func=can_fun,
        n_startup_trials=3,
    )
    opt.n_trials = 6
    opt.seed = 42

    for i in range(d):
        opt.add_parameter(f'x{i}', -1, -1, 1)

    for i in range(m):
        opt.add_objective(name=f'y{i}', fun=objective, args=(opt, i,))

    if explicit_constraint:
        opt.add_constraint(name=f'cns', fun=constraint, args=(opt,), lower_bound=0.25)

    opt.run()


def test_logei_candidates_func():
    main(logei_candidates_func,
         3, 1, False)


def test_qei_candidates_func():
    main(qei_candidates_func,
         3, 1, True)


def test_qehvi_candidates_func():
    main(qehvi_candidates_func,
         3, 2, True)


def test_qparego_candidates_func():
    main(qparego_candidates_func,
         5, 4, True)


def test_ehvi_candidates_func():
    main(ehvi_candidates_func,
         5, 4, False)


if __name__ == '__main__':
    test_logei_candidates_func()
    test_qehvi_candidates_func()
    test_qehvi_candidates_func()
    test_qparego_candidates_func()
    test_ehvi_candidates_func()
