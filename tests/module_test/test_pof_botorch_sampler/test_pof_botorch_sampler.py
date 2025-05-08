from time import time, sleep
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
    PoFConfig,
    PartialOptimizeACQFConfig,
)
from pyfemtet.opt.exceptions import SolveError


def constraint(_: NoFEM, opt_: OptunaOptimizer):
    x: np.array = opt_.get_variables('values')
    return np.abs(x).sum()


def constraint2(_, opt_: OptunaOptimizer, i_: int):
    x: np.array = opt_.get_variables('values')
    sleep(0.1)  # scipy.optimize を遅らせるための sleep
    return x[i_]


def eval_hidden_constraint(x: np.ndarray):
    if np.all(x > 0):
        raise SolveError


def objective(_, opt_: OptunaOptimizer, i_: int):
    x: np.array = opt_.get_variables('values')
    eval_hidden_constraint(x)
    return x[i_] ** 2


def objective2(_, opt_: OptunaOptimizer, i_: int, time_check: list):
    x: np.array = opt_.get_variables('values')
    time_check.append(time())
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


def test_timeout():

    def sub(timeout_sec):

        opt = OptunaOptimizer()
        opt.fem = NoFEM()
        opt.sampler_class = PoFBoTorchSampler
        opt.sampler_kwargs = dict(
            n_startup_trials=1,
            partial_optimize_acqf_kwargs=PartialOptimizeACQFConfig(
                timeout_sec=timeout_sec,
                # scipy_minimize_kwargs=
            ),
            pof_config=PoFConfig(

            ),
        )
        opt.n_trials = 2
        opt.seed = 42

        d = 5
        m = 1

        time_check = []

        for i in range(d):
            opt.add_parameter(f'x{i}', 1, -1, 1)

        for i in range(m):
            opt.add_objective(
                name=f'y{i}',
                fun=objective2,
                args=(opt, i, time_check),
            )

        for i in range(d):
            opt.add_constraint(name=f'c{i}', fun=constraint2, args=(opt, i,), lower_bound=0.1)

        opt.run()

        print(int(time_check[-1] - time_check[-2]), 'sec')

        return int(time_check[-1] - time_check[-2])

    time_1 = sub(timeout_sec=10)
    time_2 = sub(timeout_sec=1)
    assert time_1 > time_2


if __name__ == '__main__':
    test_logei_candidates_func()
    test_qehvi_candidates_func()
    test_qehvi_candidates_func()
    test_qparego_candidates_func()
    test_ehvi_candidates_func()
    test_timeout()
