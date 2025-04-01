import numpy as np

from optuna.samplers import TPESampler, RandomSampler, NSGAIISampler

from pyfemtet.opt.interface import NoFEM
from pyfemtet.opt.optimizer import OptunaOptimizer
from pyfemtet.opt.optimizer.optuna_optimizer.pof_botorch.pof_botorch_sampler import (
    PoFBoTorchSampler,
    PoFConfig,
    PartialOptimizeACQFConfig,
)
from pyfemtet.opt.exceptions import SolveError

from tests import HyperSphere

feasible_in_barrier = False


def objective(_, opt: OptunaOptimizer, hs: HyperSphere, b, explicit_constraint):
    x: np.array = opt.get_variables('values')
    y = hs.calc(x)

    obj = np.linalg.norm(y - b)
    if feasible_in_barrier:
        print('理論的な最小値: ', 1 - 1 / np.sqrt(hs.n))
    else:
        print('理論的な最小値: ', 0)
    print('今の値: ', obj)

    if not explicit_constraint:
        if feasible_in_barrier:
            if barrier(_, opt, hs) > 1:
                raise SolveError
        else:
            if barrier(_, opt, hs) < 0.8:
                raise SolveError

    return obj


def barrier(_, opt: OptunaOptimizer, hs: HyperSphere):
    x: np.array = opt.get_variables('values')
    y = hs.calc(x)
    return np.sum(np.abs(y))


def main(d=5, explicit_constraint=True):

    hs = HyperSphere(d)
    b = np.array([1/np.sqrt(hs.n)] * hs.n)

    opt = OptunaOptimizer()
    opt.sampler_class = PoFBoTorchSampler
    opt.sampler_kwargs = dict(

        observation_noise='no',  # 'no' or None or specific value
        # observation_noise=None,  # 'no' or None or specific value

        pof_config=PoFConfig(
            # feasibility_noise='no',
            feasibility_noise=None,
        ),

        partial_optimize_acqf_kwargs=PartialOptimizeACQFConfig(

        ),
    )

    opt.add_parameter('r', 1, 0, 1)
    for i in range(d-2):
        opt.add_parameter(f'x{i}', 0, 0, np.pi)
    opt.add_parameter(f'x{d-1}', 0, 0, 2*np.pi)

    opt.add_objective('distance', objective, args=(opt, hs, b, explicit_constraint))

    if explicit_constraint:
        if feasible_in_barrier:
            opt.add_constraint('inside diamond', barrier, upper_bound=1, args=(opt, hs,))
        else:
            opt.add_constraint('outside diamond', barrier, lower_bound=0.8, args=(opt, hs,))

    opt.fem = NoFEM()
    opt.n_trials = 50
    opt.seed = 42
    opt.run()


if __name__ == '__main__':
    main(5, False)
