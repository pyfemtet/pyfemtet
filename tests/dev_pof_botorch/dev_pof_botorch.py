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
explicit_constraint = False


def objective(_, opt: OptunaOptimizer, hs: HyperSphere, b):
    x: np.array = opt.get_variables('values')
    y = hs.calc(x)

    obj = np.linalg.norm(y - b)
    if feasible_in_barrier:
        print('理論的な最小値: ', 1 - 1 / np.sqrt(hs.n))
    else:
        print('理論的な最小値: ', 0)
    df = opt.history.get_df(equality_filters=opt.history.MAIN_FILTER)
    obj_name = opt.history.obj_names[0]
    print('今の最小値: ', min(obj, df[obj_name].dropna().min()))

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


def core(
        d=5,
        seed=None,
        observation_noise=None,
        feasibility_noise=None,
):

    hs = HyperSphere(d)
    b = np.array([1/np.sqrt(hs.n)] * hs.n)

    opt = OptunaOptimizer()
    opt.history.path = f'{observation_noise=}_{feasibility_noise=}_{seed=}.csv'

    import os
    if os.path.isfile(opt.history.path):
        return

    opt.sampler_class = PoFBoTorchSampler
    opt.sampler_kwargs = dict(

        # observation_noise='no',  # 'no' or None or specific value
        # observation_noise=None,  # 'no' or None or specific value
        observation_noise=observation_noise,

        pof_config=PoFConfig(
            # feasibility_noise='no',
            # feasibility_noise=None,
            feasibility_noise=feasibility_noise,
        ),

        partial_optimize_acqf_kwargs=PartialOptimizeACQFConfig(
            method='SLSQP',
            # scipy_minimize_kwargs=dict(ftol=0.01, disp=False),
        ),
    )

    opt.add_parameter('r', 1, 0, 1)
    for i in range(d-2):
        opt.add_parameter(f'x{i}', 0, 0, np.pi)
    opt.add_parameter(f'x{d-1}', 0, 0, 2*np.pi)

    opt.add_objective('distance', objective, args=(opt, hs, b))

    if explicit_constraint:
        if feasible_in_barrier:
            opt.add_constraint('inside diamond', barrier, upper_bound=1, args=(opt, hs,))
        else:
            opt.add_constraint('outside diamond', barrier, lower_bound=0.8, args=(opt, hs,))

    opt.fem = NoFEM()
    opt.n_trials = 50
    opt.seed = seed
    opt.run()


def main(o_noise, f_noise):
    d = 5
    core(d, 1, o_noise, f_noise, )
    core(d, 2, o_noise, f_noise, )
    core(d, 3, o_noise, f_noise, )
    core(d, 4, o_noise, f_noise, )
    core(d, 5, o_noise, f_noise, )


if __name__ == '__main__':
    main(None, None)
    main(None, 'no')
    main('no', None)
    main('no', 'no')
