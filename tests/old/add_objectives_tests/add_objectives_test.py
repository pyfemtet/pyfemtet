import os
from optuna.samplers import QMCSampler
from pyfemtet.opt import FemtetInterface, FEMOpt, OptunaOptimizer


def objective(Femtet):
    return 1.


def objectives(Femtet, opt: OptunaOptimizer, hi='good bye'):
    print(f'===== `objectives()` is called once on one trial! =====')
    print(f'{Femtet.Version=}')
    print(f'{hi=}')
    d = opt.get_parameter()
    return d['a'], d['b']


if __name__ == '__main__':
    fem = FemtetInterface(
        femprj_path=os.path.dirname(__file__) + '/dummy.femprj',
        connect_method='new',
    )

    opt = OptunaOptimizer(
        sampler_class=QMCSampler,
    )

    femopt = FEMOpt(fem=fem, opt=opt)

    femopt.add_parameter('a', 0, 0, 1)
    femopt.add_parameter('b', 0, 0, 1)

    femopt.add_objective(objective)
    femopt.add_objectives(objectives, 2, args=(femopt.opt,), kwargs=dict(hi='hello'))

    femopt.set_random_seed(42)
    femopt.optimize(n_trials=10, confirm_before_exit=False)

    fem.quit()
