from numpy import pi, sin, cos
from pyfemtet.opt.interface import NoFEM
from pyfemtet.opt.optimizer import OptunaOptimizer, PoFConfig, PoFBoTorchSampler


def objective(_, opt: OptunaOptimizer):
    x = opt.get_variables(format='values')
    return (x ** 2).sum()


def constraint(_, opt: OptunaOptimizer):
    x = opt.get_variables(format='values')
    return sin(x[0]*4*2*pi) * cos(x[1]*4*2*pi)


def test_remove_hard_constraint_from_gp():
    # remove_hard_constraints_from_gp = False  # default
    remove_hard_constraints_from_gp = True

    opt = OptunaOptimizer(
        sampler_class=PoFBoTorchSampler,
        sampler_kwargs=dict(
            n_startup_trials=5,
            pof_config=PoFConfig(
                consider_explicit_hard_constraint=True,
                remove_hard_constraints_from_gp=remove_hard_constraints_from_gp,
            )
        )
    )

    opt.fem = NoFEM()

    for i in range(2):
        opt.add_parameter(f'x{i}', 1, -1, 1)

    opt.add_constraint('constraint', constraint, lower_bound=0, args=(opt,))
    opt.add_objective('objective', objective, direction='minimize', args=(opt,))

    opt.n_trials = 10
    opt.seed = 42
    opt.run()
    print()
    print('==========')
    print(f'{remove_hard_constraints_from_gp=}')


if __name__ == '__main__':
    test_remove_hard_constraint_from_gp()
