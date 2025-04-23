from pyfemtet.opt import FemtetInterface, FEMOpt, AbstractOptimizer


def ex_in(_, opt: AbstractOptimizer):
    ex_r, in_r = opt.get_parameter('values')
    return ex_r - in_r



if __name__ == '__main__':

    fem = FemtetInterface(
        parametric_output_indexes_use_as_objective={
            0: "minimize",
            1: "minimize",
            2: "minimize",
        },
    )

    femopt = FEMOpt(fem=fem)

    femopt.add_parameter("external_radius", 10, 1, 10)
    femopt.add_parameter("internal_radius", 5, 1, 10)
    femopt.add_constraint(ex_in, lower_bound=1, strict=True, args=(femopt.opt,))
    femopt.set_random_seed(42)
    femopt.optimize(
        n_trials=10,
        n_parallel=1,
    )
