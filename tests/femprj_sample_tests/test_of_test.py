from pyfemtet.opt import FEMOpt, NoFEM, AbstractOptimizer

n_input = 3
n_trials = 10
error_timing = 5

counter = 0


class SampleException(Exception):
    pass


def liner(opt: AbstractOptimizer):
    x = opt.get_parameter('values')
    return x.sum()


def parabola(opt: AbstractOptimizer):
    x = opt.get_parameter('values')
    return (x**2).sum()


def raise_error():
    global counter
    counter += 1
    if counter == error_timing:
        raise SampleException('テスト用例外')
    return 0


if __name__ == '__main__':

    fem = NoFEM()

    femopt = FEMOpt(fem=fem)

    # input
    for i in range(n_input):
        femopt.add_parameter(f'x{i}', 0.5, -1, 1)

    # output
    femopt.add_objective(liner, args=(femopt.opt,))
    femopt.add_objective(parabola, args=(femopt.opt,))
    femopt.add_objective(raise_error)

    # run
    femopt.optimize(n_trials=n_trials)
