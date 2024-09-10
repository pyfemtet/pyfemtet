import numpy as np

from pyfemtet.opt import FEMOpt, ScipyOptimizer, NoFEM
from pyfemtet._test_util import SuperSphere


s = SuperSphere(3)


def coordinate(opt, i):
    return s.x(*opt.get_parameter('values'))[i]


if __name__ == '__main__':
    opt = ScipyOptimizer(
        # method='SLSQP',
    )
    fem = NoFEM()
    femopt = FEMOpt(opt=opt, fem=fem)

    femopt.add_parameter('r', 0.5)
    femopt.add_parameter('fai', np.pi/2)
    femopt.add_parameter('theta', 0)

    femopt.add_objective(coordinate, 'x', args=(femopt.opt, 0))
    # femopt.add_objective(coordinate, 'y', args=(femopt.opt, 1))
    # femopt.add_objective(coordinate, 'z', args=(femopt.opt, 2))

    femopt.optimize()
    print(femopt.opt.res)
    print('終了')
    input()
    femopt.terminate_all()
