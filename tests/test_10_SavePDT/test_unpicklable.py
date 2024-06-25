from time import sleep

from pyfemtet.opt.interface._femtet import _UnPicklableNoFEM

import numpy as np

from pyfemtet.opt import FEMOpt, NoFEM


class SuperSphere:
    def __init__(self, n):
        self.n = n

    def x(self, radius, *angles):
        assert len(angles) == self.n - 1, 'invalid angles length'

        out = []

        for i in range(self.n):
            if i == 0:
                out.append(radius * np.cos(angles[0]))
            elif i < self.n - 1:
                product = radius
                for j in range(i):
                    product *= np.sin(angles[j])
                product *= np.cos(angles[i])
                out.append(product)
            else:
                product = radius
                for j in range(i):
                    product *= np.sin(angles[j])
                out.append(product)
        return out


s = SuperSphere(3)


def coordinate(_, opt, i):
    sleep(1)
    return s.x(*opt.get_parameter('values'))[i]


if __name__ == '__main__':

    fem = _UnPicklableNoFEM()
    femopt = FEMOpt(fem=fem)

    femopt.add_parameter('r', 0.5, 0, 1)
    femopt.add_parameter('fai', np.pi/2, 0, np.pi)
    femopt.add_parameter('theta', 0, 0, 2*np.pi)

    femopt.add_objective(coordinate, 'x', args=(femopt.opt, 0))
    # femopt.add_objective(coordinate, 'y', args=(femopt.opt, 1))
    # femopt.add_objective(coordinate, 'z', args=(femopt.opt, 2))

    femopt.optimize(n_parallel=3, n_trials=60)
    femopt.terminate_all()

