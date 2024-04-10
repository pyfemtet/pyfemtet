import os
import numpy as np
from pyfemtet.opt import FEMOpt, FemtetWithSolidworksInterface


here, me = os.path.split(__file__)


class HyperSphere:
    def __init__(self, n=3):
        assert n >= 2
        self.n = n
        self.r = 0.
        self.fai = np.zeros(n-1)
        self.x = np.zeros(self.n)

    def compute(self):
        tmp_x = np.empty(self.n)
        _x = self.r
        for i in range(self.n-1):
            tmp_x[i] = _x * np.cos(self.fai[i])
            _x *= np.sin(self.fai[i])
        tmp_x[self.n-1] = _x
        self.x = tmp_x

hs = HyperSphere()


def x(Femtet, opt):
    r, *fai = opt.get_parameter('value')
    hs.r = r
    hs.fai = np.array(fai)
    hs.compute()
    return hs.x[0]


def y(Femtet):
    return hs.x[1]


def z(Femtet):
    return hs.x[2]


def test_cad_sw():
    sldprt_path = os.path.join(here, f'{me.replace(".py", ".SLDPRT")}')
    femprj_path = os.path.join(here, f'{me.replace(".py", ".femprj")}')

    fem = FemtetWithSolidworksInterface(
        sldprt_path=sldprt_path,
        femprj_path=femprj_path,
        connect_method='new',
        strictly_pid_specify=False,
    )

    femopt = FEMOpt(fem=fem)
    femopt.add_parameter('r', 0.5, 0, 1)
    femopt.add_parameter('fai0', 0, 0, np.pi)
    femopt.add_parameter('fai1', 0, 0, 2*np.pi)
    femopt.add_objective(x, 'x(mm)', args=femopt.opt)
    femopt.add_objective(y, 'y(mm)')
    femopt.add_objective(z, 'z(mm)')
    femopt.optimize(n_trials=10)
    femopt.terminate_all()


if __name__ == '__main__':
    test_cad_sw()
