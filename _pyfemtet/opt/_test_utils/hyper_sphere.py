import numpy as np
from numpy import sin, cos
from pyfemtet.opt import AbstractOptimizer


class HyperSphere(object):

    def __init__(self, N):
        self.N = N
        self._x = np.zeros(self.N, dtype=float)

    def calc(self, r, *angles):
        _x = []
        for i in range(self.N - 1):
            __x = r * np.prod([sin(angles[j]) for j in range(i)])
            __x = __x * cos(angles[i])
            _x.append(__x)
        _x.append(r * np.prod([sin(angles[j]) for j in range(self.N - 1)]))
        self._x = np.array(_x)

    def x(self, opt: AbstractOptimizer, index: int):
        r, *angles = opt.get_parameter("values")
        self.calc(r, *angles)
        return self._x[index]
