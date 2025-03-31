import os
import numpy as np
from numpy import pi, sin, cos


def get(__file__, filename):
    return os.path.join(os.path.dirname(__file__), filename)


class HyperSphere:

    def __init__(self, n):
        assert n >= 2
        self.n = n
        self.bounds = []
        self.bounds.append([0, 1])
        self.bounds.append([0, 2 * pi])
        for i in range(n - 2):
            self.bounds.append([0, pi])
        self.bounds = np.array(self.bounds)

    def calc(self, x):
        r, *angles = x
        _x = []
        for i in range(self.n - 1):
            __x = r * np.prod([sin(angles[j]) for j in range(i)])
            __x = __x * cos(angles[i])
            _x.append(__x)
        _x.append(r * np.prod([sin(angles[j]) for j in range(self.n - 1)]))
        return np.array(_x)
