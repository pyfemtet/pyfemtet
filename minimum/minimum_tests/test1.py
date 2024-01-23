import os
from win32com.client import constants
from minimum.base import OptimizationBase
# from minimum.fem import FEM, Femtet


os.chdir(os.path.dirname(__file__))


def plane(x):
    print(constants.STATIC_C)
    return x.sum()


def parabola(x):
    return (x**2).sum()


if __name__ == '__main__':
    femopt = OptimizationBase()

    parameters = dict(
        name=['x', 'y', 'z'],
        value=[.5, .5, .5],
        lb=[-1, -1, -1],
        ub=[1, 1, 1]
    )

    femopt.set_parameters(parameters)
    femopt.set_objective(plane, '平面(最小最小)')
    femopt.set_objective(parabola, '放物線(原点最小)')

    femopt.main(n_parallel=3)


