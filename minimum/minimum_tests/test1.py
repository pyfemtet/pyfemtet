import os
from minimum.base import OptimizationOptuna
from minimum.fem import FEM, Femtet


os.chdir(os.path.dirname(__file__))


def plane(x):
    return x.sum()


def parabola(x):
    return (x**2).sum()


if __name__ == '__main__':
    femopt = OptimizationOptuna()

    parameters = dict(
        name=['x', 'y', 'z'],
        value=[.5, .5, .5],
        lb=[-1, -1, -1],
        ub=[1, 1, 1]
    )

    femopt.set_fem(Femtet())
    femopt.set_parameters(parameters)
    femopt.set_objective(plane, '平面(最小最小)')
    femopt.set_objective(parabola, '放物線(原点最小)')

    femopt.main()


