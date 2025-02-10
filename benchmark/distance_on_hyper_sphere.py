from pyfemtet.opt._test_utils.hyper_sphere import HyperSphere
from common import *
from mkfig import Graph


problem = None


class DistanceOnHyperSphere(Problem):

    hyper_sphere: HyperSphere

    def __init__(self, n=3, point=None, diamond_size=0.4):
        assert n >= 2
        self.n = n
        self.hyper_sphere = HyperSphere(n)
        self.x = None
        self.diamond_size = diamond_size
        if point is None:
            self.point = np.array([1. / np.sqrt(n)]*n)
        else:
            assert len(point) == n
            self.point = point

    def setup_parameters(self):
        self.femopt.add_parameter('r', 0.5, 0, 1)
        for i in range(1, self.n-1):
            self.femopt.add_parameter(f'fai{i}', 0, 0, pi)
        self.femopt.add_parameter(f'fai{self.n-1}', 0, 0, 2 * pi)

    @hidden_constraint('calc_outside_diamond', 0)
    def calc_distance(self):
        r, *fai = self.femopt.opt.get_parameter('values')
        self.hyper_sphere.calc(r, *fai)
        self.x = self.hyper_sphere._x  # x: N 次元デカルト座標
        return norm(self.x - self.point)

    def calc_outside_diamond(self):
        r, *fai = self.femopt.opt.get_parameter('values')
        self.hyper_sphere.calc(r, *fai)
        self.x = self.hyper_sphere._x  # x: N 次元デカルト座標
        abs_x = np.abs(self.x).sum()  # 各座標値の絶対値の和 = 第一象限における平面の折り返し
        return abs_x - self.diamond_size  # 外側を基準にした diamond からの違反量。>0 で feasible とする。

    def setup_objectives(self):
        self.femopt.add_objective(self.calc_distance, direction='minimize')

    def setup_constraints(self):
        # self.femopt.add_constraint(self.calc_outside_diamond, lower_bound=0, using_fem=False)
        pass


def pof_botorch(n_trials_):

    a = PoFBoTorch(
        n_startup_trials=10,
    )

    p = problem

    values, result_path = main(a, p, n_trials_)


def tpe(n_trials_=100):

    a = TPE(
        n_startup_trials=10,
    )

    p = problem

    values, result_path = main(a, p, n_trials_)


def rand(n_trials_=100):

    a = Rand(
    )

    p = problem

    values, result_path = main(a, p, n_trials_)


def nsga2(n_trials_=100):

    a = NSGA2(
        population_size=10
    )

    p = problem

    values, result_path = main(a, p, n_trials_)


def nsga3(n_trials_=100):

    a = NSGA3(
        population_size=10
    )

    p = problem

    values, result_path = main(a, p, n_trials_)


def gp(n_trials_=100):

    a = GP(
    )

    p = problem

    values, result_path = main(a, p, n_trials_)


if __name__ == '__main__':
    problem = DistanceOnHyperSphere(10)

    for i in range(5):
        # gp()
        rand(100)
        nsga2(100)
        nsga3(100)
        tpe(100)
        pof_botorch(100)
