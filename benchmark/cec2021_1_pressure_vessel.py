from pyfemtet.opt._test_utils.hyper_sphere import HyperSphere
from common import *
from mkfig import Graph


problem = None


class CEC2021_1_PressureVessel(Problem):

    def setup_parameters(self):
        self.femopt.add_parameter('x1', 0.51, 0.51, 99.49)
        self.femopt.add_parameter('x2', 0.51, 0.51, 99.49)
        self.femopt.add_parameter('x3', 10, 10, 200)
        self.femopt.add_parameter('x4', 10, 10, 200)

    def z1(self):
        prm = self.femopt.opt.get_parameter()
        return 0.0625 * prm['x1']

    def z2(self):
        prm = self.femopt.opt.get_parameter()
        return 0.0625 * prm['x2']

    @hidden_constraint('g1', upper_bound=0)
    @hidden_constraint('g2', upper_bound=0)
    def f1(self):
        x1, x2, x3, x4 = self.femopt.opt.get_parameter('values')
        z1, z2 = self.z1(), self.z2()

        out = (
                1.7781 * z1 * x3 ** 2
                + 0.6224 * z1 * x2 * x4
                + 3.16641 * z1 ** 2 * x4
                + 19.84 * z1 ** 2 * x3
        )

        return out

    def f2(self):
        x1, x2, x3, x4 = self.femopt.opt.get_parameter('values')
        z1, z2 = self.z1(), self.z2()

        out = (
                -pi * x3 ** 2 * x4
                - (4 / 3) * pi * x3 ** 3
        )

        return out

    def setup_objectives(self):
        self.femopt.add_objective(self.f1)
        self.femopt.add_objective(self.f2)

    def g1(self):
        x1, x2, x3, x4 = self.femopt.opt.get_parameter('values')
        z1, z2 = self.z1(), self.z2()
        return 0.00954 * x3 - z2

    def g2(self):
        x1, x2, x3, x4 = self.femopt.opt.get_parameter('values')
        z1, z2 = self.z1(), self.z2()
        return 0.0193 * x3 - z1

    def setup_constraints(self):
        pass

    @property
    def hv_reference(self) -> str or np.ndarray:
        return np.array([3.5964885e5, -7.3303829e3])


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
    problem = CEC2021_1_PressureVessel()

    for i in range(5):
        rand(100)
        nsga2(100)
        nsga3(100)
        tpe(100)
        pof_botorch(100)
