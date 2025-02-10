from pyfemtet.opt._test_utils.hyper_sphere import HyperSphere
from common import *
from mkfig import Graph


problem = None


class DistancesOnHyperCube(Problem):

    def __init__(self, n=3, radius=0.45, reference_x=0.2):
        assert n >= 3
        self.n = n
        self.x = None
        self.radius = radius
        self.reference_x = reference_x

    def setup_parameters(self):
        for i in range(self.n):
            self.femopt.add_parameter(f'x{i}', 1, 0, 1)

    @hidden_constraint('calc_outside_distance', 0)
    def calc_distances(self):
        self.x = self.femopt.opt.get_parameter('values')

        distances = []
        for i in range(self.n - 1):
            reference_point = np.zeros(self.n)
            reference_point[i] = self.reference_x

            distance = norm(reference_point - self.x)
            distances.append(distance)

        return distances

    def calc_outside_distance(self):
        self.x = self.femopt.opt.get_parameter('values')
        center = 0.5 + np.zeros(self.n)
        distance = norm(self.x - center)
        return distance - self.radius

    def setup_objectives(self):
        self.femopt.add_objectives(
            fun=self.calc_distances,
            n_return=self.n - 1,
        )

    def setup_constraints(self):
        pass

    @property
    def hv_reference(self) -> str or np.ndarray:
        return self.reference_x * sqrt(2) * np.ones(self.n - 1)


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
    problem = DistancesOnHyperCube(5)

    for i in range(5):
        rand(100)
        nsga2(100)
        nsga3(100)
        tpe(100)
        pof_botorch(100)
