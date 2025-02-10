import numpy as np

from pyfemtet.opt._test_utils.hyper_sphere import HyperSphere
from common import *
from mkfig import Graph


problem = None

S = np.array([[2, 3, 4], [4, 6, 3]])
t = np.array([[8, 20, 8], [16, 4, 4]])
H = 6000
alp = 250
beta = 0.6
Q1 = 40000
Q2 = 20000


class CEC2021_19_MultiProductBatchPlant(Problem):

    def setup_parameters(self):
        self.femopt.add_parameter('N1', 1, 1, 3, step=1)
        self.femopt.add_parameter('N2', 1, 1, 3, step=1)
        self.femopt.add_parameter('N3', 1, 1, 3, step=1)
        self.femopt.add_parameter('V1', 250, 250, 2500)
        self.femopt.add_parameter('V2', 250, 250, 2500)
        self.femopt.add_parameter('V3', 250, 250, 2500)
        self.femopt.add_parameter('TL1', 6, 6, 20)
        self.femopt.add_parameter('TL2', 4, 4, 16)
        self.femopt.add_parameter('B1', 40, 40, 700)
        self.femopt.add_parameter('B2', 10, 10, 450)

    @hidden_constraint('g1', upper_bound=0)
    @hidden_constraint('g2', upper_bound=0)
    @hidden_constraint('g3', upper_bound=0)
    @hidden_constraint('g4', upper_bound=0)
    @hidden_constraint('g5', upper_bound=0)
    @hidden_constraint('g6', upper_bound=0)
    @hidden_constraint('g7', upper_bound=0)
    @hidden_constraint('g8', upper_bound=0)
    @hidden_constraint('g9', upper_bound=0)
    @hidden_constraint('g10', upper_bound=0)
    def f1(self):
        (N1, N2, N3,
         V1, V2, V3,
         TL1, TL2,
         B1, B2) = self.femopt.opt.get_parameter('values')

        out = alp * (
                N1 * V1 ** beta
                + N2 * V2 ** beta
                + N3 * V3 ** beta
        )

        return out

    def f2(self):
        (N1, N2, N3,
         V1, V2, V3,
         TL1, TL2,
         B1, B2) = self.femopt.opt.get_parameter('values')

        out = 65 * (Q1 / B1 + Q2 / B2) + 0.08 * Q1 + 0.1 * Q2

        return out

    def f3(self):
        (N1, N2, N3,
         V1, V2, V3,
         TL1, TL2,
         B1, B2) = self.femopt.opt.get_parameter('values')

        out = Q1 * TL1 / B1 + Q2 * TL2 / B2

        return out

    def setup_objectives(self):
        self.femopt.add_objective(self.f1)
        self.femopt.add_objective(self.f2)
        self.femopt.add_objective(self.f3)

    def g1(self):
        (N1, N2, N3,
         V1, V2, V3,
         TL1, TL2,
         B1, B2) = self.femopt.opt.get_parameter('values')

        out = Q1 * TL1 / B1 + Q2 * TL2 / B2 - H

        return out

    def g2(self):
        (N1, N2, N3,
         V1, V2, V3,
         TL1, TL2,
         B1, B2) = self.femopt.opt.get_parameter('values')

        out = S[0, 0] * B1 + S[1, 0] * B2 - V1

        return out

    def g3(self):
        (N1, N2, N3,
         V1, V2, V3,
         TL1, TL2,
         B1, B2) = self.femopt.opt.get_parameter('values')

        out = S[0, 1] * B1 + S[1, 1] * B2 - V2

        return out

    def g4(self):
        (N1, N2, N3,
         V1, V2, V3,
         TL1, TL2,
         B1, B2) = self.femopt.opt.get_parameter('values')

        out = S[0, 2] * B1 + S[1, 2] * B2 - V3

        return out

    def g5(self):
        (N1, N2, N3,
         V1, V2, V3,
         TL1, TL2,
         B1, B2) = self.femopt.opt.get_parameter('values')

        return t[0, 0] - N1 * TL1

    def g6(self):
        (N1, N2, N3,
         V1, V2, V3,
         TL1, TL2,
         B1, B2) = self.femopt.opt.get_parameter('values')

        return t[0, 1] - N2 * TL1

    def g7(self):
        (N1, N2, N3,
         V1, V2, V3,
         TL1, TL2,
         B1, B2) = self.femopt.opt.get_parameter('values')

        return t[0, 2] - N3 * TL1

    def g8(self):
        (N1, N2, N3,
         V1, V2, V3,
         TL1, TL2,
         B1, B2) = self.femopt.opt.get_parameter('values')

        return t[1, 0] - N1 * TL2

    def g9(self):
        (N1, N2, N3,
         V1, V2, V3,
         TL1, TL2,
         B1, B2) = self.femopt.opt.get_parameter('values')

        return t[1, 1] - N2 * TL2

    def g10(self):
        (N1, N2, N3,
         V1, V2, V3,
         TL1, TL2,
         B1, B2) = self.femopt.opt.get_parameter('values')

        return t[1, 2] - N3 * TL2

    def setup_constraints(self):
        pass

    @property
    def hv_reference(self) -> str or np.ndarray:
        return np.array([2.4435182e5, 4.8572551e4, 6e3])


def pof_botorch(n_trials_=100, timeout_=3600):

    a = PoFBoTorch(
        n_startup_trials=10,
    )

    p = problem

    values, result_path = main(a, p, n_trials_, timeout_)


def tpe(n_trials_=100, timeout_=3600):

    a = TPE(
        n_startup_trials=10,
    )

    p = problem

    values, result_path = main(a, p, n_trials_, timeout_)


def rand(n_trials_=100, timeout_=3600):

    a = Rand(
    )

    p = problem

    values, result_path = main(a, p, n_trials_, timeout_)


def nsga2(n_trials_=100, timeout_=3600):

    a = NSGA2(
        population_size=10
    )

    p = problem

    values, result_path = main(a, p, n_trials_, timeout_)


def nsga3(n_trials_=100, timeout_=3600):

    a = NSGA3(
        population_size=10
    )

    p = problem

    values, result_path = main(a, p, n_trials_, timeout_)


def gp(n_trials_=100, timeout_=3600):

    a = GP(
    )

    p = problem

    values, result_path = main(a, p, n_trials_, timeout_)


if __name__ == '__main__':
    problem = CEC2021_19_MultiProductBatchPlant()

    for i in range(5):
        # rand(50, 21600)
        nsga2(100, 21600)
        nsga3(100, 21600)
        tpe(100, 21600)
        pof_botorch(100, 21600)
