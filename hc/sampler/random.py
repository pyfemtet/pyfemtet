import numpy as np

from hc.sampler._base import AbstractSampler
from hc.problem._base import Floats


class RandomSampler(AbstractSampler):

    def setup(self):
        np.random.seed(seed=42)

    def candidate_x(self) -> Floats:
        out = []
        for lb, ub in self.problem.bounds:
            out.append(np.random.rand() * (ub - lb) + lb)
        return out
