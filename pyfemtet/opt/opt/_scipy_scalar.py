# typing
import logging
from typing import Iterable

# built-in
import os

# 3rd-party
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.optimize import minimize_scalar, OptimizeResult

# pyfemtet relative
from pyfemtet.opt._femopt_core import OptimizationStatus, generate_lhs
from pyfemtet.opt.opt import AbstractOptimizer, logger, OptimizationMethodChecker
from pyfemtet.core import MeshError, ModelError, SolveError


class ScipyScalarMethodChecker(OptimizationMethodChecker):
    def check_incomplete_bounds(self, raise_error=True): return True


class ScipyScalarOptimizer(AbstractOptimizer):

    def __init__(
            self,
            **minimize_kwargs,
    ):
        """
        Args:
            **minimize_kwargs: Kwargs of `scipy.optimize.minimize_scalar` __except ``fun``.__.
        """
        super().__init__()

        # define members
        self.minimize_kwargs: dict = dict()
        self.minimize_kwargs.update(minimize_kwargs)
        self.res: OptimizeResult = None
        self.method_checker: OptimizationMethodChecker = ScipyScalarMethodChecker(self)

    def _objective(self, x: float):  # x: candidate parameter
        # update parameter
        self.parameters['value'] = x
        self.fem.update_parameter(self.parameters)

        # strict constraints
        ...

        # fem
        try:
            _, obj_values, cns_values = self.f(x)
        except (ModelError, MeshError, SolveError) as e:
            # 現状、エラーが起きたらスキップできない
            raise StopIteration

        # constraints
        ...

        # check interruption command
        if self.entire_status.get() == OptimizationStatus.INTERRUPTING:
            self.worker_status.set(OptimizationStatus.INTERRUPTING)
            raise StopIteration

        # objectives to objective

        return obj_values[0]

    def _setup_before_parallel(self):
        pass

    def run(self):

        # create init
        assert len(self.parameters) == 1

        # create bounds
        if 'bounds' not in self.minimize_kwargs.keys():
            bounds = []
            for i, row in self.parameters.iterrows():
                lb, ub = row['lower_bound'], row['upper_bound']
                if lb is None: lb = -np.inf
                if ub is None: ub = np.inf
                bounds.append([lb, ub])
            self.minimize_kwargs.update(
                {'bounds': bounds}
            )

        # run optimize
        try:
            res = minimize_scalar(
                fun=self._objective,
                **self.minimize_kwargs,
            )
        except StopIteration:
            res = None
            logger.warn('Optimization has been interrupted. '
                        'Note that you cannot acquire the OptimizationResult.')

        self.res = res
