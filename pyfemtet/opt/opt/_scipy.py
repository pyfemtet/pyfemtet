# typing
import logging
from typing import Iterable

# built-in
import os

# 3rd-party
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.optimize import minimize, OptimizeResult

# pyfemtet relative
from pyfemtet.opt._femopt_core import OptimizationStatus, generate_lhs
from pyfemtet.opt.opt import AbstractOptimizer, logger, OptimizationMethodChecker
from pyfemtet.core import MeshError, ModelError, SolveError


class ScipyMethodChecker(OptimizationMethodChecker):
    pass



class ScipyOptimizer(AbstractOptimizer):

    def __init__(
            self,
            **minimize_kwargs,
    ):
        """
        Args:
            **minimize_kwargs: Kwargs of `scipy.optimize.minimize`. __Except `fun`, `bounds` and `constraints`.__
        """
        super().__init__()

        # define members
        self.minimize_kwargs: dict = minimize_kwargs
        self.res: OptimizeResult = None
        self.method_checker: OptimizationMethodChecker = ScipyMethodChecker(self)

    def _objective(self, x: np.ndarray):  # x: candidate parameter
        # update parameter
        self.parameters['value'] = x
        self.fem.update_parameter(self.parameters)

        # strict constraints
        ...

        # fem
        try:
            _, obj_values, cns_values = self.f(x)
        except (ModelError, MeshError, SolveError) as e:
            logger.info(e)
            logger.info('以下の変数で FEM 解析に失敗しました。')
            print(self.get_parameter('dict'))

            # check interruption command
            if self.entire_status.get() == OptimizationStatus.INTERRUPTING:
                self.worker_status.set(OptimizationStatus.INTERRUPTING)
                raise StopIteration

            # skip
            ...
            raise StopIteration

        # constraints
        ...

        # check interruption command
        if self.entire_status.get() == OptimizationStatus.INTERRUPTING:
            self.worker_status.set(OptimizationStatus.INTERRUPTING)
            raise StopIteration

        # objectives to objective
        logger.setLevel(logging.DEBUG)
        logger.debug(f'x: {x}, y: {obj_values}')
        return obj_values[0]

    def _setup_before_parallel(self):
        logger.warn(f'{type(self)} is not implement parallel computing')

    def run(self):

        # seed check
        if self.seed is not None:
            raise NotImplementedError(f'{type(self)} is not implement seed specification.')

        # create init
        x0 = self.parameters['value'].values

        # create bounds
        bounds = []
        for i, row in self.parameters.iterrows():
            lb, ub = row['lb'], row['ub']
            bounds.append([lb, ub])

        # run optimize
        res = minimize(
            fun=self._objective,
            x0=x0,
            bounds=bounds,
            **self.minimize_kwargs
        )

        self.res = res
