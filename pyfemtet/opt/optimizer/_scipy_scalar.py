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
from pyfemtet.opt.optimizer import AbstractOptimizer, logger, OptimizationMethodChecker
from pyfemtet.core import MeshError, ModelError, SolveError
from pyfemtet._message import Msg


class ScipyScalarMethodChecker(OptimizationMethodChecker):
    def check_incomplete_bounds(self, raise_error=True): return True
    def check_seed(self, raise_error=True):
        logger.warning(Msg.WARN_SCIPY_DOESNT_NEED_SEED)
        return True


class ScipyScalarOptimizer(AbstractOptimizer):
    """Optimizer using ```scipy```.

    This class provides an interface for the optimization
    engine using Scipy. For more details, please refer to
    the Scipy documentation.

    See Also:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html

    Args:
        **minimize_kwargs:
            The keyword arguments of
            ```scipy.optimize.minimize_scalar```.

    Attributes:
        res (OptimizeResult):
            The return value of ```scipy.optimize.minimize_scalar```.

    """

    def __init__(
            self,
            **minimize_kwargs,
    ):
        super().__init__()

        # define members
        self.minimize_kwargs: dict = dict()
        self.minimize_kwargs.update(minimize_kwargs)
        self.res: OptimizeResult = None
        self.method_checker: OptimizationMethodChecker = ScipyScalarMethodChecker(self)

    def _objective(self, x: float):  # x: candidate parameter
        # update parameter

        df = self.get_parameter('df')
        df['value'] = x
        self.fem.update_parameter(df)

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
        params = self.get_parameter()
        assert len(params) == 1, print(f'{params} parameter(s) are passed.')

        # create bounds
        if 'bounds' not in self.minimize_kwargs.keys():
            bounds = []

            row = self.get_parameter('df')
            lb, ub = row['lower_bound'].iloc[0], row['upper_bound'].iloc[0]

            if lb is None and ub is None:
                pass
            elif lb is None or ub is None:
                raise ValueError('Both lower and upper bounds must be set.')
            else:
                bounds = [lb, ub]
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
