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
from pyfemtet.opt.optimizer import AbstractOptimizer, logger, OptimizationMethodChecker
from pyfemtet.core import MeshError, ModelError, SolveError
from pyfemtet._message import Msg


class StopIteration2(Exception):
    pass


class StopIterationCallback:
    def __init__(self, opt):
        self.opt: ScipyOptimizer = opt
        self.res: OptimizeResult = None

    def stop_iteration(self):
        # stop iteration gimmick
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        if self.opt.minimize_kwargs['method'] == "trust-constr":
            raise StopIteration2  # supports nothing
        elif (
                self.opt.minimize_kwargs['method'] == 'TNC'
                or self.opt.minimize_kwargs['method'] == 'SLSQP'
                or self.opt.minimize_kwargs['method'] == 'COBYLA'
        ):
            raise StopIteration2  # supports xk
        else:
            raise StopIteration  # supports xk , intermediate_result and StopIteration


    def __call__(self, xk=None, intermediate_result=None):
        self.res = intermediate_result
        if self.opt.entire_status.get() == OptimizationStatus.INTERRUPTING:
            self.opt.worker_status.set(OptimizationStatus.INTERRUPTING)
            self.stop_iteration()


class ScipyMethodChecker(OptimizationMethodChecker):
    def check_incomplete_bounds(self, raise_error=True): return True
    def check_seed(self, raise_error=True):
        logger.warning(Msg.WARN_SCIPY_DOESNT_NEED_SEED)
        return True


class ScipyOptimizer(AbstractOptimizer):
    """Optimizer using ```scipy```.

    This class provides an interface for the optimization
    engine using Scipy. For more details, please refer to
    the Scipy documentation.

    See Also:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Args:
        **minimize_kwargs:
            The keyword arguments of
            ```scipy.optimize.minimize```.

    Attributes:
        res (OptimizeResult):
            The return value of ```scipy.optimize.minimize```.

    """

    def __init__(
            self,
            **minimize_kwargs,
    ):
        super().__init__()

        # define members
        self.minimize_kwargs: dict = dict(
            method='L-BFGS-B',
        )
        self.minimize_kwargs.update(minimize_kwargs)
        self.res: OptimizeResult = None
        self.method_checker: OptimizationMethodChecker = ScipyMethodChecker(self)
        self.stop_iteration_callback = StopIterationCallback(self)

    def _objective(self, x: np.ndarray):  # x: candidate parameter
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
            # 現在の技術的にエラーが起きたらスキップできない
            logger.error(Msg.ERR_FEM_FAILED_AND_CANNOT_CONTINUE)
            raise StopIteration2

        # constraints
        ...

        # # check interruption command
        # if self.entire_status.get() == OptimizationStatus.INTERRUPTING:
        #     self.worker_status.set(OptimizationStatus.INTERRUPTING)
        #     raise StopOptimize

        # objectives to objective

        return obj_values[0]

    def _setup_before_parallel(self):
        pass

    def run(self):

        # create init
        x0 = self.get_parameter('values')

        # create bounds
        if 'bounds' not in self.minimize_kwargs.keys():
            bounds = []
            for i, row in self.get_parameter('df').iterrows():
                lb, ub = row['lower_bound'], row['upper_bound']
                if lb is None: lb = -np.inf
                if ub is None: ub = np.inf
                bounds.append([lb, ub])
            self.minimize_kwargs.update(
                {'bounds': bounds}
            )

        # run optimize
        try:
            res = minimize(
                fun=self._objective,
                x0=x0,
                **self.minimize_kwargs,
                callback=self.stop_iteration_callback,
            )
        except StopIteration2:
            res = None
            logger.warn(Msg.WARN_INTERRUPTED_IN_SCIPY)

        if res is None:
            self.res = self.stop_iteration_callback.res
        else:
            self.res = res
