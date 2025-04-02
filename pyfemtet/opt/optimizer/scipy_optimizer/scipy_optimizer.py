from __future__ import annotations

from typing import Callable

import datetime
from contextlib import suppress

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from scipy.optimize import NonlinearConstraint

from pyfemtet._util.closing import closing
from pyfemtet.opt.worker_status import *
from pyfemtet.opt.problem import *
from pyfemtet.opt.variable_manager import *
from pyfemtet.opt.exceptions import *
from pyfemtet.logger import get_module_logger

from pyfemtet.opt.optimizer.optimizer import AbstractOptimizer, SubFidelityModels


__all__ = [
    'ScipyOptimizer',
]


logger = get_module_logger('opt.optimizer', False)


class _ScipyCallback:

    def __init__(self, opt: ScipyOptimizer):
        self.opt = opt

    def __call__(self, xk: np.ndarray = None, intermediate_result: OptimizeResult = None):
        pass


class ScipyOptimizer(AbstractOptimizer):

    def __init__(self):
        super().__init__()

        self.method = None
        self.tol = None
        self.options = {}
        self.constraint_enhancement = 0.001

    def _get_x0(self) -> np.ndarray:

        # params を取得
        params: dict[str, Parameter] = self.variable_manager.get_variables(
            filter='parameter', format='raw'
        )

        # params のうち fix == True のものを除く
        x0 = np.array([p.value for p in params.values() if not p.properties.get('fix', False)])

        return x0

    def _warn_bounds_for_nelder_mead(self) -> None:
        # https://github.com/scipy/scipy/issues/19991

        if self.method.lower() != 'nelder-mead':
            return

        bounds = self._get_scipy_bounds()
        if bounds is None:
            return

        x0 = self._get_x0()
        if (np.allclose(x0, bounds[:, 0])
                or np.allclose(x0, bounds[:, 1])):
            logger.warning(
                'Nelder-Mead で初期値が境界上端または下端と'
                '一致していると最適化が進まない場合があります。')

    def _setup_before_parallel(self):

        if not self._done_setup_before_parallel:

            super()._setup_before_parallel()  # flag inside

            self._warn_bounds_for_nelder_mead()

    def _get_scipy_bounds(self) -> np.ndarray | None:

        has_any_bound = False

        params: dict[str, Parameter] = self.variable_manager.get_variables(filter='parameter')
        bounds = []
        for param in params.values():
            assert isinstance(param, NumericParameter)
            bounds.append([
                param.lower_bound or -np.inf,
                param.upper_bound or np.inf,
            ])
            has_any_bound += (
                    (param.lower_bound is not None)
                    or (param.upper_bound is not None))

        if has_any_bound:
            bounds = np.array(bounds)
        else:
            bounds = None

        return bounds

    def _update_vm_by_xk(self, xk):

        vm = self.variable_manager

        # check interruption
        self._check_and_raise_interruption()

        # parameter suggestion
        params = vm.get_variables(filter='parameter')
        xk_list = list(xk)
        for name, prm in params.items():

            if prm.properties.get('fix', False):  # default is False
                continue

            if isinstance(prm, NumericParameter):
                prm.value = xk_list.pop(0)

            elif isinstance(prm, CategoricalParameter):
                raise NotImplementedError('ScipyOptimizer は CategoricalParameter 未対応です。')

            else:
                raise NotImplementedError
        assert len(xk_list) == 0

        # evaluate expressions
        vm.eval_expressions()

        # check interruption
        self._check_and_raise_interruption()

    def _scipy_constraint_fun(self, xk, cns: Constraint):

        self._update_vm_by_xk(xk)

        # update fem (very slow!)
        if cns.using_fem:
            logger.warning(
                'constraint で FEM の API に'
                'アクセスするのはとても遅くなります。'
                '許容する場合のみ実行してください。')
            pass_to_fem = self.variable_manager.get_variables(filter='pass_to_fem')
            self.fem.update_parameter(pass_to_fem)

        return cns.eval(self.fem)

    def _get_scipy_constraints(self) -> (
        None
        | list[NonlinearConstraint | dict]
    ):
        if len(self.constraints) == 0:
            return None

        if self.method is None:
            method = 'SLSQP'
        else:
            method = self.method
        assert method.lower() in ('cobyla', 'cobyqa', 'slsqp', 'trust-constr')

        out = []
        for cns in self.constraints.values():

            # use Constraint object
            if method.lower() in ('trust-constr', 'cobyqa'):

                if cns.hard:
                    raise NotImplementedError(
                        f'{method} では hard constraint を扱えません。'
                    )

                scipy_cns = NonlinearConstraint(
                    fun=lambda xk_, cns_=cns: self._scipy_constraint_fun(xk_, cns_),
                    lb=(cns.lower_bound or -np.inf) + self.constraint_enhancement,
                    ub=(cns.upper_bound or np.inf) - self.constraint_enhancement,
                    keep_feasible=cns.hard,  # doesn't work??
                    finite_diff_rel_step=self.options.get('finite_diff_rel_step', None),
                )
                out.append(scipy_cns)

            # use dict object
            else:

                if method.lower() == 'slsqp' and not cns.hard:
                    logger.warning(
                        'SLSQP 法では soft constraint を扱えません。'
                        'hard constraint として扱います。')

                if method.lower() == 'cobyla' and cns.hard:
                    logger.error(
                        f'{method} では hard constraint を扱えません。')
                    raise NotImplementedError(
                        f'{method} では hard constraint を扱えません。')

                if cns.lower_bound is not None:

                    scipy_cns = dict(
                        type='ineq',
                        fun=(lambda xk_, cns_=cns:
                             self._scipy_constraint_fun(xk_, cns_)
                             - cns_.lower_bound
                             - self.constraint_enhancement),
                    )
                    out.append(scipy_cns)

                if cns.upper_bound is not None:
                    scipy_cns = dict(
                        type='ineq',
                        fun=(lambda xk_, cns_=cns:
                             cns_.upper_bound
                             - self._scipy_constraint_fun(xk_, cns_)
                             - self.constraint_enhancement),
                    )
                    out.append(scipy_cns)

        return out

    def _get_scipy_callback(self) -> (
        Callable[[OptimizeResult, ...], ...]
        | Callable[[np.ndarray, ...], ...]
    ):
        return _ScipyCallback(self)

    def _solve(
            self,
            x: TrialInput,
            x_pass_to_fem_: dict[str, SupportedVariableTypes],
            opt_: AbstractOptimizer = None,
    ) -> float:

        opt_ = opt_ or self
        vm = self.variable_manager

        # check interruption
        self._check_and_raise_interruption()

        # declare output
        y_internal_: float

        # if opt_ is not self, update variable manager
        opt_.variable_manager = vm

        # start solve
        datetime_start = datetime.datetime.now()
        try:
            y, dict_y_internal, c, record = opt_.f(
                x, x_pass_to_fem_, self.history, datetime_start
            )

        # if hidden constraint violation, raise it
        except _HiddenConstraintViolation as e:
            raise NotImplementedError(
                'ScipyOptimizer では解析ができない'
                '設計変数の組合せをスキップできません。'
            ) from e

        # if skipped
        except SkipSolve as e:
            raise NotImplementedError(
                'ScipyOptimizer では Skip はできません。'
            ) from e

        # if succeeded
        else:

            y_internal_ = tuple(dict_y_internal.values())[0]  # type: ignore

        # check interruption
        self._check_and_raise_interruption()

        return y_internal_

    def _objective(self, xk: np.ndarray) -> float:

        with self._logging():

            vm = self.variable_manager

            # parameter suggestion
            self._update_vm_by_xk(xk)

            # construct TrialInput
            x = vm.get_variables(filter='parameter')
            x_pass_to_fem: dict[str, SupportedVariableTypes] = vm.get_variables(filter='pass_to_fem', format='dict')

            # process main fidelity model
            y_internal: float = self._solve(x, x_pass_to_fem)

            return y_internal

    def run(self):

        # ===== finalize =====
        self._finalize()

        # ===== construct x0 =====
        x0 = self._get_x0()

        # ===== run =====
        with closing(self.fem):

            try:

                with suppress(InterruptOptimization):

                    minimize(
                        self._objective,
                        x0,
                        args=(),
                        method=self.method,
                        bounds=self._get_scipy_bounds(),
                        constraints=self._get_scipy_constraints(),
                        tol=self.tol,
                        callback=self._get_scipy_callback(),
                        options=self.options,
                    )

            except Exception as e:
                if self.worker_status.value < WorkerStatus.crashed:
                    self.worker_status.value = WorkerStatus.crashed
                raise e
            else:
                if self.worker_status.value < WorkerStatus.finishing:
                    self.worker_status.value = WorkerStatus.finishing
