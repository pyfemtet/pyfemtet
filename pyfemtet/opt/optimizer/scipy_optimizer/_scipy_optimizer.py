from __future__ import annotations

from typing import Callable

from contextlib import suppress

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from scipy.optimize import NonlinearConstraint

from pyfemtet._i18n import Msg, _
from pyfemtet._util.closing import closing
from pyfemtet.opt.problem.variable_manager import *
from pyfemtet.opt.problem.problem import *
from pyfemtet.opt.exceptions import *
from pyfemtet.logger import get_module_logger

from pyfemtet.opt.optimizer._base_optimizer import *


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
    """
    Optimizer class that utilizes SciPy optimization methods.

    This class serves as a wrapper around SciPy's optimization routines,
    allowing customization of the optimization method, tolerance, and options.
    It also provides mechanisms for handling constraints with enhancement and scaling.

    Attributes:
        method (str): The optimization method to use (e.g., 'BFGS', 'Nelder-Mead').
        tol (float or None): Tolerance for termination.
        options (dict): Additional options to pass to the SciPy optimizer.
        constraint_enhancement (float): Small value added to enhance constraint handling.
        constraint_scaling (float): Scaling factor applied to constraints.

    Args:
        method (str): The optimization method to use (e.g., 'BFGS', 'Nelder-Mead').
        tol (float or None): Tolerance for termination.
        
    """

    _timeout: None = None
    _n_trials: None = None

    def __init__(self, method: str = None, tol=None):
        super().__init__()

        self.method = method
        self.tol = tol
        self.options = {}
        self.constraint_enhancement = 0.001
        self.constraint_scaling = 1.

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        if value is not None:
            raise NotImplementedError(_(
                en_message='`ScipyOptimizer` cannot use timeout.',
                jp_message='`ScipyOptimizer` では timeout は指定できません。'
            ))

    @property
    def n_trials(self):
        return self._n_trials

    @n_trials.setter
    def n_trials(self, value):
        if value is not None:
            raise NotImplementedError(_(
                en_message='`ScipyOptimizer` cannot use n_trials.',
                jp_message='`ScipyOptimizer` では n_trials は指定できません。'
            ))

    def add_trial(self, parameters: dict[str, SupportedVariableTypes]):
        raise NotImplementedError(_(
            en_message='You cannot use `add_trial()` in `ScipyOptimizer`.',
            jp_message='`ScipyOptimizer` では `add_trial()` は使えません。',
        ))

    def _get_x0(self) -> np.ndarray:

        # params を取得
        params: dict[str, Parameter] = self.variable_manager.get_variables(
            filter='parameter', format='raw'
        )

        for param in params.values():
            if isinstance(param, CategoricalVariable):
                raise NotImplementedError(_(
                    en_message='Scipy can optimize only numerical parameters.',
                    jp_message='Scipy では数値パラメータのみ最適化できます。'
                ))

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
            logger.warning(Msg.WARN_SCIPY_NELDER_MEAD_BOUND)

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
                raise NotImplementedError(Msg.ERR_SCIPY_NOT_IMPLEMENT_CATEGORICAL)

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
            logger.warning(Msg.WARN_USING_FEM_IN_NLC)
            pass_to_fem = self.variable_manager.get_variables(
                filter='pass_to_fem', format='raw'
            )
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
                        Msg.F_ERR_SCIPY_METHOD_NOT_IMPLEMENT_HARD_CONSTRAINT(
                            method
                        )
                    )

                # constraint_scaling を使うためには violation を計算しなければならない
                # TODO: 上下両端が決められている場合は二回計算することになるのでそれを解消する
                if cns.lower_bound is not None:
                    scipy_cns = NonlinearConstraint(
                        fun=(
                            lambda xk_, cns_=cns:
                                (
                                    cns.lower_bound
                                    - self._scipy_constraint_fun(xk_, cns_)
                                ) * self.constraint_scaling
                                + self.constraint_enhancement
                        ),
                        lb=-np.inf,
                        ub=0,
                        keep_feasible=cns.hard,
                        finite_diff_rel_step=self.options.get('finite_diff_rel_step', None),
                    )
                    out.append(scipy_cns)
                if cns.upper_bound is not None:
                    scipy_cns = NonlinearConstraint(
                        fun=(
                            lambda xk_, cns_=cns:
                                (
                                    self._scipy_constraint_fun(xk_, cns_)
                                    - cns.upper_bound
                                ) * self.constraint_scaling
                                + self.constraint_enhancement
                        ),
                        lb=-np.inf,
                        ub=0,
                        keep_feasible=cns.hard,
                        finite_diff_rel_step=self.options.get('finite_diff_rel_step', None),
                    )
                    out.append(scipy_cns)

            # scipy_cns = NonlinearConstraint(
                #     fun=lambda xk_, cns_=cns: self._scipy_constraint_fun(xk_, cns_),
                #     lb=(cns.lower_bound or -np.inf) + self.constraint_enhancement,
                #     ub=(cns.upper_bound or np.inf) - self.constraint_enhancement,
                #     keep_feasible=cns.hard,
                #     finite_diff_rel_step=self.options.get('finite_diff_rel_step', None),
                # )
                # out.append(scipy_cns)

            # use dict object
            else:

                if method.lower() == 'slsqp' and not cns.hard:
                    logger.warning(Msg.WARN_SCIPY_SLSQP_CANNOT_PROCESS_SOFT_CONSTRAINT)

                if method.lower() == 'cobyla' and cns.hard:
                    logger.error(
                        Msg.F_ERR_SCIPY_METHOD_NOT_IMPLEMENT_HARD_CONSTRAINT(
                            method))
                    raise NotImplementedError(
                        Msg.F_ERR_SCIPY_METHOD_NOT_IMPLEMENT_HARD_CONSTRAINT(
                            method))

                if cns.lower_bound is not None:

                    scipy_cns = dict(
                        type='ineq',
                        fun=(lambda xk_, cns_=cns:
                             (
                                 self._scipy_constraint_fun(xk_, cns_)
                                 - cns_.lower_bound
                             ) * self.constraint_scaling
                             - self.constraint_enhancement),
                    )
                    out.append(scipy_cns)

                if cns.upper_bound is not None:
                    scipy_cns = dict(
                        type='ineq',
                        fun=(lambda xk_, cns_=cns:
                             (
                                 cns_.upper_bound
                                 - self._scipy_constraint_fun(xk_, cns_)
                             ) * self.constraint_scaling
                             - self.constraint_enhancement),
                    )
                    out.append(scipy_cns)

        return out

    def _get_scipy_callback(self) -> (
        Callable[[OptimizeResult, ...], ...]
        | Callable[[np.ndarray, ...], ...]
    ):
        return _ScipyCallback(self)

    class _SolveSet(AbstractOptimizer._SolveSet):

        def _hard_constraint_handling(self, e: HardConstraintViolation):
            raise NotImplementedError(
                Msg.ERR_SCIPY_HARD_CONSTRAINT_VIOLATION
            ) from e

        def _hidden_constraint_handling(self, e: _HiddenConstraintViolation):
            raise NotImplementedError(
                Msg.ERR_SCIPY_HIDDEN_CONSTRAINT
            ) from e

        def _skip_handling(self, e: SkipSolve):
            raise NotImplementedError(
                Msg.ERR_SCIPY_NOT_IMPLEMENT_SKIP
            ) from e

    def _objective(self, xk: np.ndarray) -> float:

        with self._logging():

            vm = self.variable_manager

            # parameter suggestion
            self._update_vm_by_xk(xk)

            # construct TrialInput
            x = vm.get_variables(filter='parameter')

            # process main fidelity model
            solve_set = self._get_solve_set()
            f_return = solve_set.solve(x)
            assert f_return is not None
            dict_y_internal = f_return[1]
            y_internal: float = tuple(dict_y_internal.values())[0]  # type: ignore

            return y_internal

    def run(self):

        # ===== finalize =====
        self._finalize()

        # ===== construct x0 =====
        x0 = self._get_x0()

        # ===== run =====
        with closing(self.fem):

            with self._setting_status(), suppress(InterruptOptimization):

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
