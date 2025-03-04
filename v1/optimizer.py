from __future__ import annotations
from typing import Callable

import datetime
from contextlib import suppress, closing

import numpy as np

from v1.variable_manager import *
from v1.problem import *
from v1.interface import *
from v1.history import *
from v1.worker_status import *
from v1.logger import get_module_logger

logger = get_module_logger('opt.optimizer')


class ConstraintViolation(Exception):
    pass


class Interrupt(Exception):
    pass


class AbstractOptimizer:

    # problem
    variable_manager: VariableManager
    objectives: Objectives
    constraints: Constraints
    fidelity: Fidelity
    sub_fidelity_models: SubFidelityModels

    # system
    history: History
    fem: AbstractFEMInterface
    current_trial_start: datetime.datetime | None
    entire_status: WorkerStatus
    worker_status: WorkerStatus

    def __init__(self):

        # Problem
        self.variable_manager = VariableManager()
        self.objectives = Objectives()
        self.constraints = Constraints()

        # multi-fidelity
        self.fidelity: Fidelity = None
        self.sub_fidelity_models: SubFidelityModels = None

        # System
        self._fem: AbstractFEMInterface = None
        self.history: History = History()
        self.solve_condition: Callable[[History], bool] = lambda _: True
        self.entire_status: WorkerStatus = WorkerStatus(entire_process_status_key)
        self.worker_status: WorkerStatus = WorkerStatus()

        # Util
        self.current_trial_start = None

    def add_variable(
            self,
            name: str,
            value: float,
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
    ):
        var = NumericVariable()
        var.name = name
        var.value = value
        var.pass_to_fem = pass_to_fem
        var.properties = properties if properties is not None else {}
        self.variable_manager.variables.update({name, var})

    def add_parameter(
            self,
            name: str,
            initial_value: float | None,
            lower_bound: float | None,
            upper_bound: float | None,
            step: float | None = None,
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
            fix: bool = False,
    ) -> None:
        properties = properties if properties is not None else {}

        if fix:
            properties.update({'fix': True})

        prm = NumericParameter()
        prm.name = name
        prm.value = initial_value
        prm.lower_bound = lower_bound
        prm.upper_bound = upper_bound
        prm.step = step
        prm.properties = properties
        prm.pass_to_fem = pass_to_fem
        self.variable_manager.variables.update({name: prm})

    def add_categorical_parameter(
            self,
            name: str,
            initial_value: str | None,
            choices: list[str] | None,
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
            fix: bool = False,
    ):
        properties = properties if properties is not None else {}

        if fix:
            properties.update({'fix': True})

        prm = CategoricalParameter()
        prm.name = name
        prm.value = initial_value
        prm.choices = choices
        prm.properties = properties
        prm.pass_to_fem = pass_to_fem
        self.variable_manager.variables.update({name: prm})

    def add_objective(
            self,
            name: str,
            fun: Callable[[...], float],
            direction: str | float = 'minimize',
            args: tuple | None = None,
            kwargs: dict | None = None,
    ) -> None:
        obj = Objective()
        obj.fun = fun
        obj.args = args or ()
        obj.kwargs = kwargs or {}
        obj.direction = direction
        self.objectives.update({name: obj})

    def add_constraint(
            self,
            name: str,
            fun: Callable[[...], float],
            lower_bound: float | None = None,
            upper_bound: float | None = None,
            args: tuple | None = None,
            kwargs: dict | None = None,
            strict: bool = True,
    ):
        cns = Constraint()
        cns.fun = fun
        cns.args = args or ()
        cns.kwargs = kwargs or {}
        cns.lower_bound = lower_bound
        cns.upper_bound = upper_bound
        cns.hard = strict
        self.constraints.update({name: cns})

    def add_sub_fidelity_model(
            self,
            name: str,
            sub_fidelity_model: SubFidelityModel
    ):
        sub_fidelity_model.variable_manager = self.variable_manager
        d = {name: sub_fidelity_model}
        if self.sub_fidelity_models is None:
            self.sub_fidelity_models = d
        else:
            self.sub_fidelity_models.update(d)

    def get_variables(self, format='dict'):
        return self.variable_manager.get_variables(
            format=format,
        )

    def _should_solve(self, history):
        return self.solve_condition(history)

    def set_solve_condition(self, fun: Callable[[History], bool]):
        self.solve_condition = fun

    def _y(self) -> TrialOutput:
        out = TrialOutput()
        for name, obj in self.objectives.items():
            obj_result = ObjectiveResult(obj)
            out.update({name: obj_result})
        return out

    def _convert_y(self, y: TrialOutput) -> dict:
        out = dict()
        for name, result in y.items():
            obj = self.objectives[name]
            value_internal = obj.convert(result.value)
            out.update({name: value_internal})
        return out

    def _hard_c(self, out: TrialConstraintOutput) -> TrialConstraintOutput:
        for name, cns in self.constraints.items():
            if cns.hard:
                cns_result = ConstraintResult(cns)
                out.update({name: cns_result})
        return out

    def _soft_c(self, out: TrialConstraintOutput) -> TrialConstraintOutput:
        for name, cns in self.constraints.items():
            if not cns.hard:
                cns_result = ConstraintResult(cns)
                out.update({name: cns_result})
        return out

    def _process_hidden_constraint(self, e: Exception):
        err_msg = create_err_msg_from_exception(e)
        logger.warning('----- Hidden constraint violation! -----')
        logger.warning(f'エラー: {err_msg}')

    def _process_hard_constraint(self, hard_c: TrialConstraintOutput) -> list[str]:
        violation_names = []
        for name, result in hard_c.items():
            cns = self.constraints[name]
            l_or_u = result.check_violation()
            if l_or_u is not None:
                logger.warning('----- Hard constraint violation! -----')
                logger.warning(f'拘束名: {name}')
                logger.warning(f'計算値: {result.value}')
                if l_or_u == 'lower_bound':
                    logger.info(f'下限値: {cns.lower_bound}')
                    violation_names.append(name + '_' + l_or_u)
                elif l_or_u == 'upper_bound':
                    logger.info(f'上限値: {cns.upper_bound}')
                    violation_names.append(name + '_' + l_or_u)
                else:
                    raise NotImplementedError
        return violation_names

    def _check_entire_interruption(self):
        # raise Interrupt
        interrupted = self.entire_status.value == WorkerStatus.interrupting
        if interrupted:
            self.worker_status.value = WorkerStatus.interrupting
            raise Interrupt

    def f(
            self,
            parameters: TrialInput,
            variables_pass_to_fem: TrialInput,
            history: History = None,
            datetime_start=None,
    ) -> tuple[TrialOutput, dict[str, float], TrialConstraintOutput]:

        logger.info('入力:')
        logger.info(parameters)

        # update FEM parameter
        self.fem.update_parameter(variables_pass_to_fem)
        self._check_entire_interruption()

        # evaluate hard constraint
        hard_c = TrialConstraintOutput()
        try:
            self._hard_c(hard_c)

        except FEMError as e:

            self._process_hidden_constraint(e)
            if history is not None:
                if isinstance(e, ModelError):
                    state = TrialState.model_error
                elif isinstance(e, MeshError):
                    state = TrialState.mesh_error
                elif isinstance(e, SolveError):
                    state = TrialState.solve_error
                elif isinstance(e, PostProcessError):
                    state = TrialState.post_error
                else:
                    state = TrialState.unknown_error
                history.record(
                    x=parameters,
                    c=hard_c,
                    state=state,
                    fidelity=self.fidelity,
                    datetime_start=datetime_start,
                    message='Hidden constraint violation hard constraint evaluation: ' + create_err_msg_from_exception(e),
                )

            raise e

        # check hard constraint violation
        violation_names = self._process_hard_constraint(hard_c)
        if len(violation_names) > 0:

            if history is not None:
                history.record(
                    x=parameters,
                    c=hard_c,
                    state=TrialState.hard_constraint_violation,
                    fidelity=self.fidelity,
                    datetime_start=datetime_start,
                    message=f'Hard constraint violation during hard constraint evaluation: ' + ', '.join(violation_names),
                )

            raise ConstraintViolation

        # update FEM
        try:
            self.fem.update()
            self._check_entire_interruption()

        # if hidden constraint violation
        except FEMError as e:

            self._check_entire_interruption()

            self._process_hidden_constraint(e)

            if history is not None:
                if isinstance(e, ModelError):
                    state = TrialState.model_error
                elif isinstance(e, MeshError):
                    state = TrialState.mesh_error
                elif isinstance(e, SolveError):
                    state = TrialState.solve_error
                elif isinstance(e, PostProcessError):
                    state = TrialState.post_error
                else:
                    state = TrialState.unknown_error
                history.record(
                    x=parameters,
                    c=hard_c,
                    state=state,
                    fidelity=self.fidelity,
                    datetime_start=datetime_start,
                    message='Hidden constraint violation in FEM update: ' + create_err_msg_from_exception(e),
                )

            raise e

        # evaluate y
        try:
            y: TrialOutput = self._y()
            self._check_entire_interruption()

        # if intentional error (by user)
        except FEMError as e:
            self._check_entire_interruption()

            self._process_hidden_constraint(e)

            if history is not None:
                if isinstance(e, ModelError):
                    state = TrialState.model_error
                elif isinstance(e, MeshError):
                    state = TrialState.mesh_error
                elif isinstance(e, SolveError):
                    state = TrialState.solve_error
                elif isinstance(e, PostProcessError):
                    state = TrialState.post_error
                else:
                    state = TrialState.unknown_error
                history.record(
                    x=parameters,
                    c=hard_c,
                    state=state,
                    fidelity=self.fidelity,
                    datetime_start=datetime_start,
                    message='Hidden constraint violation during objective function evaluation: ' + create_err_msg_from_exception(e),
                )

            raise e

        # evaluate soft constraint
        soft_c = TrialConstraintOutput()
        try:
            self._soft_c(soft_c)

        # if intentional error (by user)
        except FEMError as e:
            self._process_hidden_constraint(e)

            if history is not None:
                _c = {}
                _c.update(soft_c)
                _c.update(hard_c)
                history.record(
                    x=parameters,
                    y=y,
                    c=_c,
                    state=TrialState.post_error,
                    fidelity=self.fidelity,
                    datetime_start=datetime_start,
                    message='Hidden constraint violation during soft constraint function evaluation: ' + create_err_msg_from_exception(e),
                )

            raise e

        # merge and sort constraints to follow self.constraints
        c = TrialConstraintOutput()
        c.update(soft_c)
        c.update(hard_c)

        # get values as minimize
        y_internal: TrialOutput = self._convert_y(y)

        logger.info('出力:')
        logger.info(y)

        if history is not None:
            history.record(
                x=parameters,
                y=y,
                c=c,
                state=TrialState.succeeded,
                fidelity=self.fidelity,
                datetime_start=datetime_start,
            )

        self._check_entire_interruption()

        return y, y_internal, c

    def run(self) -> None:
        raise NotImplementedError

    def _run(
            self,
            worker_idx: int | str,
            history: History,
            entire_status: WorkerStatus,
            worker_status: WorkerStatus,
    ) -> None:

        self.history = history
        self.entire_status = entire_status
        self.worker_status = worker_status

        class SetStatus:

            # noinspection PyMethodParameters
            def __enter__(self_):
                self.worker_status.value = WorkerStatus.running

            # noinspection PyMethodParameters
            def __exit__(self_, exc_type, exc_val, exc_tb):
                if exc_type is not None:
                    self.worker_status.value = WorkerStatus.crashed
                else:
                    self.worker_status.value = WorkerStatus.finished

        logger.info(f'worker {worker_idx}')

        self.worker_status.value = WorkerStatus.initializing

        self.worker_status.value = WorkerStatus.launching_fem
        self.fem._setup_after_parallel()

        self.worker_status.value = WorkerStatus.waiting

        with SetStatus():
            self.run()

        logger.info(f'worker {worker_idx} complete!')

    def _logging_output(self):

        class LoggingOutput:

            def __enter__(self):
                logger.info('===== trial start =====')

            def __exit__(self, exc_type, exc_val, exc_tb):
                logger.info('===== trial end =====\n')

        return LoggingOutput()

    @property
    def fem(self) -> AbstractFEMInterface:
        return self._fem

    @fem.setter
    def fem(self, value: AbstractFEMInterface):
        self._fem = value
        if self._fem._load_problem_from_fem:
            self._load_problem_from_fem()

    def _load_problem_from_fem(self):
        self.fem.load_variables(self)
        self.fem.load_objectives(self)
        self.fem.load_constraints(self)

    def _finalize_history(self):
        self.history.finalize(
            list(self.variable_manager.get_variables(filter='parameter', format='dict').keys()),
            list(self.objectives.keys()),
            list(self.constraints.keys()),
        )


class SubFidelityModel(AbstractOptimizer): ...


class SubFidelityModels(dict[str, SubFidelityModel]): ...


import warnings

import optuna

from v1.logger import get_optuna_logger, remove_all_output

remove_all_output(get_optuna_logger())

warnings.filterwarnings('ignore', 'set_metric_names', optuna.exceptions.ExperimentalWarning)

CONSTRAINT_ATTR_KEY = 'constraint'


class OptunaOptimizer(AbstractOptimizer):

    current_trial: optuna.trial.Trial

    def _create_infeasible_constraints(self) -> np.ndarray:
        count = 0
        for name, cns in self.constraints.items():
            if cns.lower_bound is not None:
                count += 1
            if cns.upper_bound is not None:
                count += 1
        return np.ones(count, dtype=np.float64)

    def _set_constraint(self, trial: optuna.trial.Trial, cns_values: tuple[float] | None = None):

        if cns_values is None:
            cns_values = self._create_infeasible_constraints()

        trial.set_user_attr(
            CONSTRAINT_ATTR_KEY,
            cns_values
        )

    def _get_constraint(self, trial: optuna.trial.FrozenTrial):
        return trial.user_attrs[CONSTRAINT_ATTR_KEY]

    def _objective(self, trial: optuna.trial.Trial):

        self.current_trial = trial

        with self._logging_output():

            vm = self.variable_manager

            # check interruption
            self._check_entire_interruption()

            # parameter suggestion
            params = vm.get_variables(filter='parameter')
            for name, prm in params.items():

                if prm.properties.get('fix', False):  # default is False
                    continue

                if isinstance(prm, NumericParameter):
                    prm.value = trial.suggest_float(
                        name,
                        prm.lower_bound,
                        prm.upper_bound,
                        step=prm.step,
                        log=prm.properties.get('log', __default := False),
                    )
                elif isinstance(prm, CategoricalParameter):
                    prm.value = trial.suggest_categorical(
                        name, prm.choices
                    )
                else:
                    raise NotImplementedError

            # evaluate expressions
            vm.evaluate()

            # check interruption
            self._check_entire_interruption()

            # construct TrialInput
            x = vm.get_variables(filter='parameter')
            x_pass_to_fem: TrialInput = vm.get_variables(filter='pass_to_fem')

            # main
            datetime_start = datetime.datetime.now()
            y, y_internal, c = None, None, None
            if self._should_solve(self.history):
                try:
                    y, y_internal, c = self.f(x, x_pass_to_fem, self.history, datetime_start)

                except (ConstraintViolation, FEMError):
                    self._set_constraint(trial)
                    raise optuna.TrialPruned

                else:
                    # convert constraint to **sorted** violation
                    assert len(c) == len(self.constraints)
                    v = {}
                    for name, cns in self.constraints.items():
                        # This is {lower or upper: violation_value} dict
                        violation: dict[str, float] = c[name].calc_violation()
                        for l_or_u, violation_value in violation.items():
                            key = name + '_' + l_or_u
                            v.update({key: violation_value})

                    # register constraint
                    self._set_constraint(trial, tuple(v.values()))

            # check interruption
            self._check_entire_interruption()

            # sub-fidelity of f
            for name, sub_opt in self.sub_fidelity_models.items():

                if sub_opt._should_solve(self.history):

                    sub_opt.variable_manager = vm
                    datetime_start = datetime.datetime.now()

                    try:
                        sub_y, sub_y_internal, _ = sub_opt.f(x, self.history, datetime_start)

                    except (ConstraintViolation, FEMError):

                        # register sub-fidelity values
                        trial.set_user_attr(
                            'sub_fidelity_' + name,
                            dict(
                                fidelity=sub_opt.fidelity,
                                y=tuple(sub_y_internal.values()),
                            )
                        )

            # check interruption
            self._check_entire_interruption()

            self.current_trial = None

            # if main skipped
            if y is None:
                return None  # set trial FAILED

            # if main solved
            else:
                return tuple(y_internal.values())

    def run(self):

        # quit FEM even if abnormal termination
        with closing(self.fem):

            # sub fidelity
            if self.sub_fidelity_models is None:
                self.sub_fidelity_models = SubFidelityModels()

            # finalize
            self._finalize_history()

            # optuna
            study = optuna.create_study(
                directions=['minimize'] * len(self.objectives)
            )

            study.set_metric_names(list(self.objectives.keys()))

            with suppress(Interrupt):
                study.optimize(
                    self._objective,
                    n_trials=20,
                )

    def _check_entire_interruption(self):
        try:
            AbstractOptimizer._check_entire_interruption(self)
        except Interrupt as e:
            if self.current_trial is not None:
                self.current_trial.study.stop()
            raise e


if __name__ == '__main__':

    def _parabola(_fem: AbstractFEMInterface, _opt: AbstractOptimizer):
        x = _opt.get_variables('values')
        return (x ** 2).sum()

    def _cns(_fem: AbstractFEMInterface, _opt: AbstractOptimizer):
        x = _opt.get_variables('values')
        return x[0]

    _opt = OptunaOptimizer()

    _fem = NoFEM()
    _opt.fem = _fem

    _args = (_fem, _opt)

    _opt.add_parameter('x1', 1, -1, 1, step=0.5)
    _opt.add_parameter('x2', 1, -1, 1, step=0.5)

    _opt.add_constraint('cns', _cns, lower_bound=0, args=_args)

    _opt.add_objective('obj', _parabola, args=_args)

    _opt.run()
