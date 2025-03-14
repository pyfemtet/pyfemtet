from __future__ import annotations

from typing import Callable

from v1.history import *
from v1.problem import *
from v1.interface import *
from v1.exceptions import *
from v1.worker_status import *
from v1.variable_manager import *
from v1.logger import get_module_logger

__all__ = [
    'AbstractOptimizer',
    'SubFidelityModel',
    'SubFidelityModels',
]


logger = get_module_logger('opt.optimizer')


def _log_hidden_constraint(e: Exception):
    err_msg = create_err_msg_from_exception(e)
    logger.warning('----- Hidden constraint violation! -----')
    logger.warning(f'エラー: {err_msg}')


class AbstractOptimizer:

    # problem
    variable_manager: VariableManager
    objectives: Objectives
    constraints: Constraints
    fidelity: Fidelity | None
    sub_fidelity_name: str
    sub_fidelity_models: SubFidelityModels | None

    # system
    history: History
    fem: AbstractFEMInterface
    entire_status: WorkerStatus
    worker_status: WorkerStatus

    def __init__(self):

        # Problem
        self.variable_manager = VariableManager()
        self.objectives = Objectives()
        self.constraints = Constraints()

        # multi-fidelity
        self.fidelity = None
        self.sub_fidelity_name = MAIN_FIDELITY_NAME
        self.sub_fidelity_models = None

        # System
        self._fem: AbstractFEMInterface | None = None
        self.history: History = History()
        self.solve_condition: Callable[[History], bool] = lambda _: True
        self.entire_status: WorkerStatus = WorkerStatus(ENTIRE_PROCESS_STATUS_KEY)
        self.worker_status: WorkerStatus = WorkerStatus()

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
        self.variable_manager.variables.update({name: var})

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
            fun: Callable[..., float],
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
            fun: Callable[..., float],
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
        cns._opt = self
        self.constraints.update({name: cns})

    def add_sub_fidelity_model(
            self,
            name: str,
            sub_fidelity_model: SubFidelityModel,
            fidelity: Fidelity,
    ):
        sub_fidelity_model.variable_manager = self.variable_manager
        if self.sub_fidelity_models is None:
            self.sub_fidelity_models = SubFidelityModels()
            self.fidelity = 1.
        self.sub_fidelity_models.add(name, sub_fidelity_model, fidelity)

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

    def _get_hard_constraint_violation_names(self, hard_c: TrialConstraintOutput) -> list[str]:
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

    def _check_and_raise_interruption(self):
        # raise Interrupt
        interrupted = self.entire_status.value == WorkerStatus.interrupting
        if interrupted:
            self.worker_status.value = WorkerStatus.interrupting
            raise InterruptOptimization

    def f(
            self,
            parameters: TrialInput,
            variables_pass_to_fem: TrialInput,
            history: History = None,
            datetime_start=None,
    ) -> tuple[TrialOutput, dict[str, float], TrialConstraintOutput, Record]:

        # create context
        if history is not None:
            record_to_history = history.recording()
        else:
            class DummyRecordContext:
                def __enter__(self):
                    return Record()

                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass

            record_to_history = DummyRecordContext()

        # processing with recording
        with record_to_history as record:

            # record common result
            record.x = parameters
            record.sub_fidelity_name = self.sub_fidelity_name
            record.fidelity = self.fidelity
            record.datetime_start = datetime_start

            # check skip
            if not self._should_solve(history):
                record.state = TrialState.skipped
                raise SkipSolve

            # start solve
            if self.sub_fidelity_models == '':
                logger.info(f'===== Run =====')
            else:
                logger.info(f'===== Run {self.sub_fidelity_name} =====')
            logger.info('入力:')
            logger.info(parameters)

            # ===== update FEM parameter =====
            self.fem.update_parameter(variables_pass_to_fem)
            self._check_and_raise_interruption()

            # ===== evaluate hard constraint =====
            hard_c = TrialConstraintOutput()
            try:
                self._hard_c(hard_c)

            except HiddenConstraintViolation as e:

                _log_hidden_constraint(e)
                record.c = hard_c
                record.message = 'Hidden constraint violation hard constraint evaluation: ' \
                                 + create_err_msg_from_exception(e)

                raise e

            # check hard constraint violation
            violation_names = self._get_hard_constraint_violation_names(hard_c)
            if len(violation_names) > 0:

                record.c = hard_c
                record.state = TrialState.hard_constraint_violation
                record.message = f'Hard constraint violation: ' \
                                 + ', '.join(violation_names)

                raise HardConstraintViolation

            # ===== update FEM =====
            try:
                self.fem.update()
                self._check_and_raise_interruption()

            # if hidden constraint violation
            except HiddenConstraintViolation as e:

                self._check_and_raise_interruption()

                _log_hidden_constraint(e)
                record.c = hard_c
                record.message = 'Hidden constraint violation in FEM update: ' \
                                 + create_err_msg_from_exception(e)

                raise e

            # ===== evaluate y =====
            try:
                y: TrialOutput = self._y()
                self._check_and_raise_interruption()

            # if intentional error (by user)
            except HiddenConstraintViolation as e:
                self._check_and_raise_interruption()

                _log_hidden_constraint(e)
                record.c = hard_c
                record.message = 'Hidden constraint violation during objective function evaluation: ' \
                                 + create_err_msg_from_exception(e)

                raise e

            # ===== evaluate soft constraint =====
            soft_c = TrialConstraintOutput()
            try:
                self._soft_c(soft_c)

            # if intentional error (by user)
            except HiddenConstraintViolation as e:
                _log_hidden_constraint(e)

                _c = {}
                _c.update(soft_c)
                _c.update(hard_c)

                record.y = y
                record.c = _c
                record.message = 'Hidden constraint violation during soft constraint function evaluation: ' \
                                 + create_err_msg_from_exception(e)

                raise e

            # ===== merge and sort constraints =====
            c = TrialConstraintOutput()
            c.update(soft_c)
            c.update(hard_c)

            # get values as minimize
            y_internal: dict = self._convert_y(y)

            logger.info('出力:')
            logger.info(y)
            record.y = y
            record.c = c
            record.state = TrialState.succeeded

            self._check_and_raise_interruption()

        return y, y_internal, c, record

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

        class SettingStatusFinished:

            # noinspection PyMethodParameters
            def __enter__(self_):
                pass

            # noinspection PyMethodParameters
            def __exit__(self_, exc_type, exc_val, exc_tb):
                if exc_type is not None:
                    self.worker_status.value = WorkerStatus.crashed

                    import sys
                    from traceback import print_tb
                    print(f'===== Exception raised in worker {worker_idx} =====', file=sys.stderr)
                    print_tb(exc_tb)
                    print(f'{exc_type.__name__}: {exc_val}', file=sys.stderr)

                else:
                    self.worker_status.value = WorkerStatus.finished

        logger.info(f'worker {worker_idx} started')

        with SettingStatusFinished():

            self.worker_status.value = WorkerStatus.initializing

            self.worker_status.value = WorkerStatus.launching_fem
            self.fem._setup_after_parallel()

            self.worker_status.value = WorkerStatus.waiting
            pass

            self.worker_status.value = WorkerStatus.running

            self.run()

        logger.info(f'worker {worker_idx} successfully finished!')

    def _logging(self):

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
        parameters = self.variable_manager.get_variables(
            filter='parameter', format='raw'
        )
        self.history.finalize(
            parameters,
            list(self.objectives.keys()),
            list(self.constraints.keys()),
        )


class SubFidelityModel(AbstractOptimizer): ...


class SubFidelityModels(dict[str, SubFidelityModel]):

    def add(self, name, model: SubFidelityModel, fidelity: Fidelity):
        model.sub_fidelity_name = name
        model.fidelity = fidelity
        self.update({name: model})
