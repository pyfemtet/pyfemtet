from __future__ import annotations

import datetime
from typing import Callable, TypeAlias, Sequence, Literal
from numbers import Real  # マイナーなので型ヒントでは使わず、isinstance で使う
from time import sleep
import os

import sympy

from pyfemtet._i18n.messages import _
from pyfemtet.opt.history import *
from pyfemtet.opt.interface import *
from pyfemtet.opt.exceptions import *
from pyfemtet.opt.worker_status import *
from pyfemtet.opt.problem.problem import *
from pyfemtet.opt.problem.variable_manager import *
from pyfemtet.opt.optimizer._trial_queue import *
from pyfemtet.logger import get_module_logger

__all__ = [
    'AbstractOptimizer',
    'SubFidelityModel',
    'SubFidelityModels',
    '_FReturnValue',
]


DIRECTION: TypeAlias = (
        float
        | Literal[
            'minimize',
            'maximize',
        ]
)

logger = get_module_logger('opt.optimizer', False)


def _log_hidden_constraint(e: Exception):
    err_msg = create_err_msg_from_exception(e)
    logger.warning(_('----- Hidden constraint violation! -----'))
    logger.warning(_('error: {err_msg}', err_msg=err_msg))


_FReturnValue: TypeAlias = tuple[TrialOutput, dict[str, float], TrialConstraintOutput, Record]


def _duplicated_name_check(name, names):
    if name in names:
        logger.warning(
            _(
                en_message='There are duplicated name {name}. If there are '
                           'duplicate names for parameters or objective '
                           'functions, the later defined ones will overwrite '
                           'the earlier ones. Please be careful to ensure '
                           'that this overwriting is intentional.',
                jp_message='名前 {name} が重複しています。パラメータや目的関数'
                           'などの名前が重複すると、後から定義したものが上書き'
                           'されます。この上書きが意図したものであるかどうか、'
                           '十分に注意してください。',
                name=name,
            )
        )


class AbstractOptimizer:

    # optimize
    n_trials: int | None
    timeout: float | None
    seed: int | None
    include_queued_in_n_trials: bool

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
    worker_status_list: list[WorkerStatus]
    _done_setup_before_parallel: bool
    _done_load_problem_from_fem: bool
    _worker_index: int | str | None
    _worker_name: str | None

    def __init__(self):

        # optimization
        self.seed = None
        self.n_trials = None
        self.timeout = None
        self.include_queued_in_n_trials = False

        # Problem
        self.variable_manager = VariableManager()
        self.objectives = Objectives()
        self.constraints = Constraints()
        self.other_outputs = Functions()

        # multi-fidelity
        self.fidelity = None
        self.sub_fidelity_name = MAIN_FIDELITY_NAME
        self.sub_fidelity_models = SubFidelityModels()

        # System
        self.fem: AbstractFEMInterface | None = None
        self.history: History = History()
        self.solve_condition: Callable[[History], bool] = lambda _: True
        self.entire_status: WorkerStatus = WorkerStatus(ENTIRE_PROCESS_STATUS_KEY)
        self.worker_status: WorkerStatus = WorkerStatus('worker-status')
        self.worker_status_list: list[WorkerStatus] = [self.worker_status]
        self.trial_queue: TrialQueue = TrialQueue()
        self._done_setup_before_parallel = False
        self._done_load_problem_from_fem = False
        self._worker_index: int | str | None = None
        self._worker_name: str | None = None

    # ===== public =====

    def add_constant_value(
            self,
            name: str,
            value: SupportedVariableTypes,
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
    ):
        var: Variable
        # noinspection PyUnreachableCode
        if isinstance(value, Real):
            var = NumericVariable()
        elif isinstance(value, str):
            var = CategoricalVariable()
        else:
            raise ValueError(_(
                en_message='Supported variable types are Real or str, got {type}',
                jp_message='サポートされている変数の型は Real か str ですが、{type} が指定されました。',
                type=type(value),
            ))
        var.name = name
        var.value = value
        var.pass_to_fem = pass_to_fem
        var.properties = properties if properties is not None else {}
        _duplicated_name_check(name, self.variable_manager.variables.keys())
        self.variable_manager.set_variable(var)

    def add_parameter(
            self,
            name: str,
            initial_value: float | None = None,
            lower_bound: float | None = None,
            upper_bound: float | None = None,
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
        _duplicated_name_check(name, self.variable_manager.variables.keys())
        self.variable_manager.set_variable(prm)

    def add_expression_string(
            self,
            name: str,
            expression_string: str,
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
            _disable_matmul_operator: bool = True
    ) -> None:
        var = ExpressionFromString()
        var.name = name
        var._expr = ExpressionFromString.InternalClass(
            expression_string=expression_string,
            _disable_matmul_operator=_disable_matmul_operator,
        )
        var.properties = properties or dict()
        var.pass_to_fem = pass_to_fem
        _duplicated_name_check(name, self.variable_manager.variables.keys())
        self.variable_manager.set_variable(var)

    def add_expression_sympy(
            self,
            name: str,
            sympy_expr: sympy.Expr,
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
    ) -> None:
        var = ExpressionFromString()
        var.name = name
        var._expr = ExpressionFromString.InternalClass(sympy_expr=sympy_expr)
        var.properties = properties or dict()
        var.pass_to_fem = pass_to_fem
        _duplicated_name_check(name, self.variable_manager.variables.keys())
        self.variable_manager.set_variable(var)

    def add_expression(
            self,
            name: str,
            fun: Callable[..., float],
            properties: dict[str, ...] | None = None,
            args: tuple | None = None,
            kwargs: dict | None = None,
            *,
            pass_to_fem: bool = True,
    ) -> None:
        var = ExpressionFromFunction()
        var.name = name
        var.fun = fun
        var.args = args or tuple()
        var.kwargs = kwargs or dict()
        var.properties = properties or dict()
        var.pass_to_fem = pass_to_fem
        _duplicated_name_check(name, self.variable_manager.variables.keys())
        self.variable_manager.set_variable(var)

    def add_categorical_parameter(
            self,
            name: str,
            initial_value: SupportedVariableTypes | None = None,
            choices: list[SupportedVariableTypes] | None = None,
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
            fix: bool = False,
    ) -> None:
        properties = properties if properties is not None else {}

        if fix:
            properties.update({'fix': True})

        prm = CategoricalParameter()
        prm.name = name
        prm.value = initial_value
        prm.choices = choices
        prm.properties = properties
        prm.pass_to_fem = pass_to_fem
        _duplicated_name_check(name, self.variable_manager.variables.keys())
        self.variable_manager.set_variable(prm)

    def add_objective(
            self,
            name: str,
            fun: Callable[..., float],
            direction: DIRECTION = 'minimize',
            args: tuple | None = None,
            kwargs: dict | None = None,
    ) -> None:
        obj = Objective()
        obj.fun = fun
        obj.args = args or ()
        obj.kwargs = kwargs or {}
        obj.direction = direction
        _duplicated_name_check(name, self.objectives.keys())
        self.objectives.update({name: obj})

    def add_objectives(
            self,
            names: str | list[str],
            fun: Callable[..., Sequence[float]],
            n_return: int,
            directions: DIRECTION | Sequence[DIRECTION | None] | None = None,
            args: tuple | None = None,
            kwargs: dict | None = None,
    ):
        # argument processing
        # noinspection PyUnreachableCode
        if isinstance(names, str):
            names = [f'{names}_{i}' for i in range(n_return)]
        elif isinstance(names, Sequence):
            # names = names
            pass
        else:
            raise ValueError(
                _(en_message='`names` must be a string or an array of strings.',
                  jp_message='`names` は文字列か文字列の配列でなければなりません。',)
            )

        if directions is None:
            directions = ['minimize' for __ in range(n_return)]
        else:
            if isinstance(directions, str) or isinstance(directions, Real):
                directions = [directions for __ in range(n_return)]
            else:
                # directions = directions
                pass

        assert len(names) == len(directions) == n_return

        function_factory = ObjectivesFunc(fun, n_return)
        for i, (name, direction) in enumerate(zip(names, directions)):
            fun_i = function_factory.get_fun_that_returns_ith_value(i)
            self.add_objective(
                fun=fun_i,
                name=name,
                direction=direction,
                args=args,
                kwargs=kwargs,
            )

    def add_constraint(
            self,
            name: str,
            fun: Callable[..., float],
            lower_bound: float | None = None,
            upper_bound: float | None = None,
            args: tuple | None = None,
            kwargs: dict | None = None,
            strict: bool = True,
            using_fem: bool | None = None,
    ):

        if lower_bound is None and upper_bound is None:
            raise ValueError(_(
                en_message='One of `lower_bound` and `upper_bound` '
                           'should be set.',
                jp_message='`lower_bound` and `upper_bound` のうち'
                           '少なくとも一つは指定されなければなりません。'
            ))

        cns = Constraint()
        cns.fun = fun
        cns.args = args or ()
        cns.kwargs = kwargs or {}
        cns.lower_bound = lower_bound
        cns.upper_bound = upper_bound
        cns.hard = strict
        cns._opt = self
        cns.using_fem = using_fem
        _duplicated_name_check(name, self.constraints.keys())
        self.constraints.update({name: cns})

    def add_other_output(
            self,
            name: str,
            fun: Callable[..., float],
            args: tuple | None = None,
            kwargs: dict | None = None,
    ):

        other_func = Function()
        other_func.fun = fun
        other_func.args = args or ()
        other_func.kwargs = kwargs or {}
        _duplicated_name_check(name, self.other_outputs.keys())
        self.other_outputs.update({name: other_func})

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
        _duplicated_name_check(name, self.sub_fidelity_models.keys())
        self.sub_fidelity_models._update(name, sub_fidelity_model, fidelity)

    def add_trial(
            self,
            parameters: dict[str, SupportedVariableTypes],
    ):
        self.trial_queue.enqueue(parameters)

    def get_variables(self, format: Literal['dict', 'values', 'raw'] = 'dict'):
        return self.variable_manager.get_variables(
            format=format,
        )

    def get_parameter(self, format: Literal['dict', 'values', 'raw'] = 'dict'):
        return self.variable_manager.get_variables(
            format=format, filter='parameter'
        )

    def set_solve_condition(self, fun: Callable[[History], bool]):
        self.solve_condition = fun

    # ===== private =====

    def _setup_enqueued_trials(self):
        # Insert initial trial
        params: dict = self.variable_manager.get_variables(format='dict', filter='parameter')
        self.trial_queue.enqueue_first(params, flags={_IS_INITIAL_TRIAL_FLAG_KEY: True})

        # Remove trials included in history
        tried: list[dict] = get_tried_list_from_history(
            self.history,
            equality_filters=MAIN_FILTER,
        )
        self.trial_queue.remove_tried(tried)

        # Remove duplicated
        self.trial_queue.remove_duplicated()

    def _should_solve(self, history):
        return self.solve_condition(history)

    def _y(self) -> TrialOutput:
        out = TrialOutput()
        for name, obj in self.objectives.items():
            obj_result = ObjectiveResult(obj, self.fem)
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
                cns_result = ConstraintResult(cns, self.fem)
                out.update({name: cns_result})
        return out

    def _soft_c(self, out: TrialConstraintOutput) -> TrialConstraintOutput:
        for name, cns in self.constraints.items():
            if not cns.hard:
                cns_result = ConstraintResult(cns, self.fem)
                out.update({name: cns_result})
        return out

    def _other_outputs(self, out: TrialFunctionOutput) -> TrialFunctionOutput:
        for name, other_func in self.other_outputs.items():
            other_func_result = FunctionResult(other_func, self.fem)
            out.update({name: other_func_result})
        return out

    def _get_hard_constraint_violation_names(self, hard_c: TrialConstraintOutput) -> list[str]:
        violation_names = []
        for name, result in hard_c.items():
            cns = self.constraints[name]
            l_or_u = result.check_violation()
            if l_or_u is not None:
                logger.warning(_('----- Hard constraint violation! -----'))
                logger.warning(_('constraint: {name}', name=name))
                logger.warning(_('evaluated value: {value}', value=result.value))
                if l_or_u == 'lower_bound':
                    logger.info(_('lower bound: {lb}', lb=cns.lower_bound))
                    violation_names.append(name + '_' + l_or_u)
                elif l_or_u == 'upper_bound':
                    logger.info(_('upper bound: {ub}', ub=cns.upper_bound))
                    violation_names.append(name + '_' + l_or_u)
                else:
                    raise NotImplementedError
        return violation_names

    def _check_and_raise_interruption(self):
        # raise Interrupt
        interrupted = self.entire_status.value >= WorkerStatus.interrupting
        if interrupted:
            self.worker_status.value = WorkerStatus.interrupting
            raise InterruptOptimization

    # ===== solve =====

    class _SolveSet:

        opt: AbstractOptimizer
        opt_: AbstractOptimizer

        def __init__(self, opt: AbstractOptimizer):
            self.opt: AbstractOptimizer = opt
            self.subsampling_idx: SubSampling | None = None

        def _preprocess(self):
            pass

        def _hard_constraint_handling(self, e: HardConstraintViolation):
            pass

        def _hidden_constraint_handling(self, e: _HiddenConstraintViolation):
            pass

        def _skip_handling(self, e: SkipSolve):
            pass

        def _if_succeeded(self, f_return: _FReturnValue):
            pass

        def _postprocess(self):
            pass

        def _solve_or_raise(
                self,
                opt_: AbstractOptimizer,
                parameters: TrialInput,
                history: History = None,
                datetime_start=None,
                trial_id=None,
        ) -> _FReturnValue:

            # create context
            if history is not None:
                record_to_history = history.recording(opt_.fem)
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
                # input
                record.x = parameters
                # output (the value is nan, required for direction recording to graph)
                record.y = TrialOutput(
                    {obj_name: ObjectiveResult(obj, opt_.fem, float('nan'))
                     for obj_name, obj in opt_.objectives.items()}
                )
                record.sub_sampling = self.subsampling_idx
                record.trial_id = trial_id
                record.sub_fidelity_name = opt_.sub_fidelity_name
                record.fidelity = opt_.fidelity
                record.datetime_start = datetime_start
                if self.opt._worker_name is not None:
                    record.messages.append(self.opt._worker_name)

                # check skip
                if not opt_._should_solve(history):
                    record.state = TrialState.skipped
                    raise SkipSolve

                # start solve
                if opt_.sub_fidelity_name != MAIN_FIDELITY_NAME:
                    logger.info('----------')
                    logger.info(_('fidelity: ({name})', name=opt_.sub_fidelity_name))
                logger.info(_('input variables:'))
                logger.info(parameters)

                # ===== update FEM parameter =====
                pass_to_fem = self.opt.variable_manager.get_variables(
                    filter='pass_to_fem',
                    format='raw',
                )
                logger.info(_('updating variables...'))
                opt_.fem.update_parameter(pass_to_fem)
                opt_._check_and_raise_interruption()

                # ===== evaluate hard constraint =====
                logger.info(_('evaluating constraint functions...'))

                hard_c = TrialConstraintOutput()
                try:
                    opt_._hard_c(hard_c)

                except _HiddenConstraintViolation as e:
                    _log_hidden_constraint(e)
                    record.c = hard_c
                    record.state = TrialState.get_corresponding_state_from_exception(e)
                    record.messages.append(
                            _('Hidden constraint violation '
                              'during hard constraint function '
                              'evaluation: ')
                            + create_err_msg_from_exception(e)
                    )

                    raise e

                # check hard constraint violation
                violation_names = opt_._get_hard_constraint_violation_names(hard_c)
                if len(violation_names) > 0:
                    record.c = hard_c
                    record.state = TrialState.hard_constraint_violation
                    record.messages.append(
                            _('Hard constraint violation: ')
                            + ', '.join(violation_names))

                    raise HardConstraintViolation

                # ===== update FEM =====
                logger.info(_('Solving FEM...'))

                try:
                    opt_.fem.update()
                    opt_._check_and_raise_interruption()

                # if hidden constraint violation
                except _HiddenConstraintViolation as e:

                    opt_._check_and_raise_interruption()

                    _log_hidden_constraint(e)
                    record.c = hard_c
                    record.state = TrialState.get_corresponding_state_from_exception(e)
                    record.messages.append(
                            _('Hidden constraint violation in FEM update: ')
                            + create_err_msg_from_exception(e))

                    raise e

                # ===== evaluate y =====
                logger.info(_('evaluating objective functions...'))

                try:
                    y: TrialOutput = opt_._y()
                    record.y = y
                    opt_._check_and_raise_interruption()

                # if intentional error (by user)
                except _HiddenConstraintViolation as e:
                    opt_._check_and_raise_interruption()

                    _log_hidden_constraint(e)
                    record.c = hard_c
                    record.state = TrialState.get_corresponding_state_from_exception(e)
                    record.messages.append(
                            _('Hidden constraint violation during '
                              'objective function evaluation: ')
                            + create_err_msg_from_exception(e))

                    raise e

                # ===== evaluate soft constraint =====
                logger.info(_('evaluating remaining constraints...'))

                soft_c = TrialConstraintOutput()
                try:
                    opt_._soft_c(soft_c)

                # if intentional error (by user)
                except _HiddenConstraintViolation as e:
                    _log_hidden_constraint(e)

                    _c = {}
                    _c.update(soft_c)
                    _c.update(hard_c)

                    record.c = _c
                    record.state = TrialState.get_corresponding_state_from_exception(e)
                    record.messages.append(
                            _('Hidden constraint violation during '
                              'soft constraint function evaluation: ')
                            + create_err_msg_from_exception(e))

                    raise e

                # ===== merge and sort constraints =====
                c = TrialConstraintOutput()
                c.update(soft_c)
                c.update(hard_c)

                # ===== evaluate other functions =====
                logger.info(_('evaluating other functions...'))

                other_outputs = TrialFunctionOutput()
                try:
                    opt_._other_outputs(other_outputs)
                    record.other_outputs = other_outputs

                # if intentional error (by user)
                except _HiddenConstraintViolation as e:
                    _log_hidden_constraint(e)

                    record.other_outputs = other_outputs
                    record.state = TrialState.get_corresponding_state_from_exception(e)
                    record.messages.append(
                            _('Hidden constraint violation during '
                              'another output function evaluation: ')
                            + create_err_msg_from_exception(e))

                    raise e

                # get values as minimize
                y_internal: dict = opt_._convert_y(y)

                logger.info(_('output:'))
                logger.info(y)
                record.c = c
                record.state = TrialState.succeeded

                opt_._check_and_raise_interruption()

            return y, y_internal, c, record

        def solve(
                self,
                x: TrialInput,
                opt_: AbstractOptimizer | None = None,
                trial_id: str = None,
        ) -> _FReturnValue | None:
            """Nothing will be raised even if infeasible."""

            vm = self.opt.variable_manager

            # opt_ はメインの opt 又は sub_fidelity
            opt_ = opt_ or self.opt
            self.opt_ = opt_

            # check interruption
            self.opt._check_and_raise_interruption()

            # if opt_ is not self, update variable manager
            opt_.variable_manager = vm

            # noinspection PyMethodParameters
            class Process:
                def __enter__(self_):
                    # preprocess
                    self._preprocess()

                def __exit__(self_, exc_type, exc_val, exc_tb):
                    # postprocess
                    self._postprocess()

            with Process():

                # declare output
                f_return = None

                # start solve
                datetime_start = datetime.datetime.now()
                try:
                    f_return = self._solve_or_raise(
                        opt_, x, self.opt.history,
                        datetime_start, trial_id
                    )

                except HardConstraintViolation as e:
                    self._hard_constraint_handling(e)

                except _HiddenConstraintViolation as e:
                    self._hidden_constraint_handling(e)

                except SkipSolve as e:
                    self._skip_handling(e)

                else:
                    self._if_succeeded(f_return)

            # check interruption
            self.opt._check_and_raise_interruption()

            return f_return

    def _get_solve_set(self):
        return self._SolveSet(self)

    # ===== run and setup =====

    def _setting_status(self):

        # noinspection PyMethodParameters
        class _SettingStatus:

            def __enter__(self_):
                pass

            def __exit__(self_, exc_type, exc_val, exc_tb):
                if exc_type is not None:
                    if self.worker_status.value < WorkerStatus.crashed:
                        self.worker_status.value = WorkerStatus.crashed
                else:
                    if self.worker_status.value < WorkerStatus.finishing:
                        self.worker_status.value = WorkerStatus.finishing

        return _SettingStatus()

    def run(self) -> None:
        raise NotImplementedError

    def _run(
            self,
            worker_idx: int | str,
            worker_name: str,
            history: History,
            entire_status: WorkerStatus,
            worker_status: WorkerStatus,
            worker_status_list: list[WorkerStatus],
            wait_other_process_setup: bool,
    ) -> None:

        self.history = history
        self.entire_status = entire_status
        self.worker_status = worker_status
        self._worker_index = worker_idx
        self._worker_name = worker_name
        self.worker_status_list = worker_status_list

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
                    print(
                        _(
                            '===== Exception raised in worker {worker_idx} =====',
                            worker_idx=worker_idx
                        ),
                        file=sys.stderr
                    )
                    print_tb(exc_tb)
                    print(
                        _(
                            '{name}: {exc_val}',
                            name=exc_type.__name__,
                            exc_val=exc_val
                        ),
                        file=sys.stderr
                    )

                else:
                    self.worker_status.value = WorkerStatus.finished

        logger.info(_(
            en_message='worker `{worker}` started.',
            worker=worker_name,
        ))

        with SettingStatusFinished():

            self.worker_status.value = WorkerStatus.initializing

            self.worker_status.value = WorkerStatus.launching_fem
            self.fem._setup_after_parallel(self)

            if wait_other_process_setup:
                self.worker_status.value = WorkerStatus.waiting
                while True:
                    self._check_and_raise_interruption()

                    # 他のすべての worker_status が wait 以上になったら break
                    logger.debug([ws.value for ws in worker_status_list])
                    if all([ws.value >= WorkerStatus.waiting
                            for ws in worker_status_list]):

                        # リソースの競合等を避けるため
                        # break する前に index 秒待つ
                        if isinstance(worker_idx, str):
                            wait_second = 0.
                        else:
                            wait_second = int(worker_idx + 1)
                        sleep(wait_second)
                        break

                    sleep(1)

            self.worker_status.value = WorkerStatus.running

            if os.environ.get('DEBUG_FEMOPT_PARALLEL'):
                if isinstance(worker_idx, int):
                    sleep(worker_idx)

            self.run()

        logger.info(_(
            en_message='worker `{worker}` successfully finished!',
            worker=worker_name,
        ))

    def _logging(self):

        # noinspection PyMethodParameters
        class LoggingOutput:
            def __enter__(self_):
                df = self.history.get_df(
                    equality_filters=MAIN_FILTER
                )
                self_.count = len(df) + 1

                succeeded_count = len(df[df['state'] == TrialState.succeeded])
                succeeded_text = _(
                    en_message='{succeeded_count} succeeded trials',
                    jp_message='成功した試行数: {succeeded_count}',
                    succeeded_count=succeeded_count,
                )

                logger.info(f'▼▼▼▼▼ solve {self_.count} ({succeeded_text}) start ▼▼▼▼▼')

            def __exit__(self_, exc_type, exc_val, exc_tb):
                logger.info(f'▲▲▲▲▲ solve {self_.count} end ▲▲▲▲▲\n')

        return LoggingOutput()

    def _load_problem_from_fem(self):
        if self.fem._load_problem_from_fem and not self._done_load_problem_from_fem:
            self.fem.load_variables(self)
            self.fem.load_objectives(self)
            self.fem.load_constraints(self)
        self._done_load_problem_from_fem = True

    # noinspection PyMethodMayBeStatic
    def _get_additional_data(self) -> dict:
        return dict()

    def _collect_additional_data(self) -> dict:
        additional_data = {}
        additional_data.update(self._get_additional_data())
        additional_data.update(self.fem._get_additional_data())
        return additional_data

    def _finalize_history(self):
        if not self.history._finalized:
            parameters = self.variable_manager.get_variables(
                filter='parameter', format='raw'
            )
            self.history.finalize(
                parameters=parameters,
                obj_names=list(self.objectives.keys()),
                cns_names=list(self.constraints.keys()),
                other_output_names=list(self.other_outputs.keys()),
                sub_fidelity_names=[self.sub_fidelity_name] + list(self.sub_fidelity_models.keys()),
                additional_data=self._collect_additional_data()
            )

    def _setup_before_parallel(self):

        if not self._done_setup_before_parallel:

            # check compatibility with fem if needed
            variables = self.variable_manager.get_variables()
            for var_name, variable in variables.items():
                if variable.pass_to_fem:
                    self.fem._check_param_and_raise(var_name)

            # resolve evaluation order
            self.variable_manager.resolve()

            # check the enqueued trials is
            # compatible with current optimization
            # problem setup
            history_set = set(self.history.prm_names)
            for t in self.trial_queue.queue:
                params: dict = t.d
                enqueued_set: set = set(params.keys())

                # Warning if over
                if len(enqueued_set - history_set) > 0:
                    logger.warning(
                        _(
                            en_message='Enqueued parameter set contains '
                                       'more parameters than the optimization '
                                       'problem setup. The extra parameters '
                                       'will be ignored.\n'
                                       'Enqueued set: {enqueued_set}\n'
                                       'Setup set: {history_set}\n'
                                       'Parameters ignored: {over_set}',
                            jp_message='予約された入力変数セットは'
                                       '最適化のセットアップで指定されたよりも'
                                       '多くの変数を含んでいます。'
                                       'そのような変数は無視されます。\n'
                                       '予約された入力変数: {enqueued_set}\n'
                                       '最適化する変数: {history_set}\n'
                                       '無視される変数: {over_set}',
                            enqueued_set=enqueued_set,
                            history_set=history_set,
                            over_set=enqueued_set - history_set,
                        )
                    )

                # Error if not enough
                if len(history_set - enqueued_set) > 0:
                    raise ValueError(
                        _(
                            en_message='The enqueued parameter set lacks '
                                       'some parameters to be optimized.\n'
                                       'Enqueued set: {enqueued_set}\n'
                                       'Parameters to optimize: {history_set}\n'
                                       'Lacked set: {lacked_set}',
                            jp_message='予約された入力変数セットに'
                                       '変数が不足しています。\n'
                                       '予約された変数: {enqueued_set}\n'
                                       '最適化する変数: {history_set}\n'
                                       '足りない変数: {lacked_set}',
                            enqueued_set=enqueued_set,
                            history_set=history_set,
                            lacked_set=history_set - enqueued_set,
                        )
                    )

            # remove duplicated enqueued trials
            self._setup_enqueued_trials()

            self._done_setup_before_parallel = True

    def _setup_after_parallel(self):
        pass

    def _finalize(self):

        # check sub fidelity models
        if self.sub_fidelity_models is None:
            self.sub_fidelity_models = SubFidelityModels()
        for sub_fidelity_model in self.sub_fidelity_models.values():
            assert sub_fidelity_model.objectives.keys() == self.objectives.keys()
            assert sub_fidelity_model.constraints.keys() == self.constraints.keys()

        # finalize
        self._load_problem_from_fem()
        self._finalize_history()

        # setup if needed
        self._setup_before_parallel()
        self._setup_after_parallel()


class SubFidelityModel(AbstractOptimizer):
    pass


class SubFidelityModels(dict[str, SubFidelityModel]):

    def _update(self, name, model: SubFidelityModel, fidelity: Fidelity):
        model.sub_fidelity_name = name
        model.fidelity = fidelity
        self.update({name: model})
