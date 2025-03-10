from __future__ import annotations

from typing import Callable

import datetime
from contextlib import closing

import numpy as np

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
    # -----
    'OptunaAttribute',
    'OptunaOptimizer'
]


logger = get_module_logger('opt.optimizer')


class HardConstraintViolation(Exception): ...
class InterruptOptimization(Exception): ...
class SkipSolve(Exception): ...


class AbstractOptimizer:

    # problem
    variable_manager: VariableManager
    objectives: Objectives
    constraints: Constraints
    fidelity: Fidelity
    sub_fidelity_name: str
    sub_fidelity_models: SubFidelityModels

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
        self.fidelity: Fidelity = None
        self.sub_fidelity_name = MAIN_FIDELITY_NAME
        self.sub_fidelity_models: SubFidelityModels = None

        # System
        self._fem: AbstractFEMInterface = None
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

    def _process_hidden_constraint(self, e: Exception, record: Record):
        err_msg = create_err_msg_from_exception(e)
        logger.warning('----- Hidden constraint violation! -----')
        logger.warning(f'エラー: {err_msg}')


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
            record_to_history = history.record2()
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

                self._process_hidden_constraint(e, record)
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

                self._process_hidden_constraint(e, record)
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

                self._process_hidden_constraint(e, record)
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
                self._process_hidden_constraint(e, record)

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
        self.history.finalize(
            list(self.variable_manager.get_variables(filter='parameter', format='dict').keys()),
            list(self.objectives.keys()),
            list(self.constraints.keys()),
        )


class SubFidelityModel(AbstractOptimizer): ...


class SubFidelityModels(dict[str, SubFidelityModel]):

    def add(self, name, model: SubFidelityModel, fidelity: Fidelity):
        model.sub_fidelity_name = name
        model.fidelity = fidelity
        self.update({name: model})


# ===== optuna module =====
import warnings

import optuna

from v1.logger import get_optuna_logger, remove_all_output

remove_all_output(get_optuna_logger())

warnings.filterwarnings('ignore', 'set_metric_names', optuna.exceptions.ExperimentalWarning)


class OptunaAttribute:
    """Manage optuna user attribute

    user attributes are:
        sub_fidelity_name:
            fidelity: ...
            OBJECTIVE_ATTR_KEY: ...
            PYFEMTET_STATE_ATTR_KEY: ...
            CONSTRAINT_ATTR_KEY: ...

    """

    OBJECTIVE_KEY = 'internal_objective'
    CONSTRAINT_KEY = 'constraint'
    PYFEMTET_TRIAL_STATE_KEY = 'pyfemtet_trial_state'

    sub_fidelity_name: str  # key
    fidelity: Fidelity
    v_values: tuple  # violation
    y_values: tuple  # internal objective
    pf_state: tuple  # PyFemtet state

    def __init__(self, opt: AbstractOptimizer):
        self.sub_fidelity_name = opt.sub_fidelity_name
        self.fidelity = None
        self.v_values = None
        self.y_values = None
        self.pf_state = None

    # noinspection PyPropertyDefinition
    @classmethod
    @property
    def main_fidelity_key(cls):
        return MAIN_FIDELITY_NAME

    @property
    def key(self):
        return self.sub_fidelity_name

    @property
    def value(self):
        d = {
            'fidelity': self.fidelity,
            self.OBJECTIVE_KEY: self.y_values,
            self.CONSTRAINT_KEY: self.v_values,
            self.PYFEMTET_TRIAL_STATE_KEY: self.pf_state,
        }
        return d

    @staticmethod
    def get_fidelity(optuna_attribute: OptunaAttribute):
        return optuna_attribute.value['fidelity']

    @staticmethod
    def get_violation(optuna_attribute: OptunaAttribute):
        return optuna_attribute.value[OptunaAttribute.CONSTRAINT_KEY]

    @staticmethod
    def get_violation_from_trial_attr(trial_attr: dict):  # value is OptunaAttribute.value
        return trial_attr[OptunaAttribute.CONSTRAINT_KEY]

    @staticmethod
    def get_pf_state_from_trial_attr(trial_attr: dict):  # value is OptunaAttribute.value
        return trial_attr[OptunaAttribute.PYFEMTET_TRIAL_STATE_KEY]


class OptunaOptimizer(AbstractOptimizer):

    current_trial: optuna.trial.Trial

    def _create_infeasible_constraints(self, opt_: AbstractOptimizer = None) -> np.ndarray:
        opt_ = opt_ if opt_ is not None else self
        count = 0
        for name, cns in opt_.constraints.items():
            if cns.lower_bound is not None:
                count += 1
            if cns.upper_bound is not None:
                count += 1
        return tuple(np.ones(count, dtype=np.float64))

    def _constraint(self, trial: optuna.trial.FrozenTrial):
        key = OptunaAttribute(self).key
        value = trial.user_attrs[key]
        return OptunaAttribute.get_violation_from_trial_attr(value)

    def _objective(self, trial: optuna.trial.Trial):

        self.current_trial = trial

        with self._logging():

            vm = self.variable_manager

            # check interruption
            self._check_and_raise_interruption()

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
            self._check_and_raise_interruption()

            # construct TrialInput
            x = vm.get_variables(filter='parameter')
            x_pass_to_fem: TrialInput = vm.get_variables(filter='pass_to_fem')


            def solve(
                    opt_: AbstractOptimizer = self
            ) -> tuple[float] | None:

                # check interruption
                self._check_and_raise_interruption()

                # declare output
                y_internal_: tuple[float] | None = None

                # prepare attribute
                optuna_attr = OptunaAttribute(opt_)

                # if opt_ is not self, update variable manager
                opt_.variable_manager = vm

                # start solve
                datetime_start = datetime.datetime.now()
                try:
                    _, dict_or_None_y_internal, c, record = opt_.f(x, x_pass_to_fem, self.history, datetime_start)

                    # convert dict or None to tuple or None
                    y_internal_ = dict_or_None_y_internal if dict_or_None_y_internal is None else tuple(dict_or_None_y_internal.values())

                # if (hidden) constraint violation, set trial attribute
                except (HardConstraintViolation, HiddenConstraintViolation) as e:
                    optuna_attr.pf_state = TrialState.get_corresponding_state_from_exception(e)
                    optuna_attr.v_values = self._create_infeasible_constraints(opt_)

                # if skipped
                except SkipSolve:
                    optuna_attr.pf_state = TrialState.skipped

                # if succeeded
                else:

                    # convert constraint to **sorted** violation
                    assert len(c) == len(opt_.constraints)
                    v = {}
                    for cns_name, cns in opt_.constraints.items():
                        # This is {lower or upper: violation_value} dict
                        violation: dict[str, float] = c[cns_name].calc_violation()
                        for l_or_u, violation_value in violation.items():
                            key_ = cns_name + '_' + l_or_u
                            v.update({key_: violation_value})

                    # register results
                    optuna_attr.v_values = tuple(v.values())
                    optuna_attr.y_values = y_internal_
                    optuna_attr.pf_state = record.state

                # update trial attribute
                trial.set_user_attr(optuna_attr.key, optuna_attr.value)

                # check interruption
                self._check_and_raise_interruption()

                return y_internal_


            # process main fidelity model
            y_internal: tuple[float] | None = solve()

            # process sub_fidelity_models
            for sub_fidelity_name, sub_opt in self.sub_fidelity_models.items():
                solve(sub_opt)

            # check interruption
            self._check_and_raise_interruption()

            # clear trial
            self.current_trial = None

            # To avoid trial FAILED with hard constraint
            # violation, check pf_state and raise TrialPruned.
            key = OptunaAttribute(self).key
            value = trial.user_attrs[key]
            state = OptunaAttribute.get_pf_state_from_trial_attr(value)
            if state in [
                TrialState.hard_constraint_violation,
                TrialState.model_error,
                TrialState.mesh_error,
                TrialState.solve_error,
                TrialState.post_error,
            ]:
                raise optuna.TrialPruned

            # if main solve skipped, y_internal is empty.
            # this should be processed as FAIL.
            elif state == TrialState.skipped:
                return None

            return y_internal

    def run(self):

        # quit FEM even if abnormal termination
        with closing(self.fem):

            # sub fidelity
            if self.sub_fidelity_models is None:
                self.sub_fidelity_models = SubFidelityModels()
                for sub_fidelity_model in self.sub_fidelity_models.values():
                    assert sub_fidelity_model.objectives.keys() == self.objectives.keys()
                    assert sub_fidelity_model.constraints.keys() == self.constraints.keys()

            # finalize
            self._finalize_history()

            # optuna
            from v1.pof_botorch.pof_botorch_sampler import PoFBoTorchSampler
            from v1.pof_botorch.pof_botorch_sampler import PoFConfig, PartialOptimizeACQFConfig
            sampler = PoFBoTorchSampler(
                n_startup_trials=5,
                seed=42,
                constraints_func=self._constraint,
                pof_config=PoFConfig(
                    # consider_pof=False,
                    # feasibility_threshold='mean',
                ),
                partial_optimize_acqf_kwargs=PartialOptimizeACQFConfig(
                    # gen_candidates='scipy',
                    timeout_sec=5.,
                    # method='SLSQP'  # 'COBYLA, COBYQA, SLSQP or trust-constr
                    tol=0.1,
                    # scipy_minimize_kwargs=dict(),
                ),
            )
            # from optuna_integration import BoTorchSampler
            # sampler = BoTorchSampler(n_startup_trials=5)

            if isinstance(sampler, PoFBoTorchSampler):
                sampler.pyfemtet_optimizer = self  # FIXME: multi-fidelity に対応できない?

            study = optuna.create_study(
                directions=['minimize'] * len(self.objectives),
                sampler=sampler,
            )

            study.set_metric_names(list(self.objectives.keys()))

            study.optimize(
                self._objective,
                n_trials=100,
                catch=InterruptOptimization
            )

    def _check_and_raise_interruption(self):
        try:
            AbstractOptimizer._check_and_raise_interruption(self)
        except InterruptOptimization as e:
            if self.current_trial is not None:
                self.current_trial.study.stop()
            raise e


if __name__ == '__main__':

    from v1.exceptions import PostProcessError

    def _parabola(_fem: AbstractFEMInterface, _opt: AbstractOptimizer):
        x = _opt.get_variables('values')
        # if _cns(_fem, _opt) < 0:
        #     raise PostProcessError
        return (x ** 2).sum()

    def _parabola2(_fem: AbstractFEMInterface, _opt: AbstractOptimizer):
        x = _opt.get_variables('values')
        return ((x-0.1) ** 2).sum()

    def _cns(_fem: AbstractFEMInterface, _opt: AbstractOptimizer):
        x = _opt.get_variables('values')
        return x[0]

    _fem = NoFEM()
    _opt = OptunaOptimizer()
    _opt.fem = _fem
    _opt.add_parameter('x1', 1, -1, 1, step=0.1)
    _opt.add_parameter('x2', 1, -1, 1, step=0.1)
    _opt.add_constraint('cns', _cns, lower_bound=0.8, args=(_fem, _opt))
    _opt.add_objective('obj1', _parabola, args=(_fem, _opt))
    # _opt.add_objective('obj2', _parabola2, args=(_fem, _opt))


    # # ===== sub-fidelity =====
    # __fem = NoFEM()
    # __opt = SubFidelityModel()
    # __opt.fem = __fem
    # __opt.add_objective('obj1', _parabola, args=(__fem, __opt))
    # __opt.add_objective('obj2', _parabola2, args=(__fem, __opt))
    #
    # _opt.add_sub_fidelity_model(name='low-fidelity', sub_fidelity_model=__opt, fidelity=0.5)
    #
    # def _solve_condition(_history: History):
    #
    #     sub_fidelity_df = _history.get_df(
    #         {'sub_fidelity_name': 'low-fidelity'}
    #     )
    #     idx = sub_fidelity_df['state'] == TrialState.succeeded
    #     pdf = sub_fidelity_df[idx]
    #
    #     return len(pdf) % 5 == 0
    #
    # _opt.set_solve_condition(_solve_condition)


    # _opt.history.path = 'restart-test.csv'
    _opt.run()

    # import plotly.express as px
    # _df = _opt.history.get_df()
    # px.scatter_3d(_df, x='x1', y='x2', z='obj', color='fidelity', opacity=0.5).show()

    _opt.history.save()
