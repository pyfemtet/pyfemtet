from __future__ import annotations

from typing import Callable, TypeAlias, Sequence, Literal, NamedTuple, final, Any
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
from pyfemtet.opt.exceptions import show_experimental_warning

__all__ = [
    'AbstractOptimizer',
    'SubFidelityModel',
    'SubFidelityModels',
    '_FReturnValue',
    'OptimizationDataPerFEM',
]


DIRECTION: TypeAlias = (
        float
        | Literal[
            'minimize',
            'maximize',
        ]
)


class _UpdateMode(NamedTuple):
    update_function: bool
    update_fem: bool


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


# 最適化問題の情報をストアするクラス。
class OptimizationData:
    def __init__(self):
        self._initialize_problem()

    def _initialize_objectives(self):  # SurrogateModel で使用
        self.objectives = Objectives()

    def _initialize_problem(self):
        self.variable_manager = VariableManager()
        self._initialize_objectives()
        self.constraints = Constraints()
        self.other_outputs = Functions()

    def add_constant_value(
            self,
            name: str,
            value: SupportedVariableTypes,
            properties: dict[str, Any] | None = None,
            *,
            pass_to_fem: bool = True,
            supress_duplicated_name_check: bool = False,
    ) -> Variable:
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
        if not supress_duplicated_name_check:
            _duplicated_name_check(name, self.variable_manager.variables.keys())
        self.variable_manager.set_variable(var)
        return var

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
            supress_duplicated_name_check: bool = False,
    ) -> NumericParameter:
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
        if not supress_duplicated_name_check:
            _duplicated_name_check(name, self.variable_manager.variables.keys())
        self.variable_manager.set_variable(prm)
        return prm

    def add_expression_string(
            self,
            name: str,
            expression_string: str,
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
            supress_duplicated_name_check: bool = False,
    ) -> ExpressionFromString:
        var = ExpressionFromString()
        var.name = name
        var._expr = ExpressionFromString.InternalClass(
            expression_string=expression_string,
        )
        var.properties = properties or dict()
        var.pass_to_fem = pass_to_fem
        if not supress_duplicated_name_check:
            _duplicated_name_check(name, self.variable_manager.variables.keys())
        self.variable_manager.set_variable(var)
        return var

    def add_expression_sympy(
            self,
            name: str,
            sympy_expr: sympy.Expr,
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
            supress_duplicated_name_check: bool = False,
    ) -> ExpressionFromString:
        var = ExpressionFromString()
        var.name = name
        var._expr = ExpressionFromString.InternalClass(sympy_expr=sympy_expr)
        var.properties = properties or dict()
        var.pass_to_fem = pass_to_fem
        if not supress_duplicated_name_check:
            _duplicated_name_check(name, self.variable_manager.variables.keys())
        self.variable_manager.set_variable(var)
        return var

    def add_expression(
            self,
            name: str,
            fun: Callable[..., float],
            properties: dict[str, ...] | None = None,
            args: tuple | None = None,
            kwargs: dict | None = None,
            *,
            pass_to_fem: bool = True,
            supress_duplicated_name_check: bool = False,
    ) -> ExpressionFromFunction:
        var = ExpressionFromFunction()
        var.name = name
        var.fun = fun
        var.args = args or tuple()
        var.kwargs = kwargs or dict()
        var.properties = properties or dict()
        var.pass_to_fem = pass_to_fem
        if not supress_duplicated_name_check:
            _duplicated_name_check(name, self.variable_manager.variables.keys())
        self.variable_manager.set_variable(var)
        return var

    def add_categorical_parameter(
            self,
            name: str,
            initial_value: SupportedVariableTypes | None = None,
            choices: list[SupportedVariableTypes] | None = None,
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
            fix: bool = False,
            supress_duplicated_name_check: bool = False,
    ) -> CategoricalParameter:
        properties = properties if properties is not None else {}

        if fix:
            properties.update({'fix': True})

        prm = CategoricalParameter()
        prm.name = name
        prm.value = initial_value
        prm.choices = choices
        prm.properties = properties
        prm.pass_to_fem = pass_to_fem
        if not supress_duplicated_name_check:
            _duplicated_name_check(name, self.variable_manager.variables.keys())
        self.variable_manager.set_variable(prm)
        return prm

    def add_objective(
            self,
            name: str,
            fun: Callable[..., float],
            direction: DIRECTION = 'minimize',
            args: tuple | None = None,
            kwargs: dict | None = None,
            supress_duplicated_name_check: bool = False,
    ) -> Objective:
        obj = Objective()
        obj.fun = fun
        obj.args = args or ()
        obj.kwargs = kwargs or {}
        obj.direction = direction
        obj.fem_ctx = None
        if not supress_duplicated_name_check:
            _duplicated_name_check(name, self.objectives.keys())
        self.objectives.update({name: obj})
        return obj  # Context で fem_ctx をセットするために返す

    def add_objectives(
            self,
            names: str | list[str],
            fun: Callable[..., Sequence[float]],
            n_return: int,
            directions: DIRECTION | Sequence[DIRECTION | None] | None = None,
            args: tuple | None = None,
            kwargs: dict | None = None,
            supress_duplicated_name_check: bool = False,
    ) -> list[Objective]:
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

        out = []
        function_factory = ObjectivesFunc(fun, n_return)
        for i, (name, direction) in enumerate(zip(names, directions)):
            fun_i = function_factory.get_fun_that_returns_ith_value(i)
            out.append(self.add_objective(
                fun=fun_i,
                name=name,
                direction=direction,
                args=args,
                kwargs=kwargs,
                supress_duplicated_name_check=supress_duplicated_name_check,
            ))

        return out

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
            supress_duplicated_name_check: bool = False,
    ) -> Constraint:
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
        cns.fem_ctx = None
        cns.using_fem = using_fem
        if not supress_duplicated_name_check:
            _duplicated_name_check(name, self.constraints.keys())
        self.constraints.update({name: cns})
        return cns  # Context で fem_ctx をセットするために返す

    def add_other_output(
            self,
            name: str,
            fun: Callable[..., float],
            args: tuple | None = None,
            kwargs: dict | None = None,
            supress_duplicated_name_check: bool = False,
    ):
        other_func = Function()
        other_func.fun = fun
        other_func.args = args or ()
        other_func.kwargs = kwargs or {}
        other_func.fem_ctx = None
        if not supress_duplicated_name_check:
            _duplicated_name_check(name, self.other_outputs.keys())
        self.other_outputs.update({name: other_func})
        return other_func


# 最適化問題の関数を FEM ごとに管理して
# 実行制御とかを行うクラス
class OptimizationDataPerFEM(OptimizationData):

    fem: AbstractFEMInterface

    def __init__(self, fem: AbstractFEMInterface):
        self.fem = fem
        super().__init__()

    def _initialize_problem(self):
        super()._initialize_problem()
        self._done_load_problem_from_fem = False

    # TODO: Constraint が using_fem check を使うためだけにこの実装は過剰。
    @staticmethod
    def _with_add_fem_ctx(f):
        def wrapper(self, *args, **kwargs):
            def add_fem_ctx(something):
                if isinstance(something, Function):
                    something.fem_ctx = self

            out = f(self, *args, **kwargs)
            if isinstance(out, Sequence):
                for item in out:
                    add_fem_ctx(item)
            else:
                add_fem_ctx(out)

            return out
        return wrapper

    @_with_add_fem_ctx
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
        supress_duplicated_name_check: bool = False,
    ):
        return super().add_constraint(
            name=name,
            fun=fun,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            args=args,
            kwargs=kwargs,
            strict=strict,
            using_fem=using_fem,
            supress_duplicated_name_check=supress_duplicated_name_check,
        )

    @_with_add_fem_ctx
    def add_objective(
            self,
            name: str,
            fun: Callable[..., float],
            direction: DIRECTION = 'minimize',
            args: tuple | None = None,
            kwargs: dict | None = None,
            supress_duplicated_name_check: bool = False,
    ) -> Objective:
        return super().add_objective(
            name=name,
            fun=fun,
            direction=direction,
            args=args,
            kwargs=kwargs,
            supress_duplicated_name_check=supress_duplicated_name_check,
        )

    @_with_add_fem_ctx
    def add_objectives(
            self,
            names: str | list[str],
            fun: Callable[..., Sequence[float]],
            n_return: int,
            directions: DIRECTION | Sequence[DIRECTION | None] | None = None,
            args: tuple | None = None,
            kwargs: dict | None = None,
            supress_duplicated_name_check: bool = False,
    ) -> list[Objective]:
        return super().add_objectives(
            names=names,
            fun=fun,
            n_return=n_return,
            directions=directions,
            args=args,
            kwargs=kwargs,
            supress_duplicated_name_check=supress_duplicated_name_check,
        )

    @_with_add_fem_ctx
    def add_other_output(
            self,
            name: str,
            fun: Callable[..., float],
            args: tuple | None = None,
            kwargs: dict | None = None,
            supress_duplicated_name_check: bool = False,
    ):
        return super().add_other_output(
            name=name,
            fun=fun,
            args=args,
            kwargs=kwargs,
            supress_duplicated_name_check=supress_duplicated_name_check,
        )

    def _y_common(self, fem: AbstractFEMInterface) -> TrialOutput:
        out = TrialOutput()
        for name, obj in self.objectives.items():
            obj_result = ObjectiveResult(obj, fem)
            out.update({name: obj_result})
        return out

    def _y(self) -> TrialOutput:
        return self._y_common(self.fem)

    def _convert_y(self, y: TrialOutput) -> dict:
        out = dict()
        for name, result in y.items():
            obj = self.objectives[name]
            value_internal = obj.convert(result.value)
            out.update({name: value_internal})
        return out

    def _c_common(
            self,
            out: TrialConstraintOutput,
            fem: AbstractFEMInterface,
            hard: bool,
    ) -> TrialConstraintOutput:
        for name, cns in self.constraints.items():
            if cns.hard == hard:
                cns_result = ConstraintResult(cns, fem)
                out.update({name: cns_result})
        return out

    def _hard_c(self, out: TrialConstraintOutput) -> TrialConstraintOutput:
        return self._c_common(out, self.fem, hard=True)

    def _soft_c(self, out: TrialConstraintOutput) -> TrialConstraintOutput:
        return self._c_common(out, self.fem, hard=False)

    def _other_outputs_common(self, out: TrialFunctionOutput, fem: AbstractFEMInterface) -> TrialFunctionOutput:
        for name, other_func in self.other_outputs.items():
            other_func_result = FunctionResult(other_func, fem)
            out.update({name: other_func_result})
        return out

    def _other_outputs(self, out: TrialFunctionOutput) -> TrialFunctionOutput:
        return self._other_outputs_common(out, self.fem)

    def _load_problem_from_fem(self):
        self.fem.load_variables(self)
        self.fem.load_objectives(self)
        self.fem.load_constraints(self)


class FEMListForGlobal(FEMListInterface):

    # 単一 FEM しか使わない場合に list を意識させないため
    # 長さ 1 の時は要素を返す処理が方々で必要
    @property
    def object_pass_to_fun(self):
        out = super().object_pass_to_fun
        if len(out) == 1:
            return out[0]
        else:
            return out

    # GlobalOptimizationData が これらを操作すると二重操作になる。
    def update_parameter(self, x: TrialInput) -> None:
        pass

    def update(self):
        pass

    def _check_param_and_raise(self, prm_name) -> None:
        pass

    # ユーザーが操作できないようにする
    def append(self, fem: AbstractFEMInterface):
        raise RuntimeError("Invalid operation. Use `AbstractOptimizer.fem_manager.append` instead.")

    def _append(self, fem: AbstractFEMInterface):
        FEMListInterface.append(self, fem)


# 特定の FEM に紐づかない最適化問題データを管理する
class GlobalOptimizationData(OptimizationDataPerFEM):
    fem: FEMListForGlobal


# 特定の FEM とそれに紐づく最適化問題データを管理する
class FEMAndDataConnectionManager:
    _contexts: list[OptimizationDataPerFEM]
    _fems: list[AbstractFEMInterface]

    def __init__(self):
        self.all_fems_as_a_fem = FEMListForGlobal()
        self.global_data = GlobalOptimizationData(self.all_fems_as_a_fem)
        self._fems = []
        self._contexts = []

    @property
    def fems(self) -> tuple[AbstractFEMInterface, ...]:
        return tuple(self._fems)

    @property
    def contexts(self) -> tuple[OptimizationDataPerFEM, ...]:
        return tuple(self._contexts + [self.global_data])

    def append(self, fem: AbstractFEMInterface) -> OptimizationDataPerFEM:
        self._fems.append(fem)
        self.all_fems_as_a_fem._append(fem)
        ctx = OptimizationDataPerFEM(fem)
        self._contexts.append(ctx)
        return ctx


class AbstractOptimizer(OptimizationData):

    # optimize
    n_trials: int | None
    timeout: float | None
    seed: int | None
    include_queued_in_n_trials: bool

    # problem
    fidelity: Fidelity | None
    sub_fidelity_name: str
    sub_fidelity_models: SubFidelityModels | None

    # system
    history: History
    entire_status: WorkerStatus
    worker_status: WorkerStatus
    worker_status_list: list[WorkerStatus]
    _done_setup_before_parallel: bool
    _worker_index: int | str | None
    _worker_name: str | None

    def __init__(self):
        super().__init__()

        # optimization
        self.seed = None
        self.n_trials = None
        self.timeout = None
        self.include_queued_in_n_trials = False

        # multi-fidelity
        self.fidelity = None
        self.sub_fidelity_name = MAIN_FIDELITY_NAME
        self.sub_fidelity_models = SubFidelityModels()

        # System
        self.fem_manager = FEMAndDataConnectionManager()
        self.history: History = History()
        self.solve_condition: Callable[[History], bool] = lambda _: True
        self.termination_condition: Callable[[History], bool] = lambda _: False
        self.entire_status: WorkerStatus = WorkerStatus(ENTIRE_PROCESS_STATUS_KEY)
        self.worker_status: WorkerStatus = WorkerStatus('worker-status')
        self.worker_status_list: list[WorkerStatus] = [self.worker_status]
        self.trial_queue: TrialQueue = TrialQueue()
        self._done_setup_before_parallel = False
        self._worker_index: int | str | None = None
        self._worker_name: str | None = None

    @property
    def fem(self) -> AbstractFEMInterface | tuple[AbstractFEMInterface, ...]:
        fems: tuple[AbstractFEMInterface, ...] = self.fem_manager.fems
        if len(fems) == 1:
            return fems[0]
        else:
            return fems

    @fem.setter
    def fem(self, value: AbstractFEMInterface):
        # FEMList は初期化しつつ、
        # global_data は初期化しないようにする
        self.fem_manager.all_fems_as_a_fem = FEMListForGlobal()
        self.fem_manager.global_data.fem = self.fem_manager.all_fems_as_a_fem
        self.fem_manager.append(value)

    def add_fem(self, fem: AbstractFEMInterface) -> OptimizationDataPerFEM:
        show_experimental_warning('add_fem', logger)
        return self.fem_manager.append(fem)

    # @property
    # def global_data(self) -> GlobalOptimizationData:
    #     return self.fem_manager.global_data

    # ===== public =====
    @staticmethod
    def _dispatch_global_data_store(f):
        def wrapper(self: AbstractOptimizer, *args, **kwargs):
            instance: GlobalOptimizationData = self.fem_manager.global_data
            method_name = f.__name__
            method = getattr(
                instance,
                method_name,
            )
            out = method(*args, **kwargs)
            self._refresh_problem()
            return out

        return wrapper

    @_dispatch_global_data_store
    def add_constant_value(
            self,
            name: str,
            value: SupportedVariableTypes,
            properties: dict[str, Any] | None = None,
            *,
            pass_to_fem: bool = True,
            supress_duplicated_name_check: bool = False,
    ):
        pass

    @_dispatch_global_data_store
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
            supress_duplicated_name_check: bool = False,
    ) -> None:
        pass

    @_dispatch_global_data_store
    def add_expression_string(
            self,
            name: str,
            expression_string: str,
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
            supress_duplicated_name_check: bool = False,
    ) -> None:
        pass

    @_dispatch_global_data_store
    def add_expression_sympy(
            self,
            name: str,
            sympy_expr: sympy.Expr,
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
            supress_duplicated_name_check: bool = False,
    ) -> None:
        pass

    @_dispatch_global_data_store
    def add_expression(
            self,
            name: str,
            fun: Callable[..., float],
            properties: dict[str, ...] | None = None,
            args: tuple | None = None,
            kwargs: dict | None = None,
            *,
            pass_to_fem: bool = True,
            supress_duplicated_name_check: bool = False,
    ) -> None:
        pass

    @_dispatch_global_data_store
    def add_categorical_parameter(
            self,
            name: str,
            initial_value: SupportedVariableTypes | None = None,
            choices: list[SupportedVariableTypes] | None = None,
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
            fix: bool = False,
            supress_duplicated_name_check: bool = False,
    ) -> None:
        pass

    @_dispatch_global_data_store
    def add_objective(
            self,
            name: str,
            fun: Callable[..., float],
            direction: DIRECTION = 'minimize',
            args: tuple | None = None,
            kwargs: dict | None = None,
            supress_duplicated_name_check: bool = False,
    ) -> None:
        pass

    @_dispatch_global_data_store
    def add_objectives(
            self,
            names: str | list[str],
            fun: Callable[..., Sequence[float]],
            n_return: int,
            directions: DIRECTION | Sequence[DIRECTION | None] | None = None,
            args: tuple | None = None,
            kwargs: dict | None = None,
            supress_duplicated_name_check: bool = False,
    ):
        pass

    @_dispatch_global_data_store
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
            supress_duplicated_name_check: bool = False,
    ):
        pass

    @_dispatch_global_data_store
    def add_other_output(
            self,
            name: str,
            fun: Callable[..., float],
            args: tuple | None = None,
            kwargs: dict | None = None,
            supress_duplicated_name_check: bool = False,
    ):
        pass

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

    def set_termination_condition(self, fun: Callable[[History], bool] | None):
        if fun is None:
            self.termination_condition = lambda _: False
        else:
            self.termination_condition = fun
        show_experimental_warning('set_termination_condition', logger)

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

    @final
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
            if self.opt.termination_condition(self.opt.history):
                self.opt.entire_status.value = WorkerStatus.interrupting

        def _solve_or_raise(
            self,
            opt_: AbstractOptimizer,
            parameters: TrialInput,
            history: History = None,
            trial_id=None,
        ) -> _FReturnValue:
            # create context
            if history is not None:
                record_to_history = history.recording(opt_.fem_manager.fems)
            else:

                class DummyRecordContext:
                    def __enter__(self):
                        return Record()

                    def __exit__(self, exc_type, exc_val, exc_tb):
                        pass

                record_to_history = DummyRecordContext()

            # context manager for preprocessing and postprocessing
            class AllFEMPrePostPerFidelity:

                # noinspection PyMethodParameters
                def __enter__(self_):
                    opt_.fem_manager.all_fems_as_a_fem.trial_postprocess_per_fidelity()
                    return self_

                # noinspection PyMethodParameters
                def __exit__(self_, exc_type, exc_val, exc_tb):
                    opt_.fem_manager.all_fems_as_a_fem.trial_postprocess_per_fidelity()

            # processing with recording
            # postprocess_after_recording より後に trial_postprocess を
            # 実行するために、record_to_history の前に AllFEMPrePostPerFidelity を使う
            with AllFEMPrePostPerFidelity(), record_to_history as record:
                # output
                empty_c: TrialConstraintOutput = TrialConstraintOutput()
                empty_other_outputs: TrialFunctionOutput = TrialFunctionOutput()
                empty_y = TrialOutput()
                # (Required for direction recording to graph even if the value is nan)
                for ctx in opt_.fem_manager.contexts:
                    empty_y.update({
                        obj_name: ObjectiveResult(obj, ctx.fem, float('nan'))
                        for obj_name, obj in ctx.objectives.items()
                    })
                y_internal: dict = {}

                # record common result
                record.x = parameters
                record.y = empty_y
                record.c = empty_c
                record.other_outputs = empty_other_outputs
                record.sub_sampling = self.subsampling_idx
                record.trial_id = trial_id
                record.sub_fidelity_name = opt_.sub_fidelity_name
                record.fidelity = opt_.fidelity
                if self.opt._worker_name is not None:
                    record.messages.append(self.opt._worker_name)

                # check skip
                if not opt_._should_solve(history):
                    record.state = TrialState.skipped
                    raise SkipSolve

                from contextlib import nullcontext

                # start solve per FEM
                if opt_.sub_fidelity_name != MAIN_FIDELITY_NAME:
                    logger.info('----------')
                    logger.info(_('fidelity: ({name})', name=opt_.sub_fidelity_name))
                for ctx in opt_.fem_manager.contexts:
                    logger.info(_('input variables:'))
                    logger.info(parameters)

                    # ===== update FEM parameter =====
                    with nullcontext():
                        # 共通
                        pass_to_fem = self.opt.fem_manager.global_data.variable_manager.get_variables(
                            filter="pass_to_fem",
                            format="raw",
                        )
                        # FEM ごとのものを追加
                        pass_to_fem.update(ctx.variable_manager.get_variables(
                            filter="pass_to_fem",
                            format="raw",
                        ))
                        logger.info(_('updating variables...'))
                        ctx.fem.update_parameter(pass_to_fem)
                        opt_._check_and_raise_interruption()

                    # ===== evaluate hard constraint =====
                    with nullcontext():
                        ctx_hard_c = TrialConstraintOutput()
                        logger.info(_('evaluating constraint functions...'))

                        try:
                            ctx._hard_c(ctx_hard_c)
                            record.c.update(ctx_hard_c)

                        except _HiddenConstraintViolation as e:
                            _log_hidden_constraint(e)
                            record.c.update(ctx_hard_c)
                            record.state = (
                                TrialState.get_corresponding_state_from_exception(e)
                            )
                            record.messages.append(
                                    _('Hidden constraint violation '
                                      'during hard constraint function '
                                      'evaluation: ')
                                    + create_err_msg_from_exception(e)
                            )

                            raise e

                        # check hard constraint violation
                        violation_names = opt_._get_hard_constraint_violation_names(ctx_hard_c)
                        if len(violation_names) > 0:
                            record.state = TrialState.hard_constraint_violation
                            record.messages.append(
                                    _('Hard constraint violation: ')
                                    + ', '.join(violation_names))

                            raise HardConstraintViolation

                    # ===== update FEM =====
                    with nullcontext():
                        logger.info(_('Solving FEM...'))

                        try:
                            ctx.fem.update()
                            opt_._check_and_raise_interruption()

                        # if hidden constraint violation
                        except _HiddenConstraintViolation as e:
                            opt_._check_and_raise_interruption()

                            _log_hidden_constraint(e)
                            record.state = (
                                TrialState.get_corresponding_state_from_exception(e)
                            )
                            record.messages.append(
                                _("Hidden constraint violation in FEM update: ")
                                + create_err_msg_from_exception(e)
                            )

                            raise e

                    # ===== evaluate y =====
                    with nullcontext():
                        logger.info(_('evaluating objective functions...'))

                        try:
                            ctx_y: TrialOutput = ctx._y()
                            record.y.update(ctx_y)
                            y_internal.update(ctx._convert_y(ctx_y))
                            opt_._check_and_raise_interruption()

                        # if intentional error (by user)
                        except _HiddenConstraintViolation as e:
                            record.y.update(ctx_y)
                            opt_._check_and_raise_interruption()

                            _log_hidden_constraint(e)
                            record.state = TrialState.get_corresponding_state_from_exception(e)
                            record.messages.append(
                                    _('Hidden constraint violation during '
                                      'objective function evaluation: ')
                                    + create_err_msg_from_exception(e))

                            raise e

                    # ===== evaluate soft constraint =====
                    with nullcontext():
                        ctx_soft_c = TrialConstraintOutput()
                        logger.info(_('evaluating remaining constraints...'))

                        try:
                            ctx._soft_c(ctx_soft_c)
                            record.c.update(ctx_soft_c)

                        # if intentional error (by user)
                        except _HiddenConstraintViolation as e:
                            _log_hidden_constraint(e)
                            record.c.update(ctx_soft_c)
                            record.state = TrialState.get_corresponding_state_from_exception(e)
                            record.messages.append(
                                    _('Hidden constraint violation during '
                                      'soft constraint function evaluation: ')
                                    + create_err_msg_from_exception(e))

                            raise e

                    # ===== evaluate other functions =====
                    with nullcontext():
                        logger.info(_('evaluating other functions...'))

                        ctx_other_outputs = TrialFunctionOutput()
                        try:
                            ctx._other_outputs(ctx_other_outputs)
                            record.other_outputs.update(ctx_other_outputs)

                        # if intentional error (by user)
                        except _HiddenConstraintViolation as e:
                            _log_hidden_constraint(e)
                            record.other_outputs.update(ctx_other_outputs)
                            record.state = (
                                TrialState.get_corresponding_state_from_exception(e)
                            )
                            record.messages.append(
                                _(
                                    "Hidden constraint violation during "
                                    "another output function evaluation: "
                                )
                                + create_err_msg_from_exception(e)
                            )

                            raise e

                logger.info(_('output:'))
                logger.info(record.y)
                record.state = TrialState.succeeded

                opt_._check_and_raise_interruption()

            return record.y, y_internal, record.c, record

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
                try:
                    f_return = self._solve_or_raise(
                        opt_=opt_,
                        parameters=x,
                        history=self.opt.history,
                        trial_id=trial_id,
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

        def solve_all_fidelity(self, x: TrialInput) -> dict[str, _FReturnValue | None]:
            out: dict[str, _FReturnValue | None] = dict()

            class AllFEMPrePostPerTrial:

                # noinspection PyMethodParameters
                def __enter__(self_):
                    self.opt.fem_manager.all_fems_as_a_fem.trial_preprocess()
                    for sub in self.opt.sub_fidelity_models.values():
                        sub.fem_manager.all_fems_as_a_fem.trial_preprocess()
                    return self_

                # noinspection PyMethodParameters
                def __exit__(self_, exc_type, exc_val, exc_tb):
                    self.opt.fem_manager.all_fems_as_a_fem.trial_postprocess()
                    for sub in self.opt.sub_fidelity_models.values():
                        sub.fem_manager.all_fems_as_a_fem.trial_postprocess()

            with AllFEMPrePostPerTrial():

                main_fidelity = self.opt
                f_return = self.solve(x)
                out[main_fidelity.sub_fidelity_name] = f_return

                for sub_fidelity_name, sub_fidelity in main_fidelity.sub_fidelity_models.items():
                    f_return = self.solve(x, sub_fidelity)
                    out[sub_fidelity_name] = f_return

            return out

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
            self.fem_manager.all_fems_as_a_fem._setup_after_parallel(self)

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
                    sleep((worker_idx + 1) * 2)

            self.run()

        logger.info(
            _(
                en_message="worker `{worker}` successfully finished!",
                worker=worker_name,
            )
        )

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

    def _refresh_problem(self):
        # ctx の内容が減っている場合でも検出できるように初期化
        self._initialize_problem()

        for ctx in self.fem_manager.contexts:
            # femlist が _load すると各 FEMInterface が
            # 想定するのと異なる object_pass_to_fun (=Sequence)が
            # 渡されてしまうので、
            # 各 context のみが _load_... を行う必要がある。
            if not isinstance(ctx, GlobalOptimizationData):
                ctx._load_problem_from_fem()

            # optimizer に反映する
            self.objectives.update(ctx.objectives)
            self.constraints.update(ctx.constraints)
            self.other_outputs.update(ctx.other_outputs)
            self.variable_manager.variables.update(ctx.variable_manager.variables)

        # 特殊処理が必要な場合は最後に fem の責任で行う
        for ctx in self.fem_manager.contexts:
            if not isinstance(ctx, GlobalOptimizationData):
                ctx.fem.contact_to_optimizer(self, self.fem_manager.global_data, ctx)

    # noinspection PyMethodMayBeStatic
    def _get_additional_data(self) -> dict:
        return dict()

    def _collect_additional_data(self) -> dict:
        additional_data = {}

        additional_data.update(self._get_additional_data())
        additional_data.update(self.fem_manager.all_fems_as_a_fem._get_additional_data())
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
            for ctx in self.fem_manager.contexts:
                # 共通
                for variable in self.fem_manager.global_data.variable_manager.variables.values():
                    if variable.pass_to_fem:
                        ctx.fem._check_param_and_raise(variable.name)
                # ctx ごと
                for variable in ctx.variable_manager.variables.values():
                    if variable.pass_to_fem:
                        ctx.fem._check_param_and_raise(variable.name)

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
                            en_message="The enqueued parameter set lacks "
                            "some parameters to be optimized.\n"
                            "Enqueued set: {enqueued_set}\n"
                            "Parameters to optimize: {history_set}\n"
                            "Lacked set: {lacked_set}",
                            jp_message="予約された入力変数セットに"
                            "変数が不足しています。\n"
                            "予約された変数: {enqueued_set}\n"
                            "最適化する変数: {history_set}\n"
                            "足りない変数: {lacked_set}",
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
        self._refresh_problem()
        self.variable_manager.resolve()
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
