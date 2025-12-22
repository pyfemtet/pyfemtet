from __future__ import annotations

import gc
import os
import inspect
import tempfile
import warnings
from time import sleep
from contextlib import suppress, nullcontext

import numpy as np

import optuna
from optuna.samplers import TPESampler
from optuna.study import Study, MaxTrialsCallback
from optuna.trial import FrozenTrial, TrialState as OptunaTrialState
from optuna_integration.dask import DaskStorage

from pyfemtet._i18n import _
from pyfemtet.opt.history import *
from pyfemtet.opt.interface import *
from pyfemtet.opt.exceptions import *
from pyfemtet.opt.problem.variable_manager import *
from pyfemtet._util.dask_util import *
from pyfemtet._util.closing import closing
from pyfemtet.logger import get_optuna_logger, remove_all_output, get_module_logger

from pyfemtet.opt.optimizer._base_optimizer import *
from pyfemtet.opt.optimizer.optuna_optimizer._optuna_attribute import OptunaAttribute
from pyfemtet.opt.optimizer.optuna_optimizer._pof_botorch.pof_botorch_sampler import PoFBoTorchSampler
from pyfemtet.opt.worker_status import WorkerStatus
from pyfemtet.opt.optimizer._trial_queue import _IS_INITIAL_TRIAL_FLAG_KEY

logger = get_module_logger('opt.optimizer', True)

remove_all_output(get_optuna_logger())

warnings.filterwarnings('ignore', 'set_metric_names', optuna.exceptions.ExperimentalWarning)
warnings.filterwarnings('ignore', 'Argument ``constraints_func`` is an experimental feature.',
                        optuna.exceptions.ExperimentalWarning)


_MESSAGE_ENQUEUED = 'Enqueued trial.'


def check_float_and_raise(value, check_target):
    if isinstance(value, int | float):
        if np.isnan(value):
            raise ValueError(_(
                en_message=f'{check_target} is NaN.',
                jp_message=f'{check_target} は NaN です。',
            ))
    else:
        raise ValueError(_(
            en_message=f'{check_target} should be a number, but {value} ({type(value)}) passed.',
            jp_message=f'{check_target} は数値でなくてはなりませんが、{value} ({type(value)}) が与えられました。',
        ))


class MaxTrialsCallbackExcludingEnqueued(MaxTrialsCallback):
    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        """
        queue を考慮して終了条件を決定する

        if n_trials <= 0:
            -q_wait >= n_trials

            ```
            q_comp q_wait
               |    |
               v    v
             f=====....0.............
            ------|----+------------->
                  n_trials=-4 (<=0)
            ```

        else:
            (s_comp := comp - q_comp) >= n_trials

            ```
              q_comp    s_comp
                 |       |
                 v       v
              f========0===..........
            -----------+--|---------->
                          n_trials=4 (>0)
            ```

        """

        # 与えられた state (=COMPLETE) の trials を取得
        trials = study.get_trials(deepcopy=False, states=self._states)

        # n_trials が負か 0 なら残り WAITING 数と比較
        if self._n_trials <= 0:
            waiting_trials = study.get_trials(deepcopy=False, states=(OptunaTrialState.WAITING,))
            enqueued_trials = [trial for trial in waiting_trials if trial.user_attrs.get('enqueued')]

            if -len(enqueued_trials) >= self._n_trials:
                study.stop()

        # 正なら (初回を除く enqueued trial) を除く
        # COMPLETE 数 (=初回 + sampled_complete) と比較
        else:

            n_complete = len(trials)
            n_enqueued_complete = len([
                trial for trial in trials
                if (
                    trial.user_attrs.get('enqueued', False)
                    and not trial.user_attrs.get(_IS_INITIAL_TRIAL_FLAG_KEY, False)
                )
            ])
            n_sampled_complete = n_complete - n_enqueued_complete

            if n_sampled_complete >= self._n_trials:
                study.stop()


class OptunaOptimizer(AbstractOptimizer):
    """
    An optimizer class utilizing Optuna for hyperparameter optimization.

    This class provides an interface to conduct optimization studies using Optuna.
    It manages the study lifecycle, sampler configuration, and trial execution.

    Attributes:
        study_name (str): Name of the Optuna study.
        storage (str | optuna.storages.BaseStorage): Storage URL or object for the Optuna study.
        storage_path (str): Path to the Optuna study storage.
        current_trial (optuna.trial.Trial | None): The current Optuna trial being evaluated.
        sampler_class (type[optuna.samplers.BaseSampler]): The class of the Optuna sampler to use.
        sampler_kwargs (dict): Keyword arguments to initialize the sampler.
        n_trials (int | None): Number of trials to run in the study.
        timeout (float | None): Maximum time allowed for the optimization.
        callbacks (list): List of callback functions to invoke during optimization.

    Args:
        sampler_class (type[optuna.samplers.BaseSampler], optional): The sampler class for suggesting parameter values. Defaults to TPESampler if None.
        sampler_kwargs (dict[str, ...], optional): Dictionary of keyword arguments for the sampler. Defaults to an empty dictionary.

    Raises:
        None

    Examples:
        >>> optimizer = OptunaOptimizer()
        >>> optimizer.n_trials = 100
        >>> optimizer.timeout = 600
        >>> # Further configuration and usage...
    """

    # system
    study_name = 'pyfemtet-study'
    storage: str | optuna.storages.BaseStorage
    storage_path: str
    current_trial: optuna.trial.Trial | None

    # settings
    # sampler: optuna.samplers.BaseSampler | None = None  # reseed_rng が seed 指定できないため
    sampler_class: type[optuna.samplers.BaseSampler]
    sampler_kwargs: dict
    n_trials: int | None
    timeout: float | None
    callbacks: list

    def __init__(
            self,
            sampler_class: type[optuna.samplers.BaseSampler] = None,
            sampler_kwargs: dict[str, ...] = None,
    ):
        super().__init__()
        self.sampler_kwargs = sampler_kwargs or {}
        self.sampler_class = sampler_class or TPESampler
        self.n_trials: int | None = None
        self.timeout: float | None = None
        self.callbacks = []

    # ===== method checker =====
    # noinspection PyMethodOverriding
    def add_parameter(
            self,
            name: str,
            initial_value: float,
            lower_bound: float | None = None,
            upper_bound: float | None = None,
            step: float | None = None,
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
            fix: bool = False,
    ) -> None:

        if lower_bound is None or upper_bound is None:
            properties = properties or {}
            if properties.get('dynamic_bounds_fun') is None:
                raise ValueError(_(
                    en_message='When using `OptunaOptimizer`, you must either specify `lower_bound` and `upper_bound`, ' \
                               'or include `dynamic_bounds_fun` (Callable[[AbstractOptimizer], float]) in `properties`.',
                    jp_message='OptunaOptimizer では、lower_bound と upper_bound を両方指定するか、' \
                               'または properties に dynamic_bounds_fun (Callable[[AbstractOptimizer], float]) ' \
                               'を含めなければなりません。'
                ))
            else:
                logger.warning(_(
                    en_message='`dynamic_bounds_fun` is under development. The functionally can be changed without any announcement.',
                    jp_message='dynamic_bounds_fun は開発中の機能です。機能は予告なく変更されることがあります。',
                ))

        AbstractOptimizer.add_parameter(self, name, initial_value, lower_bound, upper_bound, step, properties,
                                        pass_to_fem=pass_to_fem, fix=fix)

    # noinspection PyMethodOverriding
    def add_categorical_parameter(
            self,
            name: str,
            initial_value: SupportedVariableTypes,
            choices: list[SupportedVariableTypes],
            properties: dict[str, ...] | None = None,
            *,
            pass_to_fem: bool = True,
            fix: bool = False,
    ) -> None:
        AbstractOptimizer.add_categorical_parameter(self, name, initial_value, choices, properties,
                                                    pass_to_fem=pass_to_fem, fix=fix)

    class _SolveSet(AbstractOptimizer._SolveSet):
        opt: OptunaOptimizer
        optuna_attr: OptunaAttribute

        def _preprocess(self):
            # prepare attribute
            self.optuna_attr = OptunaAttribute(self.opt_)

        def _common(self, e):
            # if (hidden) constraint violation, set trial attribute
            self.optuna_attr.pf_state = TrialState.get_corresponding_state_from_exception(e)
            self.optuna_attr.v_values = self.opt._create_infeasible_constraints(self.opt_)

        def _hard_constraint_handling(self, e: HardConstraintViolation):
            self._common(e)

        def _hidden_constraint_handling(self, e: _HiddenConstraintViolation):
            self._common(e)

        def _skip_handling(self, e: SkipSolve):
            self.optuna_attr.pf_state = TrialState.skipped

        def _if_succeeded(self, f_return: _FReturnValue):

            y, dict_y_internal, c, record = f_return

            # convert constraint to **sorted 1-d array** violation
            assert len(c) == len(self.opt_.constraints)
            v = {}
            for cns_name, cns in self.opt_.constraints.items():
                # This is {lower or upper: violation_value} dict
                violation: dict[str, float] = c[cns_name].calc_violation()
                for l_or_u, violation_value in violation.items():
                    key_ = cns_name + '_' + l_or_u
                    v.update({key_: violation_value})

            # register results
            self.optuna_attr.v_values = tuple(v.values())
            self.optuna_attr.y_values = tuple(dict_y_internal.values())
            self.optuna_attr.pf_state = record.state

        def _postprocess(self):
            # update trial attribute
            self.optuna_attr.set_user_attr_to_trial(self.opt.current_trial)
            super()._postprocess()

    def _create_infeasible_constraints(self, opt_: AbstractOptimizer = None) -> tuple:
        opt_ = opt_ if opt_ is not None else self
        count = 0
        for name, cns in opt_.constraints.items():
            if cns.lower_bound is not None:
                count += 1
            if cns.upper_bound is not None:
                count += 1
        return tuple(1e9 * np.ones(count, dtype=np.float64))

    def _constraint(self, trial: optuna.trial.FrozenTrial):
        main_key = OptunaAttribute(self).key
        user_attribute: OptunaAttribute.AttributeStructure = trial.user_attrs[main_key]
        return user_attribute['violation_values']

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
                    dynamic_bounds_fun = prm.properties.get('dynamic_bounds_fun')
                    if dynamic_bounds_fun:
                        lb, ub = dynamic_bounds_fun(self)
                        check_float_and_raise(lb, _(f'lower_bound of {prm.name}', f'{prm.name} の lower_bound'))
                        check_float_and_raise(ub, _(f'upper_bound of {prm.name}', f'{prm.name} の upper_bound'))
                        prm.lower_bound = lb
                        prm.upper_bound = ub
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
            vm.eval_expressions()

            # check interruption
            self._check_and_raise_interruption()

            # construct TrialInput
            x = vm.get_variables(filter='parameter')

            # prepare solve
            solve_set = self._get_solve_set()

            # # process main fidelity model
            # f_return = solve_set.solve(x)
            # if f_return is None:
            #     y_internal: None = None
            # else:
            #     y_internal: tuple[float] = tuple(f_return[1].values())  # type: ignore
            #
            # # process sub_fidelity_models
            # for sub_fidelity_name, sub_opt in self.sub_fidelity_models.items():
            #     # _SolveSet に特殊な初期化を入れていないので
            #     # sub fidelity でも初期化せず使用可能
            #     solve_set.solve(x, sub_opt)

            # TODO: これがあれば、OptunaOptimizer での
            #    _SolveSet 継承が不要になるかも。
            f_returns = solve_set.solve_all_fidelity(x)
            f_return = f_returns[self.sub_fidelity_name]
            if f_return is None:
                y_internal: None = None
            else:
                y_internal: tuple[float] = tuple(f_return[1].values())  # type: ignore

            # check interruption
            self._check_and_raise_interruption()

            # clear trial
            self.current_trial = None

            # To avoid trial FAILED with hard constraint
            # violation, check pf_state and raise TrialPruned.
            main_key = OptunaAttribute(self).key
            user_attribute: OptunaAttribute.AttributeStructure = trial.user_attrs[main_key]
            state: TrialState = user_attribute['pf_state']
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

    def _get_callback(self, n_trials: int):

        states = (optuna.trial.TrialState.COMPLETE,)

        # restart である場合、追加 N 回と見做す
        if self.history.is_restart:

            study = optuna.load_study(
                study_name=self.study_name,
                storage=self.storage,
            )
            trials = study.get_trials(deepcopy=False, states=states)
            n_complete = len(trials)

            if self.include_queued_in_n_trials:
                # 追加 n_trials 回と見做す
                n_trials = n_trials + n_complete

            else:
                if n_trials > 0:
                    # 追加 n_trials 回と見做すが、
                    # Callback で n_enqueued_complete を足しているので
                    # そのぶんを引かなければならない
                    n_enqueued_complete = len([trial for trial in trials if trial.user_attrs.get('enqueued')])
                    n_trials = n_trials + n_complete - n_enqueued_complete

                else:
                    # queue_wait と比較するので
                    # 何も補正しなくていい
                    pass

        if self.include_queued_in_n_trials:
            Class = MaxTrialsCallback
        else:
            Class = MaxTrialsCallbackExcludingEnqueued
        return Class(n_trials, states=states)

    def _setup_before_parallel(self):

        if self._done_setup_before_parallel:
            return

        AbstractOptimizer._setup_before_parallel(self)  # set flag inside this

        # set default values
        self.sampler_class = self.sampler_class or optuna.samplers.TPESampler
        self.sampler_kwargs = self.sampler_kwargs or {}

        # remove automatically-given arguments
        if 'seed' in self.sampler_kwargs:
            warnings.warn('sampler_kwargs `seed` は'
                          'Optimizer.set_random_seed() で'
                          '与えてください。引数は無視されます。')
            self.sampler_kwargs.pop('seed')
        if 'constraints_func' in self.sampler_kwargs:
            warnings.warn('sampler_kwargs `constraints_func` は'
                          'pyfemtet.opt の内部で自動的に与えられます。'
                          '引数は無視されます。')
            self.sampler_kwargs.pop('constraints_func')

        # create storage path
        self.storage_path = self.history.path.removesuffix('.csv') + '.db'

        # file check
        if self.history.is_restart:
            # check db file existing
            if not os.path.exists(self.storage_path):
                raise FileNotFoundError(self.storage_path)
        else:
            # certify no db file
            if os.path.isfile(self.storage_path):
                os.remove(self.storage_path)

        # if TPESampler and re-starting,
        # create temporary study to avoid error
        # with many pruned trials.
        if (
                issubclass(self.sampler_class, optuna.samplers.TPESampler)
                and self.history.is_restart
        ):

            # If there is a WAITING trial in queue,
            # do nothing (because the callback cannot
            # update the WAITING trial in the existing
            # study.)

            # load existing study
            existing_study = optuna.load_study(
                study_name=self.study_name,
                storage=f'sqlite:///{self.storage_path}',
            )

            if len(existing_study.get_trials(deepcopy=False, states=(OptunaTrialState.WAITING,))) == 0:

                # get unique tmp file
                tmp_storage_path = tempfile.mktemp(suffix='.db')
                self._existing_storage_path = self.storage_path
                self.storage_path = tmp_storage_path

                # create new study
                tmp_study = optuna.create_study(
                    study_name=self.study_name,
                    storage=f'sqlite:///{self.storage_path}',
                    load_if_exists=True,
                    directions=['minimize'] * len(self.objectives),
                )

                # Copy COMPLETE trials to temporary study.
                existing_trials = existing_study.get_trials(
                    states=(
                        optuna.trial.TrialState.COMPLETE,
                    )
                )
                tmp_study.add_trials(existing_trials)

        # setup storage
        client = get_client()
        if client is None:
            self.storage = optuna.storages.get_storage(f'sqlite:///{self.storage_path}')
        else:
            self.storage = DaskStorage(
                f'sqlite:///{self.storage_path}'
            )

        # if new study, create it.
        if not self.history.is_restart:
            # create study
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                load_if_exists=True,
                directions=['minimize'] * len(self.objectives),
            )

            # set objective names
            study.set_metric_names(list(self.objectives.keys()))

            # Add enqueued trials
            for t in self.trial_queue.queue:
                is_initial_step = t.flags.get(_IS_INITIAL_TRIAL_FLAG_KEY, False)
                params: dict[str, SupportedVariableTypes] = t.d
                study.enqueue_trial(
                    params,
                    user_attrs={
                        'message': _MESSAGE_ENQUEUED,
                        'enqueued': True,
                        _IS_INITIAL_TRIAL_FLAG_KEY: is_initial_step,
                    },
                )

    def _setup_after_parallel(self):
        # reseed
        worker = get_worker()
        if worker is not None:
            # self.sampler.reseed_rng()  # サブプロセスのランダム化が固定されない
            idx = self._worker_index
            assert isinstance(idx, int)
            if self.seed is not None:
                self.seed += idx

    def _is_tpe_addressing(self):
        out = False
        if hasattr(self, '_existing_storage_path'):
            if self._existing_storage_path is not None:
                assert os.path.isfile(self._existing_storage_path)
                out = True
        return out

    def _removing_tmp_db_if_needed(self):

        if not self._is_tpe_addressing():
            return nullcontext()

        # noinspection PyMethodParameters
        class RemovingTempDB:
            def __enter__(self_):
                pass

            def __exit__(self_, exc_type, exc_val, exc_tb):

                # clean up temporary file
                if isinstance(self.storage, optuna.storages._CachedStorage):
                    rdb_storage = self.storage._backend
                elif isinstance(self.storage, optuna.storages.RDBStorage):
                    rdb_storage = self.storage
                elif isinstance(self.storage, DaskStorage):
                    base_storage = self.storage.get_base_storage()
                    assert isinstance(base_storage, optuna.storages._CachedStorage)
                    rdb_storage = base_storage._backend
                else:
                    raise NotImplementedError(f'{type(self.storage)=}')

                assert isinstance(rdb_storage, optuna.storages.RDBStorage)

                client = get_client()

                # 最後のプロセスにしか消せないので、
                # 各 worker は dispose だけは行い、
                # 削除は失敗しても気にしないことにする

                if client is None:
                    # 通常 dispose
                    rdb_storage.engine.dispose()

                # run_on_scheduler での dispose
                else:

                    # 他の worker を待つ
                    while True:
                        if all([ws.value >= WorkerStatus.finishing for ws in self.worker_status_list]):
                            break
                        sleep(1)

                    # 通常 dispose
                    rdb_storage.engine.dispose()

                    def dispose_(dask_scheduler):
                        assert isinstance(self.storage, DaskStorage)
                        name_ = self.storage.name
                        ext = dask_scheduler.extensions["optuna"]
                        base_storage_ = ext.storages[name_]
                        rdb_storage_ = base_storage_._backend
                        rdb_storage_.engine.dispose()

                    client.run_on_scheduler(dispose_)
                    gc.collect()

                # try remove
                if os.path.exists(self.storage_path):
                    try:
                        os.remove(self.storage_path)
                    except PermissionError:
                        logger.debug(f'パーミッションエラー。{self.storage_path} の削除処理をスキップします。')
                    else:
                        logger.debug(f'{self.storage_path} の削除に成功しました。')

                self.storage_path = self._existing_storage_path

        return RemovingTempDB()

    def run(self):

        # ===== finalize =====
        self._finalize()

        # ===== construct sampler =====

        # automatically-given arguments
        if len(self.constraints) > 0:
            self.sampler_kwargs.update(
                constraints_func=self._constraint
            )
        if self.seed is not None:
            self.sampler_kwargs.update(
                seed=self.seed
            )

        actual_sampler_kwargs = dict()
        arguments = inspect.signature(self.sampler_class.__init__).parameters
        for k, v in self.sampler_kwargs.items():

            # the key is valid, pass to sampler
            if k in arguments.keys():
                actual_sampler_kwargs.update({k: v})

            # if not automatically-given arguments,
            # show warning
            elif k not in ('seed', 'constraints_func'):
                logger.warning(_(
                    en_message='The given argument {key} is not '
                               'included in ones of {sampler_name}. '
                               '{key} is ignored.',
                    jp_message='{key} は {sampler_name} の'
                               '有効な引数ではないので'
                               '無視されます。',
                    key=k,
                    sampler_name=self.sampler_class.__name__,
                ))

            # else, ignore it
            else:
                pass

        # noinspection PyArgumentList
        sampler = self.sampler_class(**actual_sampler_kwargs)

        # if PoFBoTorchSampler, set opt
        if isinstance(sampler, PoFBoTorchSampler):
            sampler.pyfemtet_optimizer = self  # FIXME: multi-fidelity に対応できない?

        # ===== load study and run =====

        # after quit FEM, try to remove tmp db
        with self._removing_tmp_db_if_needed():

            # quit FEM even if abnormal termination
            with closing(self.fem_manager.all_fems_as_a_fem):

                # load study creating in setup_before_parallel()
                # located on dask scheduler
                study = optuna.load_study(
                    study_name=self.study_name,
                    storage=self.storage,
                    sampler=sampler,
                )

                # if tpe_addressing, load main study
                if self._is_tpe_addressing():
                    # load it
                    existing_study = optuna.load_study(
                        study_name=self.study_name,
                        storage=f'sqlite:///{self._existing_storage_path}',
                    )

                    # and add callback to copy-back
                    # from processing study to existing one.
                    def copy_back(_, trial):
                        # ここで existing_study 内の WAITING を
                        # 更新したり削除したりできないので
                        # tpe_addressing は WAITING がない状態でしか使わない
                        existing_study.add_trial(trial)

                    self.callbacks.append(copy_back)

                # callback
                if self.n_trials is not None:
                    self.callbacks.append(self._get_callback(self.n_trials))

                # run
                with self._setting_status(), suppress(InterruptOptimization):
                    study.optimize(
                        self._objective,
                        timeout=self.timeout,
                        callbacks=self.callbacks,
                    )


def debug_1():
    # from pyfemtet.opt.optimizer.optuna_optimizer.pof_botorch.pof_botorch_sampler import
    # sampler = PoFBoTorchSampler(
    #     n_startup_trials=5,
    #     seed=42,
    #     constraints_func=self._constraint,
    #     pof_config=PoFConfig(
    #         # consider_pof=False,
    #         # feasibility_cdf_threshold='mean',
    #     ),
    #     partial_optimize_acqf_kwargs=PartialOptimizeACQFConfig(
    #         # gen_candidates='scipy',
    #         timeout_sec=5.,
    #         # method='SLSQP'  # 'COBYLA, COBYQA, SLSQP or trust-constr
    #         tol=0.1,
    #         # scipy_minimize_kwargs=dict(),
    #     ),
    # )
    # from optuna_integration import BoTorchSampler
    # sampler = BoTorchSampler(n_startup_trials=5)

    os.chdir(os.path.dirname(__file__))

    def _parabola(_fem: AbstractFEMInterface, _opt: AbstractOptimizer) -> float:
        d = _opt.get_variables()
        x1 = d['x1']
        x2 = d['x2']
        # if _cns(_fem, _opt) < 0:
        #     raise PostProcessError
        return x1 ** 2 + x2 ** 2

    def _parabola2(_fem: AbstractFEMInterface, _opt: AbstractOptimizer) -> float:
        x = _opt.get_variables('values')
        return ((x - 0.1) ** 2).sum()

    def _cns(_fem: AbstractFEMInterface, _opt: AbstractOptimizer) -> float:
        x = _opt.get_variables('values')
        return x[0]

    _fem = NoFEM()
    _opt = OptunaOptimizer()
    _opt.fem = _fem

    # _opt.sampler = optuna.samplers.RandomSampler(seed=42)
    _opt.seed = 42
    _opt.sampler_class = optuna.samplers.TPESampler
    # _opt.sampler_class = optuna.samplers.RandomSampler
    _opt.sampler_kwargs = dict(
        n_startup_trials=5,
    )
    _opt.n_trials = 10

    _opt.add_parameter('x1', 1, -1, 1, step=0.1)
    _opt.add_parameter('x2', 1, -1, 1, step=0.1)
    _opt.add_categorical_parameter('x3', 'a', choices=['a', 'b', 'c'])
    # _opt.add_constraint('cns', _cns, lower_bound=-0.9, args=(_opt,))
    _opt.add_objective('obj1', _parabola, args=(_opt,))
    # _opt.add_objective('obj2', _parabola2, args=(_opt,))

    # # ===== sub-fidelity =====
    # __fem = NoFEM()
    # __opt = SubFidelityModel()
    # __opt.fem = __fem
    # __opt.add_objective('obj1', _parabola, args=(__opt,))
    # # __opt.add_objective('obj2', _parabola2, args=(__opt,))
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


def debug_1s():
    # from pyfemtet.opt.optimizer.optuna_optimizer.pof_botorch.pof_botorch_sampler import
    # sampler = PoFBoTorchSampler(
    #     n_startup_trials=5,
    #     seed=42,
    #     constraints_func=self._constraint,
    #     pof_config=PoFConfig(
    #         # consider_pof=False,
    #         # feasibility_cdf_threshold='mean',
    #     ),
    #     partial_optimize_acqf_kwargs=PartialOptimizeACQFConfig(
    #         # gen_candidates='scipy',
    #         timeout_sec=5.,
    #         # method='SLSQP'  # 'COBYLA, COBYQA, SLSQP or trust-constr
    #         tol=0.1,
    #         # scipy_minimize_kwargs=dict(),
    #     ),
    # )
    # from optuna_integration import BoTorchSampler
    # sampler = BoTorchSampler(n_startup_trials=5)

    os.chdir(os.path.dirname(__file__))

    def _parabola(_fem: AbstractFEMInterface, _opt: AbstractOptimizer) -> float:
        d = _opt.get_variables()
        x1 = d['x1']
        x2 = d['x2']
        # if _cns(_fem, _opt) < 0:
        #     raise PostProcessError
        return x1 ** 2 + x2 ** 2

    def _parabola2(_fem: AbstractFEMInterface, _opt: AbstractOptimizer) -> float:
        x = _opt.get_variables('values')
        return ((x - 0.1) ** 2).sum()

    def _cns(_fem: AbstractFEMInterface, _opt: AbstractOptimizer) -> float:
        x = _opt.get_variables('values')
        return x[0]

    _fem = NoFEM()
    _opt = OptunaOptimizer()
    _opt.fem = _fem

    # _opt.sampler = optuna.samplers.RandomSampler(seed=42)
    _opt.seed = 42
    _opt.sampler_class = optuna.samplers.TPESampler
    # _opt.sampler_class = optuna.samplers.RandomSampler
    _opt.sampler_kwargs = dict(
        n_startup_trials=5,
    )
    _opt.n_trials = 10

    _opt.add_parameter('x1', 1, -1, 1, step=0.1)
    _opt.add_parameter('x2', 1, -1, 1, step=0.1)
    _opt.add_categorical_parameter('x3', 'a', choices=['a', 'b', 'c'])
    # _opt.add_constraint('cns', _cns, lower_bound=-0.9, args=(_opt,))
    _opt.add_objective('obj1', _parabola, args=(_opt,))
    # _opt.add_objective('obj2', _parabola2, args=(_opt,))

    # ===== sub-fidelity =====
    __fem = NoFEM()
    __opt = SubFidelityModel()
    __opt.fem = __fem
    __opt.add_objective('obj1', _parabola, args=(__opt,))
    # __opt.add_objective('obj2', _parabola2, args=(__opt,))

    _opt.add_sub_fidelity_model(name='low-fidelity', sub_fidelity_model=__opt, fidelity=0.5)

    def _solve_condition(_history: History):

        sub_fidelity_df = _history.get_df(
            {'sub_fidelity_name': 'low-fidelity'}
        )
        idx = sub_fidelity_df['state'] == TrialState.succeeded
        pdf = sub_fidelity_df[idx]

        return len(pdf) % 5 == 0

    _opt.set_solve_condition(_solve_condition)

    # _opt.history.path = 'restart-test.csv'
    _opt.run()

    # import plotly.express as px
    # _df = _opt.history.get_df()
    # px.scatter_3d(_df, x='x1', y='x2', z='obj', color='fidelity', opacity=0.5).show()

    _opt.history.save()


def substrate_size(Femtet):
    """基板のXY平面上での専有面積を計算します。"""
    substrate_w = Femtet.GetVariableValue('substrate_w')
    substrate_d = Femtet.GetVariableValue('substrate_d')
    return substrate_w * substrate_d  # 単位: mm2


def debug_2():
    from pyfemtet.opt.interface._femtet_interface.femtet_interface import FemtetInterface

    fem = FemtetInterface(
        femprj_path=os.path.join(os.path.dirname(__file__), 'wat_ex14_parametric_jp.femprj'),
    )

    opt = OptunaOptimizer()

    opt.fem = fem

    opt.add_parameter(name="substrate_w", initial_value=40, lower_bound=22, upper_bound=60)
    opt.add_parameter(name="substrate_d", initial_value=60, lower_bound=34, upper_bound=60)
    opt.add_objective(name='基板サイズ(mm2)', fun=substrate_size)

    opt.n_trials = 5
    opt.history.path = os.path.join(os.path.dirname(__file__), 'femtet-test.csv')

    opt.run()


def debug_3():
    from pyfemtet.opt.interface._femtet_interface.femtet_interface import FemtetInterface

    fem = FemtetInterface(
        femprj_path=os.path.join(os.path.dirname(__file__), 'wat_ex14_parametric_jp.femprj'),
    )

    fem.use_parametric_output_as_objective(
        number=1, direction='minimize',
    )

    opt = OptunaOptimizer()

    opt.fem = fem

    opt.add_parameter(name="substrate_w", initial_value=40, lower_bound=22, upper_bound=60)
    opt.add_parameter(name="substrate_d", initial_value=60, lower_bound=34, upper_bound=60)

    opt.n_trials = 5
    opt.history.path = os.path.join(os.path.dirname(__file__), 'femtet-test-2.csv')

    opt.run()


if __name__ == '__main__':
    debug_1()
    debug_1s()
    debug_2()
    debug_3()
