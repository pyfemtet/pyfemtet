from __future__ import annotations

import gc
import os
import inspect
import tempfile
import warnings
import datetime
from contextlib import closing, nullcontext

import numpy as np

import optuna
from optuna.samplers import TPESampler
from optuna.study import MaxTrialsCallback
from optuna_integration.dask import DaskStorage

from v1.history.history import *
from v1.problem import *
from v1.interface import *
from v1.exceptions import *
from v1.variable_manager import *
from v1.utils.dask_util import *
from v1.logger import get_optuna_logger, remove_all_output, get_module_logger

from v1.optimizer.optimizer import *
from v1.optimizer.optuna_optimizer.optuna_attribute import OptunaAttribute

logger = get_module_logger('opt.optimizer', False)

remove_all_output(get_optuna_logger())

warnings.filterwarnings('ignore', 'set_metric_names', optuna.exceptions.ExperimentalWarning)


class OptunaOptimizer(AbstractOptimizer):

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
    seed: int | None
    callbacks: list

    def __init__(self):
        super().__init__()
        self.sampler_kwargs = {}
        self.sampler_class = TPESampler
        self.n_trials: int | None = None
        self.timeout: float | None = None
        self.seed: int | None = None
        self.callbacks = []

    def solve(
            self,
            x: TrialInput,
            x_pass_to_fem_: dict[str, Variable],
            opt_: AbstractOptimizer = None,
    ) -> tuple[float] | None:

        opt_ = opt_ or self
        vm = self.variable_manager

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
            _, dict_or_None_y_internal, c, record = opt_.f(
                x, x_pass_to_fem_, self.history, datetime_start
            )

            # convert dict or None to tuple or None
            y_internal_ = dict_or_None_y_internal if dict_or_None_y_internal is None \
                else tuple(dict_or_None_y_internal.values())

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
        self.current_trial.set_user_attr(optuna_attr.key, optuna_attr.value)

        # check interruption
        self._check_and_raise_interruption()

        return y_internal_

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

            # process main fidelity model
            y_internal: tuple[float] | None = self.solve(x, x_pass_to_fem)

            # process sub_fidelity_models
            for sub_fidelity_name, sub_opt in self.sub_fidelity_models.items():
                self.solve(x, x_pass_to_fem, sub_opt)

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

    def _get_callback(self, n_trials: int):

        # restart である場合、追加 N 回と見做す
        if self.history.is_restart:
            n_existing_trials = len(self.history.get_df(
                equality_filters=History.MAIN_FILTER
            ))
            n_trials = n_trials + n_existing_trials

        return MaxTrialsCallback(n_trials, states=(optuna.trial.TrialState.COMPLETE,))

    def _setup_before_parallel(self):

        if self._done_setup_before_parallel:
            return

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
        if issubclass(self.sampler_class, optuna.samplers.TPESampler) \
                and self.history.is_restart:

            # get unique tmp file
            tmp_storage_path = tempfile.mktemp(suffix='.db')
            self._existing_storage_path = self.storage_path
            self.storage_path = tmp_storage_path

            # load existing study
            existing_study = optuna.load_study(
                study_name=self.study_name,
                storage=f'sqlite:///{self._existing_storage_path}',
            )

            # create new study
            tmp_study = optuna.create_study(
                study_name=self.study_name,
                storage=f'sqlite:///{self.storage_path}',
                load_if_exists=True,
                directions=['minimize'] * len(self.objectives),
            )

            # Copy COMPLETE trials to temporary study.
            existing_trials = existing_study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
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

            # initial trial
            params = self.variable_manager.get_variables(format='dict', filter='parameter')
            study.enqueue_trial(params, user_attrs={"message": "Initial values"})

        self._done_setup_before_parallel = True

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

                # 通常 dispose
                rdb_storage.engine.dispose()

                # run_on_scheduler での dispose
                if client is not None:

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

        # sub fidelity
        if self.sub_fidelity_models is None:
            self.sub_fidelity_models = SubFidelityModels()
            for sub_fidelity_model in self.sub_fidelity_models.values():
                assert sub_fidelity_model.objectives.keys() == self.objectives.keys()
                assert sub_fidelity_model.constraints.keys() == self.constraints.keys()

        # finalize
        self._finalize_history()

        # setup if needed
        self._setup_before_parallel()
        self._setup_after_parallel()

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
                logger.warning(f'与えられた引数 {k} はサンプラー {self.sampler_class.__name__} の'
                               f'引数に含まれません。無視されます。')

            # else, ignore it
            else:
                pass

        # noinspection PyArgumentList
        sampler = self.sampler_class(**actual_sampler_kwargs)

        # sampler specific processes
        from v1.optimizer.optuna_optimizer.pof_botorch.pof_botorch_sampler import PoFBoTorchSampler

        # if PoFBoTorchSampler, set opt
        if isinstance(sampler, PoFBoTorchSampler):
            sampler.pyfemtet_optimizer = self  # FIXME: multi-fidelity に対応できない?

        # quit FEM even if abnormal termination
        with closing(self.fem), self._removing_tmp_db_if_needed():

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
                    existing_study.add_trial(trial)
                self.callbacks.append(copy_back)

            # run
            if self.n_trials is not None:
                self.callbacks.append(self._get_callback(self.n_trials))
            study.optimize(
                self._objective,
                timeout=self.timeout,
                callbacks=self.callbacks,
                catch=[InterruptOptimization]
            )

    def _check_and_raise_interruption(self):
        try:
            AbstractOptimizer._check_and_raise_interruption(self)
        except InterruptOptimization as e:
            if self.current_trial is not None:
                self.current_trial.study.stop()
            raise e


if __name__ == '__main__':
    # from v1.optimizer.optuna_optimizer.pof_botorch.pof_botorch_sampler import
    # sampler = PoFBoTorchSampler(
    #     n_startup_trials=5,
    #     seed=42,
    #     constraints_func=self._constraint,
    #     pof_config=PoFConfig(
    #         # consider_pof=False,
    #         # feasibility_threshold='mean',
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

    from v1.exceptions import PostProcessError

    def _parabola(_fem: AbstractFEMInterface, _opt: AbstractOptimizer) -> float:
        d = _opt.get_variables()
        x1 = d['x1']
        x2 = d['x2']
        # if _cns(_fem, _opt) < 0:
        #     raise PostProcessError
        return x1 ** 2 + x2 ** 2

    def _parabola2(_fem: AbstractFEMInterface, _opt: AbstractOptimizer) -> float:
        x = _opt.get_variables('values')
        return ((x-0.1) ** 2).sum()

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
    _opt.add_constraint('cns', _cns, lower_bound=-0.9, args=(_fem, _opt))
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

    _opt.history.path = 'restart-test.csv'
    _opt.run()

    # import plotly.express as px
    # _df = _opt.history.get_df()
    # px.scatter_3d(_df, x='x1', y='x2', z='obj', color='fidelity', opacity=0.5).show()

    _opt.history.save()
