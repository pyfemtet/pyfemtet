# typing
from typing import Iterable

# built-in
import os
import inspect

# 3rd-party
import optuna
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback

# pyfemtet relative
from pyfemtet.opt._femopt_core import OptimizationStatus, generate_lhs
from pyfemtet.opt.optimizer import AbstractOptimizer, logger, OptimizationMethodChecker
from pyfemtet.core import MeshError, ModelError, SolveError
from pyfemtet._message import Msg

# filter warnings
import warnings
from optuna.exceptions import ExperimentalWarning


optuna.logging.set_verbosity(optuna.logging.ERROR)

warnings.filterwarnings('ignore', category=ExperimentalWarning)


class OptunaMethodChecker(OptimizationMethodChecker):
    def check_multi_objective(self, raise_error=True): return True
    def check_timeout(self, raise_error=True): return True
    def check_parallel(self, raise_error=True): return True
    def check_constraint(self, raise_error=True): return True
    def check_strict_constraint(self, raise_error=True): return True
    def check_skip(self, raise_error=True): return True
    def check_seed(self, raise_error=True): return True


class OptunaOptimizer(AbstractOptimizer):
    """Optimizer using ```optuna```.

    This class provides an interface for the optimization
    engine using Optuna. For more details, please refer to
    the Optuna documentation.

    See Also:
        https://optuna.readthedocs.io/en/stable/reference/index.html

    Args:
        sampler_class (optuna.samplers.BaseSampler, optional):
            A sampler class from Optuna. If not specified,
            ```optuna.samplers.TPESampler``` is specified.
            This defines the sampling strategy used during
            optimization. Defaults to None.
        sampler_kwargs (dict, optional):
            A dictionary of keyword arguments to be passed to
            the sampler class. This allows for customization
            of the sampling process. Defaults to None.
        add_init_method (str or Iterable[str], optional):
            A method or a collection of methods to be added
            during initialization. This can be used to specify
            additional setup procedures.
            Currently, the only valid value is 'LHS'
            (using Latin Hypercube Sampling).
            Defaults to None.

    Warnings:
        Do not include ```constraints_func``` in ```sampler_kwargs```.
        It is generated and provided by :func:`FEMOpt.add_constraint`.

    """

    def __init__(
            self,
            sampler_class: optuna.samplers.BaseSampler or None = None,
            sampler_kwargs: dict or None = None,
            add_init_method: str or Iterable[str] or None = None
    ):
        super().__init__()
        self.study_name = None
        self.storage = None
        self.study = None
        self.optimize_callbacks = []
        self.sampler_class = optuna.samplers.TPESampler if sampler_class is None else sampler_class
        self.sampler_kwargs = dict() if sampler_kwargs is None else sampler_kwargs
        self.additional_initial_parameter = []
        self.additional_initial_methods = add_init_method if hasattr(add_init_method, '__iter__') else [add_init_method]
        self.method_checker = OptunaMethodChecker(self)

    def _objective(self, trial):

        logger.info('')
        if self._retry_counter == 0:
            logger.info(f'===== trial {1 + len(self.history.get_df())} start =====')
        else:
            logger.info(f'===== trial {1 + len(self.history.get_df())} (retry {self._retry_counter}) start =====')

        # 中断の確認 (FAIL loop に陥る対策)
        if self.entire_status.get() == OptimizationStatus.INTERRUPTING:
            self.worker_status.set(OptimizationStatus.INTERRUPTING)
            trial.study.stop()  # 現在実行中の trial を最後にする
            self._retry_counter = 0
            return None  # set TrialState FAIL

        # candidate x and update parameters
        logger.info('Searching new parameter set...')
        for prm in self.variables.get_variables(format='raw', filter_parameter=True):
            value = trial.suggest_float(
                name=prm.name,
                low=prm.lower_bound,
                high=prm.upper_bound,
                step=prm.step,
            )
            self.variables.variables[prm.name].value = value

        # update expressions
        self.variables.evaluate()

        # message の設定
        self.message = trial.user_attrs['message'] if 'message' in trial.user_attrs.keys() else ''

        # fem 経由で変数を取得して constraint を計算する時のためにアップデート
        df_fem = self.variables.get_variables(format='df', filter_pass_to_fem=True)
        self.fem.update_parameter(df_fem)

        # strict 拘束
        strict_constraints = [cns for cns in self.constraints.values() if cns.strict]
        for cns in strict_constraints:
            feasible = True
            cns_value = cns.calc(self.fem)
            if cns.lb is not None:
                feasible = feasible and (cns_value >= cns.lb)
            if cns.ub is not None:
                feasible = feasible and (cns.ub >= cns_value)
            if not feasible:
                logger.info('----- Out of constraint! -----')
                logger.info(Msg.INFO_INFEASIBLE)
                logger.info(f'Constraint: {cns.name}')
                logger.info(self.variables.get_variables('dict', filter_parameter=True))
                self._retry_counter += 1
                raise optuna.TrialPruned()  # set TrialState PRUNED because FAIL causes similar candidate loop.

        # 計算
        x = self.variables.get_variables(format='values', filter_parameter=True)
        try:
            _, _y, c = self.f(x)  # f の中で info は出している
        except (ModelError, MeshError, SolveError) as e:
            # 中断の確認 (解析中に interrupt されている場合対策)
            if self.entire_status.get() == OptimizationStatus.INTERRUPTING:
                self.worker_status.set(OptimizationStatus.INTERRUPTING)
                trial.study.stop()  # 現在実行中の trial を最後にする
                return None  # set TrialState FAIL

            logger.warning('----- Infeasible! -----')
            logger.warning(Msg.INFO_INFEASIBLE)
            logger.warning(f'Hidden Constraint ({type(e).__name__})')
            logger.warning(self.variables.get_variables('dict', filter_parameter=True))
            logger.warning('Please consider to determine the cause'
                           'of the above error and modify the model'
                           'or analysis.')

            self._retry_counter += 1
            raise optuna.TrialPruned()  # set TrialState PRUNED because FAIL causes similar candidate loop.

        # 拘束 attr の更新
        if len(self.constraints) > 0:
            _c = []  # <= 0 is feasible
            for (name, cns), c_value in zip(self.constraints.items(), c):
                lb, ub = cns.lb, cns.ub
                if lb is not None:  # fun >= lb  <=>  lb - fun <= 0
                    _c.append(lb - c_value)
                if ub is not None:  # ub >= fun  <=>  fun - ub <= 0
                    _c.append(c_value - ub)
            trial.set_user_attr('constraints', _c)

        # 中断の確認 (解析中に interrupt されている場合対策)
        if self.entire_status.get() == OptimizationStatus.INTERRUPTING:
            self.worker_status.set(OptimizationStatus.INTERRUPTING)
            trial.study.stop()  # 現在実行中の trial を最後にする
            self._retry_counter = 0
            return None  # set TrialState FAIL

        # 結果
        self._retry_counter = 0
        return tuple(_y)

    def _constraint(self, trial):
        # if break trial without weak constraint calculation, return 1 (as infeasible).
        if 'constraints' in trial.user_attrs.keys():
            return trial.user_attrs['constraints']
        else:
            _c = []
            for name, cns in self.constraints.items():
                lb, ub = cns.lb, cns.ub
                if lb is not None:
                    _c.append(1.)
                if ub is not None:
                    _c.append(1.)
            return _c

    def _setup_before_parallel(self):
        """Create storage, study and set initial parameter."""

        # create storage
        self.study_name = os.path.basename(self.history.path)
        storage_path = self.history.path.replace('.csv', '.db')  # history と同じところに保存
        if self.is_cluster:  # remote cluster なら scheduler の working dir に保存
            storage_path = os.path.basename(self.history.path).replace('.csv', '.db')

        # callback to terminate
        if self.n_trials is not None:
            n_trials = self.n_trials

            # restart である場合、追加 N 回と見做す
            if self.history.is_restart:
                n_existing_trials = len(self.history.get_df())
                n_trials += n_existing_trials

            self.optimize_callbacks.append(MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE,)))

        # if not restart, create study if storage is not exists
        if not self.history.is_restart:

            self.storage = optuna.integration.dask.DaskStorage(
                f'sqlite:///{storage_path}',
            )

            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                load_if_exists=True,
                directions=['minimize'] * len(self.objectives),
            )

            # 初期値の設定
            if len(self.study.trials) == 0:  # リスタートでなければ
                # ユーザーの指定した初期値
                params = self.variables.get_variables('dict', filter_parameter=True)
                self.study.enqueue_trial(params, user_attrs={"message": "initial"})

                # add_initial_parameter で追加された初期値
                for prm, prm_set_name in self.additional_initial_parameter:
                    if type(prm) is dict:
                        assert prm.keys() == params.keys(), Msg.ERR_INCONSISTENT_PARAMETER
                    else:
                        assert len(prm) == len(params.keys()), Msg.ERR_INCONSISTENT_PARAMETER
                        prm = dict(zip(params.keys(), prm))

                    self.study.enqueue_trial(
                        prm,
                        user_attrs={"message": prm_set_name}
                    )

                # add_init で指定された方法による初期値
                if 'LHS' in self.additional_initial_methods:
                    names = []
                    bounds = []
                    for i, row in self.get_parameter('df').iterrows():
                        names.append(row['name'])
                        lb = row['lower_bound']
                        ub = row['upper_bound']
                        bounds.append([lb, ub])
                    data = generate_lhs(bounds, seed=self.seed)
                    for datum in data:
                        d = {}
                        for name, v in zip(names, datum):
                            d[name] = v
                        self.study.enqueue_trial(
                            d, user_attrs={"message": "additional initial (Latin Hypercube Sampling)"}
                        )

        # if is_restart, load study
        else:
            if not os.path.exists(storage_path):
                raise FileNotFoundError(storage_path)
            self.storage = optuna.integration.dask.DaskStorage(
                f'sqlite:///{storage_path}',
            )

    def add_init_parameter(
            self,
            parameter: dict or Iterable,
            name: str or None = None,
    ):
        """Add additional initial parameter for evaluate.

        The parameter set is ignored if the main() is continued.

        Args:
            parameter (dict or Iterable): Parameter to evaluate before run optimization algorithm.
            name (str or None): Optional. If specified, the name is saved in the history row. Default to None.

        """
        if name is None:
            name = 'additional initial'
        else:
            name = f'additional initial ({name})'
        self.additional_initial_parameter.append([parameter, name])

    def run(self):
        """Set random seed, sampler, study and run study.optimize()."""

        # (re)set random seed
        seed = self.seed
        if seed is not None:
            if self.subprocess_idx is not None:
                seed += self.subprocess_idx

        # restore sampler
        if len(self.constraints) > 0:
            self.sampler_kwargs.update(
                constraints_func=self._constraint
            )
        if seed is not None:
            self.sampler_kwargs.update(
                seed=seed
            )
        parameters = inspect.signature(self.sampler_class.__init__).parameters
        sampler_kwargs = dict()
        for k, v in self.sampler_kwargs.items():
            if k in parameters.keys():
                sampler_kwargs.update({k: v})
        sampler = self.sampler_class(
            **sampler_kwargs
        )

        from pyfemtet.opt.optimizer._optuna._pof_botorch import PoFBoTorchSampler
        if isinstance(sampler, PoFBoTorchSampler):
            sampler._pyfemtet_constraints = [cns for cns in self.constraints.values() if cns.strict]
            sampler._pyfemtet_optimizer = self

        # load study
        study = optuna.load_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=sampler,
        )

        # 一時的な実装。
        #   TPESampler の場合、リスタート時などの場合で、
        #   Pruned が多いとエラーを起こす挙動があるので、
        #   Pruned な Trial は remove したい。
        #   study.remove_trial がないので、一度ダミー
        #   study を作成して最適化終了後に結果をコピーする。
        if isinstance(sampler, optuna.samplers.TPESampler):
            tmp_db = f"tmp{self.subprocess_idx}.db"
            if os.path.exists(tmp_db):
                os.remove(tmp_db)

            _study = optuna.create_study(
                study_name="tmp",
                storage=f"sqlite:///{tmp_db}",
                sampler=sampler,
                directions=['minimize']*len(self.objectives),
                load_if_exists=False,
            )

            # 既存の trials のうち COMPLETE のものを取得
            existing_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
            _study.add_trials(existing_trials)

            # run
            _study.optimize(
                self._objective,
                timeout=self.timeout,
                callbacks=self.optimize_callbacks,
            )

            # trial.number と trial_id は _study への add_trials 時に
            # 振りなおされるため重複したものをフィルタアウトするために
            # datetime_start を利用。
            added_trials = []
            for _trial in _study.get_trials():
                if _trial.datetime_start not in [t.datetime_start for t in existing_trials]:
                    added_trials.append(_trial)

            # Write back added trials to the existing study.
            study.add_trials(added_trials)

            # clean up
            from optuna.storages import get_storage
            storage = get_storage(f"sqlite:///{tmp_db}")
            storage.remove_session()
            del _study
            del storage
            import gc
            gc.collect()
            if os.path.exists(tmp_db):
                os.remove(tmp_db)

        else:
            # run
            study.optimize(
                self._objective,
                timeout=self.timeout,
                callbacks=self.optimize_callbacks,
            )
