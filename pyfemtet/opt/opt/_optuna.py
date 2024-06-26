# typing
from typing import Iterable

# built-in
import os

# 3rd-party
import numpy as np
import optuna
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback

# pyfemtet relative
from pyfemtet.opt._femopt_core import OptimizationStatus, generate_lhs
from pyfemtet.opt.opt import AbstractOptimizer, logger
from pyfemtet.core import MeshError, ModelError, SolveError

# filter warnings
import warnings
from optuna.exceptions import ExperimentalWarning
warnings.filterwarnings('ignore', category=ExperimentalWarning)


class OptunaOptimizer(AbstractOptimizer):

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

    def _objective(self, trial):

        # 中断の確認 (FAIL loop に陥る対策)
        if self.entire_status.get() == OptimizationStatus.INTERRUPTING:
            self.worker_status.set(OptimizationStatus.INTERRUPTING)
            trial.study.stop()  # 現在実行中の trial を最後にする
            return None  # set TrialState FAIL

        # candidate x
        x = []
        for i, row in self.parameters.iterrows():
            v = trial.suggest_float(row['name'], row['lb'], row['ub'])
            x.append(v)
        x = np.array(x).astype(float)

        # message の設定
        self.message = trial.user_attrs['message'] if 'message' in trial.user_attrs.keys() else ''

        # fem や opt 経由で変数を取得して constraint を計算する時のためにアップデート
        self.parameters['value'] = x
        self.fem.update_parameter(self.parameters)

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
                logger.info(f'以下の変数で拘束 {cns.name} が満たされませんでした。')
                print(self.get_parameter('dict'))
                raise optuna.TrialPruned()  # set TrialState PRUNED because FAIL causes similar candidate loop.

        # 計算
        try:
            _, _y, c = self.f(x)
        except (ModelError, MeshError, SolveError) as e:
            logger.info(e)
            logger.info('以下の変数で FEM 解析に失敗しました。')
            print(self.get_parameter('dict'))

            # 中断の確認 (解析中に interrupt されている場合対策)
            if self.entire_status.get() == OptimizationStatus.INTERRUPTING:
                self.worker_status.set(OptimizationStatus.INTERRUPTING)
                trial.study.stop()  # 現在実行中の trial を最後にする
                return None  # set TrialState FAIL

            raise optuna.TrialPruned()  # set TrialState PRUNED because FAIL causes similar candidate loop.

        # 拘束 attr の更新
        _c = []  # 非正なら OK
        for (name, cns), c_value in zip(self.constraints.items(), c):
            lb, ub = cns.lb, cns.ub
            if lb is not None:  # fun >= lb  <=>  lb - fun <= 0
                _c.append(lb - c_value)
            if ub is not None:  # ub >= fun  <=>  fun - ub <= 0
                _c.append(c_value - ub)
        trial.set_user_attr('constraint', _c)

        # 中断の確認 (解析中に interrupt されている場合対策)
        if self.entire_status.get() == OptimizationStatus.INTERRUPTING:
            self.worker_status.set(OptimizationStatus.INTERRUPTING)
            trial.study.stop()  # 現在実行中の trial を最後にする
            return None  # set TrialState FAIL

        # 結果
        return tuple(_y)

    def _constraint(self, trial):
        return trial.user_attrs['constraint'] if 'constraint' in trial.user_attrs.keys() else (1,)  # infeasible

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
                n_existing_trials = len(self.history.actor_data)
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
                params = self.get_parameter('dict')
                self.study.enqueue_trial(params, user_attrs={"message": "initial"})

                # add_initial_parameter で追加された初期値
                for prm, prm_set_name in self.additional_initial_parameter:
                    if type(prm) is dict:
                        assert prm.keys() == params.keys(), '設定されたパラメータ名と add_init_parameter で追加されたパラメータ名が一致しません。'
                    else:
                        assert len(prm) == len(params.keys()), '設定されたパラメータ数と add_init_parameter で追加されたパラメータ数が一致しません。'
                        prm = dict(zip(params.keys(), prm))

                    self.study.enqueue_trial(
                        prm,
                        user_attrs={"message": prm_set_name}
                    )

                # add_init で指定された方法による初期値
                if 'LHS' in self.additional_initial_methods:
                    names = []
                    bounds = []
                    for i, row in self.parameters.iterrows():
                        names.append(row['name'])
                        lb = row['lb']
                        ub = row['ub']
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
                msg = f'{storage_path} が見つかりません。'
                msg += '.db ファイルは .csv ファイルと同じフォルダに生成されます。'
                msg += 'クラスター解析の場合は、スケジューラを起動したフォルダに生成されます。'
                raise FileNotFoundError(msg)
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
        sampler = self.sampler_class(
            seed=seed,
            constraints_func=self._constraint,
            **self.sampler_kwargs
        )

        # load study
        study = optuna.load_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=sampler,
        )

        # run
        study.optimize(
            self._objective,
            timeout=self.timeout,
            callbacks=self.optimize_callbacks,
        )
