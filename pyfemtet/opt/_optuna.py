import os
import gc
import warnings

import numpy as np
from scipy.stats.qmc import LatinHypercube
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from optuna.exceptions import ExperimentalWarning
# optuna.logging.disable_default_handler()

from .core import UserInterruption, ModelError, MeshError, SolveError
from .base import OptimizerBase


warnings.filterwarnings('ignore', category=ExperimentalWarning)


def generate_lhs(bounds, seed=None) -> np.ndarray:
    """
    d 個の変数の bounds を貰い、N 個のサンプル点を得る。
    N は d を超える最小の素数 p に対し N = p**2
    """
    d = len(bounds)

    sampler = LatinHypercube(
        d,
        scramble=False,
        strength=2,
        # optimization='lloyd',
        optimization='random-cd',
        seed=seed,
    )

    LIMIT = 100

    def is_prime(p):
        for i in range(2, p):
            if p % i == 0:
                return False
        return True

    def get_prime(minimum):
        for p in range(minimum, LIMIT):
            if is_prime(p):
                return p

    n = get_prime(d + 1) ** 2
    data = sampler.random(n)  # [0,1)

    for i, (data_range, datum) in enumerate(zip(bounds, data.T)):
        minimum, maximum = data_range
        band = maximum - minimum
        converted_datum = datum * band + minimum
        data[:, i] = converted_datum

    return data  # data.shape = (N, d)


class OptimizerOptuna(OptimizerBase):

    def _objective(self, trial):

        # 中断指令の処理
        if self.ipv.get_state() == 'interrupted':
            raise UserInterruption

        # message の設定
        try:
            message = trial.user_attrs["message"]  # あれば
        except (AttributeError, KeyError):
            message = ''  # なければ

        # x の生成
        x = []
        for i, row in self.parameters.iterrows():
            x.append(trial.suggest_float(row['name'], row['lb'], row['ub']))
        x = np.array(x)

        # strict 拘束の計算で Prune することになったとき
        # constraint attr がないとエラーになるのでダミーを置いておく
        trial.set_user_attr("constraint", (1.,))  # 非正が feasible 扱い

        # strict 拘束の計算
        self.parameters['value'] = x
        tmp = [[cns.calc(self.fem), cns.lb, cns.ub, name] for name, cns in self.constraints.items() if cns.strict]
        for val, lb, ub, name in tmp:
            if lb is not None:
                if not (lb <= val):
                    print(f'拘束 {name} が満たされませんでした。')
                    print('変数の組み合わせは以下の通りです。')
                    print(self.parameters)
                    raise optuna.TrialPruned()
            if ub is not None:
                if not (val <= ub):
                    print(f'拘束 {name} が満たされませんでした。')
                    print('変数の組み合わせは以下の通りです。')
                    print(self.parameters)
                    raise optuna.TrialPruned()

        # 解析実行
        try:
            obj_values = self.f(x, message)  # obj_val と cns_val の更新
        except (ModelError, MeshError, SolveError):
            raise optuna.TrialPruned()

        # 拘束 attr の更新
        _cns_values = []  # 非正なら OK
        for (name, cns), cns_val in zip(self.constraints.items(), self.cns_values):
            lb, ub = cns.lb, cns.ub
            if lb is not None:  # fun >= lb  <=>  lb - fun <= 0
                _cns_values.append(lb - cns_val)
            if ub is not None:  # ub >= fun  <=>  fun - ub <= 0
                _cns_values.append(cns_val - ub)
        trial.set_user_attr("constraint", _cns_values)

        # 一応
        gc.collect()

        return tuple(obj_values)

    def _constraint_function(self, trial):
        return trial.user_attrs["constraint"]

    def _setup_main(self, use_lhs_init=True):

        # sampler の設定
        self.sampler_kwargs = dict(
            # n_startup_trials=5,
            constraints_func=self._constraint_function
        )
        self.sampler_class = optuna.samplers.TPESampler
        if self.method == 'botorch':
            self.sampler_class = optuna.integration.BoTorchSampler

        # study name の設定
        self.study_name = os.path.splitext(os.path.basename(self.history.path))[0]
        self.storage_path = os.path.splitext(self.history.path)[0] + '.db'
        self.storage = f"sqlite:///{self.storage_path}"

        # storage の設定
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            directions=['minimize']*len(self.objectives),
            )

        # 初期値の設定
        params = self.get_parameter('dict')
        study.enqueue_trial(params, user_attrs={"message": "initial"})

        # LHS を初期値にする
        if use_lhs_init:
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
                study.enqueue_trial(d, user_attrs={"message": "initial Latin Hypercube Sampling"})


    def _main(self, subprocess_idx=0):

        # 乱数シードをプロセス固有にする
        seed = self.seed
        if seed is not None:
            seed = seed + (1 + subprocess_idx)  # main process と subprocess0 が重複する

        # sampler の restore
        sampler = self.sampler_class(
            seed=seed,
            **self.sampler_kwargs
        )

        # storage の restore
        study = optuna.load_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=sampler,
        )

        # run
        study.optimize(
            self._objective,
            timeout=self.timeout,
            callbacks=[MaxTrialsCallback(self.n_trials, states=(TrialState.COMPLETE,))],
        )

        return study
