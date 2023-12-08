import os
import datetime
import functools
import warnings

import numpy as np
import pandas as pd

from .core import FemtetOptimizationCore

import optuna
# optuna.logging.disable_default_handler()
from optuna.study import MaxTrialsCallback
# from optuna.trial import TrialState
from optuna.exceptions import ExperimentalWarning
# optuna.logging.disable_default_handler()

from scipy.stats.qmc import LatinHypercube

from multiprocessing import Process, Value

import gc

warnings.filterwarnings('ignore', category=ExperimentalWarning)


def generate_LHS(bounds, seed=None)->np.ndarray:
    '''
    d 個の変数の bounds を貰い、N 個のサンプル点を得る。
    N は d を超える最小の素数 p に対し N = p**2
    '''
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
            if p%i==0:
                return False
        return True
    
    def get_prime(minimum):
        for p in range(minimum, LIMIT):
            if is_prime(p):
                return p
        
    n = get_prime(d+1)**2
    data = sampler.random(n) # [0,1)
    
    for i, (data_range, datum) in enumerate(zip(bounds, data.T)):
        minimum, maximum = data_range
        band = maximum - minimum
        converted_datum = datum * band + minimum
        data[:,i] = converted_datum
    
    return data # data.shape = (N, d)

# https://optuna.readthedocs.io/en/stable/index.html
class FemtetOptuna(FemtetOptimizationCore):

    def main(
            # 実装固有
            self,
            timeout=None,
            n_trials=None,
            study_name=None,
            use_init_LHS=True, # Latin Hypercube Sampling を初期値にする
            # 共通
            n_parallel=1,
            ):
        '''抽象クラスの main が実行される前に実装固有の設定を行う関数'''

        # 宣言
        self._constraints = []

        # 引数の処理
        self.use_init_LHS = use_init_LHS
        self.n_parallel = n_parallel
        
        #### study name のセットアップ
        # study_name（ファイル名）の設定
        if study_name is None:
            self.study_name = os.path.splitext(os.path.basename(self.history_path))[0] + ".db"
        # csv 保存パスと同じところに db ファイルを保存する
        self.study_db_path = os.path.abspath(
            os.path.join(
                os.path.dirname(self.history_path),
                self.study_name
                )
            )
        self.storage_name = f"sqlite:///{self.study_db_path}"

        #### timeout or n_trials の設定
        self._n_trials = n_trials
        # self.optimize_callbacks = []
        if n_trials is not None:
            self._n_trials = self._n_trials//self.n_parallel
            # self.optimize_callbacks.append(MaxTrialsCallback(n_trials)) # なぜか正常に機能しない
        self.timeout = timeout
        
        #### study の設定
        self._setup_study()

        # 本番処理に移る
        if n_parallel is None:
            super().main()
        else:
            super().main(n_parallel)


    def _setup_study(self):
        #### sampler の設定
        self.sampler_class = optuna.samplers.TPESampler
        self.sampler_kwargs = dict(constraints_func=self._constraint_function)


        #### study ファイルの作成
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_name,
            load_if_exists=True,
            directions=['minimize']*len(self.objectives),
            )

        #### 初期値の設定
        params = self.get_parameter('dict')
        study.enqueue_trial(params, user_attrs={"memo": "initial"})
        
        #### LHS を初期値にする
        if self.use_init_LHS:
            df = self.get_parameter('df')
            names = []
            bounds = []
            for i, row in df.iterrows():
                names.append(row['name'])
                lb = row['lbound']
                ub = row['ubound']
                bounds.append([lb, ub])
            data = generate_LHS(bounds, seed=self.seed)
            for datum in data:
                d = {}
                for name, v in zip(names, datum):
                    d[name] = v
                study.enqueue_trial(d, user_attrs={"memo": "initial Latin Hypercube Sampling"})        
        

    def _main(self, process_idx=0):
        # setup_study が生成した study に接続
        seed = self.seed
        if seed is not None:
            seed += process_idx
        sampler = self.sampler_class(
            seed=seed,
            **self.sampler_kwargs,
        )
        study = optuna.load_study(
            study_name=self.study_name,
            storage=self.storage_name,
            sampler=sampler

        )

        # 最適化の実行
        study.optimize(
            self._objective_function,
            timeout=self.timeout,
            n_trials=self._n_trials,
            # callbacks = self.optimize_callbacks,
            )
        

    #### 目的関数
    def _objective_function(self, trial):

        # 中断指令の処理
        if self.shared_interruption_flag.value == 1:
            self.release_FEM()
            from .core import UserInterruption
            raise UserInterruption('ユーザーによって最適化が取り消されました。')


        #### 変数の設定
        x = []
        df = self.get_parameter('df')
        for i, row in df.iterrows():
            name = row['name']
            lb = row['lbound']
            ub = row['ubound']
            x.append(trial.suggest_float(name, lb, ub))
        x = np.array(x)

        #### user_attribute の設定
        # memo
        try:
            # あれば
            message = trial.user_attrs["memo"]
        except (AttributeError, KeyError):
            # なければ
            message = ''
        # 拘束のアトリビュート（Prune などの対策、計算できたら後で上書きする）
        dummy_data = tuple([1 for f in self._constraints])
        trial.set_user_attr("constraint", dummy_data)
        
        #### strict 拘束の計算
        # is_calcrated の直後まで変数をアップデートしてはいけないため
        # 一度 buff に現在の変数を控え、処理が終わったら戻す
        # 退避+更新
        buff = self._parameter.copy()
        self._parameter['value'] = x
        # restore の注意 / strict 拘束の処理から抜けうるところには
        # すべてに以下の処理を配置すること
        # self._parameter = buff
        # Femtet 関係の FEM システムであれば変数を更新
        from .core import Femtet, ModelError
        if isinstance(self.FEM, Femtet):
            try:
                # ModelError が起きうる
                self.FEM.update_model(self._parameter)
            except ModelError:
                raise optuna.TrialPruned()
        # strict 拘束の計算
        val_lb_ub_fn_list = [[cns.calc(), cns.lb, cns.ub, cns.f.__name__] for cns in self.constraints if cns.strict==True]
        for val, lb, ub, fn in val_lb_ub_fn_list:
            if lb is not None:
                if not (lb <= val):
                    print(f'拘束 {fn} が満たされませんでした。')
                    print('変数の組み合わせは以下の通りです。')
                    print(self._parameter)
                    self._parameter = buff # Femtet は戻す必要ない、以下同文
                    raise optuna.TrialPruned()
            if ub is not None:
                if not (val <= ub):
                    self._parameter = buff
                    print(f'拘束 {fn} が満たされませんでした。')
                    print('変数の組み合わせは以下の通りです。')
                    print(self._parameter)
                    raise optuna.TrialPruned()
        self._parameter = buff

        #### 解析実行
        # ModelError 等が起こりうる
        from .core import ModelError, MeshError, SolveError
        try:
            # parameters を更新し、解析、目的関数、全ての拘束の計算
            ojectiveValuesToMinimize = self.f(x, optimization_message=message)
        except (ModelError, MeshError, SolveError):
            raise optuna.TrialPruned()
        
        #### 拘束の計算
        _constraint_values = [] # 非正なら OK
        for i, cns in enumerate(self.constraints):
            lb, ub = cns.lb, cns.ub
            if lb is not None: # fun >= lb  <=>  lb - fun <= 0
                _constraint_values.append(lb - cns.calc())
            if ub is not None: # ub >= fun  <=>  fun - ub <= 0
                _constraint_values.append(cns.calc() - ub)
        trial.set_user_attr("constraint", _constraint_values)
        
        # 一応
        gc.collect()

        # 結果
        return tuple(ojectiveValuesToMinimize)


    #### 拘束関数(experimental)
    def _constraint_function(self, trial):
        return trial.user_attrs["constraint"]





