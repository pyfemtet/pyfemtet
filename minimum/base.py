import sys
import datetime
from threading import Thread
from time import sleep
import numpy as np
import pandas as pd
import optuna
from pythoncom import CoInitialize, CoUninitialize

from minimum.core import OptimizationState, History
from minimum.monitor import Monitor
from minimum.fem import Femtet
from dask.distributed import Client, LocalCluster
from concurrent.futures._base import CancelledError


class OptimizationMethodBase:

    def __init__(self):
        self.fem = None
        self.parameters = pd.DataFrame()
        self.objectives = dict()
        self.state = None  # shared
        self.history = None  # shared

    def f(self, x):
        self.parameters['value'] = x
        self.fem.update(self.parameters)
        y = []
        for name, [fun, args] in self.objectives.items():
            y.append(fun(x, *args))
        self.history.record(x, y, []).result()
        return np.array(y)

    def set_fem(self):
        CoInitialize()
        self.fem = Femtet()

    # def __del__(self):
    #     CoUninitialize()  # Win32 exception occurred releasing IUnknown at 0x0000022427692748


class OptimizationMethod(OptimizationMethodBase):

    def __init__(self):
        super().__init__()
        self.storage = None
        self.sampler = None
        self.study = None

    def _objective(self, trial):
        # x の作成
        x = []
        for i, row in self.parameters.iterrows():
            v = trial.suggest_float(row['name'], row['lb'], row['ub'])
            x.append(v)
        x = np.array(x)

        # 計算
        y = self.f(x)

        # 中断有無
        state = self.state.get_state().result()
        if state == 'interrupted':
            trial.study.stop()  # 現在実行中の trial を最後にする

        # 結果
        return tuple(y)

    def setup_before_parallel(self):
        """Main process から呼ばれる関数"""

        # storage
        # self.storage = datetime.datetime.now().strftime('sqlite:///%Y%m%d_%H%M%S.db')
        self.storage = optuna.integration.dask.DaskStorage(datetime.datetime.now().strftime('sqlite:///%Y%m%d_%H%M%S.db'))

        # study
        self.study = optuna.create_study(
            storage=self.storage,
            directions=['minimize'] * len(self.objectives),
        )

    def main(self, subprocess_idx):
        """Dask worker から呼ばれる関数"""
        print(subprocess_idx, 'started')
        self.set_fem()

        # study
        study = optuna.load_study(study_name=None, storage=self.storage)

        # run
        study.optimize(self._objective, n_trials=30)


class OptimizationBase:

    def __init__(self):
        self.parameters = pd.DataFrame()
        self.objectives = dict()
        self.opt = None

        # parallel setup
        scheduler = 'tcp://xxxx.xxxx.xxxx.xxxx:xxxx'
        self.client = Client(scheduler)
        # cluster = LocalCluster(processes=True, threads_per_worker=1)
        # self.client = Client(cluster, direct_to_workers=False)
        self.state = self.client.submit(OptimizationState, actor=True).result()
        self.state.set_state('ready').result()
        self.history = self.client.submit(History, actor=True).result()

    def set_parameters(self, d):
        self.parameters = pd.DataFrame(d)

    def set_objective(self, fun, name, *args):
        self.objectives[name] = [fun, args]

    def set_opt(self, opt):
        self.opt = opt

    def main(self, n_parallel=3):
        # before parallel
        self.state.set_state('setup').result()
        self.set_opt(OptimizationMethod())
        self.opt.objectives = self.objectives
        self.opt.parameters = self.parameters
        self.opt.state = self.state
        self.history.init(prm_names=self.parameters['name'], obj_names=list(self.objectives.keys())).result()
        self.opt.history = self.history
        self.opt.setup_before_parallel()

        # monitor
        monitor = Monitor(self)
        t = Thread(target=monitor.start_server)
        t.start()

        # state
        self.state.set_state('running').result()

        # parallel fem
        calc_futures = self.client.map(self.opt.main, list(range(n_parallel)))

        self.client.gather(calc_futures)
        self.state.set_state('terminated').result()
