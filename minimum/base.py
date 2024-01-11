from threading import Thread
from time import sleep
import numpy as np
import pandas as pd
import optuna
from minimum.core import OptimizationState, History
from minimum.monitor import Monitor
from dask.distributed import Client


class OptimizationBase:

    def __init__(self):
        self.parameters = pd.DataFrame()
        self.objectives = dict()
        self.fem = None
        self.history = History()  # actor にする
        self.monitor = Monitor(self)
        self.state = OptimizationState()  # TODO: actor にする
        self.state.state = 'ready'

    def set_parameters(self, d):
        self.parameters = pd.DataFrame(d)

    def set_objective(self, fun, name, *args):
        self.objectives[name] = [fun, args]

    def set_fem(self, fem):
        self.fem = fem

    def f(self, x):
        if self.state.state == 'interrupted':
            raise Exception('interrupted')

        self.parameters['value'] = x
        self.fem.update(self.parameters)
        y = []
        for name, [fun, args] in self.objectives.items():
            y.append(fun(x, *args))
        self.history.record(x, y, cns_values=[])
        return np.array(y)

    def setup_main(self):
        pass

    def concrete_main(self):
        pass

    def main(self):
        # before parallel
        self.state.state = 'setup'
        self.history.init(self.parameters['name'].values, list(self.objectives.keys()))
        self.setup_main()

        # monitor
        t_monitor = Thread(target=self.monitor.start_server)
        t_monitor.start()

        # parallel
        self.state.state = 'running'
        t = Thread(target=self.concrete_main)
        t.start()

        # await
        while True:
            if not t.is_alive():
                break
            sleep(1)
        t.join()

        # finalize
        self.state.state = 'terminated'



class OptimizationOptuna(OptimizationBase):

    def __init__(self):
        self.sampler = None
        self.study = None
        self.storage = None
        super().__init__()

    def _objective(self, trial):
        # x の作成
        x = []
        for i, row in self.parameters.iterrows():
            v = trial.suggest_float(row['name'], row['lb'], row['ub'])
            x.append(v)
        x = np.array(x)

        # FEM の実行
        return tuple(self.f(x))

    def setup_main(self):
        """Main process から呼ばれる関数"""

        # storage
        self.storage = 'sqlite:///' + self.history.path.replace('.csv', '.db')

        # sampler
        self.sampler = optuna.samplers.TPESampler()  # TODO: actor にする必要があるかどうかの調査

        # study
        self.study = optuna.create_study(
            storage=self.storage,
            sampler=self.sampler,
            directions=['minimize'] * len(self.objectives),
        )

    def concrete_main(self):
        """Dask worker から呼ばれる関数"""

        # study
        study = optuna.load_study(study_name=None, storage=self.storage)

        # run
        study.optimize(self._objective, n_trials=30)



