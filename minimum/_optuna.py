import os

import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from .opt import OptimizerBase


class OptimizerOptuna(OptimizerBase):

    def _objective(self, trial):
        x = []
        for name, (init, lb, ub) in self.parameters.items():
            x.append(trial.suggest_float(name, lb, ub))
        obj_values = self.f(x)
        return tuple(obj_values)

    def _setup_main(self, method):
        # sampler の設定
        self.sampler_kwargs = dict(
            n_startup_trials=5,
        )
        self.sampler_class = optuna.samplers.TPESampler
        if method == 'botorch':
            self.sampler_class = optuna.integration.BoTorchSampler

        # storage の設定
        storage_path = self.history.path.replace(".csv", ".db")
        if os.path.exists(storage_path):
            os.remove(storage_path)
        optuna.create_study(
            study_name='test',
            storage=f'sqlite:///{storage_path}',
            directions=['minimize']*len(self.objectives)
        )

    def _main(self, n_trials=10, subprocess_idx=0):

        # sampler の restore
        sampler = self.sampler_class(seed=42+subprocess_idx, **self.sampler_kwargs)

        # storage の restore
        storage_path = self.history.path.replace(".csv", ".db")
        study = optuna.load_study(
            study_name='test',
            storage=f'sqlite:///{storage_path}',
            sampler=sampler,
        )

        # run
        study.optimize(
            self._objective,
            n_trials=n_trials,
            callbacks=[MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE,))],
        )

        return study
