import os

import optuna
from .opt import OptimizerBase


class OptimizerOptuna(OptimizerBase):

    def _objective(self, trial):
        x = []
        for name, (init, lb, ub) in self.parameters.items():
            x.append(trial.suggest_float(name, lb, ub))
        obj_values = self.f(x)
        return tuple(obj_values)

    def _main(self, n_trials=10, method='TPE'):

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=5,
        )
        if method == 'botorch':
            sampler = optuna.integration.BoTorchSampler(
                n_startup_trials=5,
            )

        storage_path = self.history.path.replace(".csv", ".db")

        if os.path.exists(storage_path):
            os.remove(storage_path)

        study = optuna.create_study(
            study_name='test',
            storage=f'sqlite:///{storage_path}',
            sampler=sampler,
            directions=['minimize']*len(self.objectives)
        )

        study.optimize(
            self._objective,
            n_trials=n_trials,
        )

        return study
