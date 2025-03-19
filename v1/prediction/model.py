import numpy as np
import pandas as pd

from optuna._transform import _SearchSpaceTransform

import torch

from v1.history import *
from v1.prediction.helper import *
from v1.optimizer.optuna_optimizer.pof_botorch.pof_botorch_sampler import setup_gp, SingleTaskGP


__all__ = [
    'PyFemtetModel',
    'AbstractModel',
    'SingleTaskGPModel',
    'AbstractModel',
]


class AbstractModel:

    def fit(self, x: np.ndarray, y: np.ndarray, bounds: np.ndarray = None, **kwargs): ...
    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class SingleTaskGPModel(AbstractModel):

    KWARGS = dict(dtype=torch.float64, device='cpu')
    gp: SingleTaskGP

    def fit(self, x: np.ndarray, y: np.ndarray, bounds: np.ndarray = None,
            observation_noise=None, likelihood_class=None):
        X = torch.tensor(x, **self.KWARGS)
        Y = torch.tensor(y, **self.KWARGS)
        B = torch.tensor(bounds, **self.KWARGS).transpose(1, 0)
        self.gp = setup_gp(X, Y, B, observation_noise, likelihood_class)

    def predict(self, x: np.ndarray):
        assert hasattr(self, 'gp')
        X = torch.tensor(x, **self.KWARGS)
        post = self.gp.posterior(X)
        with torch.no_grad():
            mean = post.mean.numpy()
            std = post.variance.sqrt().numpy()
        return mean, std


class PyFemtetModel:

    current_trans: _SearchSpaceTransform
    current_model: AbstractModel
    history: History

    def update_model(self, model: AbstractModel):
        self.current_model = model

    def fit(self, history: History, df: pd.DataFrame, **kwargs):
        assert hasattr(self, 'current_model')

        self.history = history

        # remove nan from df
        df = df.dropna(subset=history.obj_names + history.prm_names, how='any')

        # set current trans
        self.current_trans = get_transform_0_1(df, history)

        # transform all values
        transformed_x = get_transformed_params(df, history, self.current_trans)

        # bounds as setup maximum range
        bounds = self.current_trans.bounds

        y = df[history.obj_names].values
        self.current_model.fit(transformed_x, y, bounds, **kwargs)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert hasattr(self, 'history')
        assert hasattr(self, 'current_trans')

        transformed_x = get_transformed_params(x, self.history, self.current_trans)
        return self.current_model.predict(transformed_x)
