import numpy as np
import pandas as pd

from optuna._transform import _SearchSpaceTransform
import torch
from botorch.models import SingleTaskGP

from pyfemtet.opt.history import *
from pyfemtet.opt.prediction._helper import *
from pyfemtet.opt.prediction._botorch_utils import *
from pyfemtet.opt.prediction._gpytorch_modules_extension import get_covar_module_with_dim_scaled_prior_extension


__all__ = [
    'PyFemtetModel',
    'AbstractModel',
    'SingleTaskGPModel',
]


class AbstractModel:

    def fit(self, x: np.ndarray, y: np.ndarray, bounds: np.ndarray = None, **kwargs): ...
    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class SingleTaskGPModel(AbstractModel):

    KWARGS = dict(dtype=torch.float64, device='cpu')
    gp: SingleTaskGP

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            bounds: np.ndarray = None,
            observation_noise=None,
            likelihood_class=None,
            covar_module_settings: dict = None,
    ):

        covar_module = None

        X = torch.tensor(x, **self.KWARGS)
        Y = torch.tensor(y, **self.KWARGS)
        B = torch.tensor(bounds, **self.KWARGS).transpose(1, 0) if bounds is not None else None

        if covar_module_settings is not None:
            if covar_module_settings['name'] == 'matern_kernel_with_gamma_prior':
                covar_module_settings.pop('name')
                covar_module = get_matern_kernel_with_gamma_prior_as_covar_module(
                    X, Y,
                    **covar_module_settings,
                )
            elif covar_module_settings['name'] == 'get_covar_module_with_dim_scaled_prior_extension':
                covar_module_settings.pop('name')

                _input_batch_shape, _aug_batch_shape = SingleTaskGP.get_batch_dimensions(X, Y)
                batch_shape = _aug_batch_shape

                covar_module = get_covar_module_with_dim_scaled_prior_extension(
                    ard_num_dims=X.shape[-1],
                    batch_shape=batch_shape,
                    **covar_module_settings,
                )
            else:
                raise NotImplementedError(f'{covar_module_settings["name"]=}')

        self.gp = setup_gp(X, Y, B, observation_noise, likelihood_class, covar_module)

    def predict(self, x: np.ndarray):
        assert hasattr(self, 'gp')
        X = torch.tensor(x, **self.KWARGS)
        post = self.gp.posterior(X)
        with torch.no_grad():
            mean = post.mean.cpu().numpy()
            std = post.variance.sqrt().cpu().numpy()
        return mean, std


class PyFemtetModel:

    current_trans: _SearchSpaceTransform
    current_model: AbstractModel
    history: History

    def update_model(self, model: AbstractModel):
        self.current_model = model

    def fit(self, history: History, df: pd.DataFrame, **kwargs):
        assert hasattr(self, 'current_model')
        assert 'x' not in kwargs
        assert 'y' not in kwargs
        assert 'bounds' not in kwargs

        self.history = history

        # remove nan from df
        df = df.dropna(subset=history.obj_names + history.prm_names, how='any')

        # set current trans
        self.current_trans = get_transform_0_1(df, history)

        # transform all values
        # trans を作るときの search_space に含まれない prm_name はここで無視される
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
