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
    y_mean: torch.Tensor  # Y の平均を保存（復元用）
    y_std: torch.Tensor   # Y の標準偏差を保存（復元用）

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

        # Y を外部で標準化（BoTorch 内部の Standardize を使わない）
        self.y_mean = Y.mean(dim=0, keepdim=True)
        self.y_std = Y.std(dim=0, keepdim=True)
        self.y_std[self.y_std == 0] = 1.0  # ゼロ除算防止
        Y_normalized = (Y - self.y_mean) / self.y_std

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

        # 標準化済み Y を渡し、outcome_transform を使用しない
        self.gp = setup_gp(X, Y_normalized, B, observation_noise, likelihood_class, covar_module, use_outcome_transform=False)

    def predict(self, x: np.ndarray):
        assert hasattr(self, 'gp')
        X = torch.tensor(x, **self.KWARGS)
        post = self.gp.posterior(X)
        with torch.no_grad():
            mean = post.mean.cpu().numpy()
            std = post.variance.sqrt().cpu().numpy()

        # 元のスケールに復元
        y_mean_np = self.y_mean.cpu().numpy()
        y_std_np = self.y_std.cpu().numpy()
        mean_restored = mean * y_std_np + y_mean_np
        std_restored = std * y_std_np

        return mean_restored, std_restored


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
