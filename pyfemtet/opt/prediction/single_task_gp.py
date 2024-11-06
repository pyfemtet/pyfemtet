import numpy as np
import torch
import gpytorch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from pyfemtet.opt.prediction._base import PredictionModelBase


class _StandardScaler:

    # noinspection PyAttributeOutsideInit
    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        self.m = x.numpy().mean(axis=0)
        self.s = x.numpy().std(axis=0, ddof=1)
        return self.transform(x)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor((x.numpy() - self.m) / self.s).double()

    def inverse_transform_mean(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(x.numpy() * self.s + self.m).double()

    def inverse_transform_var(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(x.numpy() * self.s**2).double()


class _MinMaxScaler:

    # noinspection PyAttributeOutsideInit
    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        self.max = x.numpy().max(axis=0)
        self.min = x.numpy().min(axis=0)
        return self.transform(x)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor((x.numpy() - self.min) / (self.max - self.min)).double()

    def inverse_transform_mean(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(x.numpy() * (self.max - self.min) + self.min).double()

    def inverse_transform_var(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(x.numpy() * (self.max - self.min)**2).double()


class SingleTaskGPModel(PredictionModelBase):
    """Simple interface surrogate model using ```SingleTaskGP```.

    See Also:
        https://botorch.org/api/models.html#botorch.models.gp_regression.SingleTaskGP
    """

    # noinspection PyAttributeOutsideInit
    def fit(self, x: np.ndarray, y: np.ndarray):
        train_x = torch.tensor(x).double()
        train_y = torch.tensor(y).double()

        # check y shape (if single objective problem, output dimension is (n,) )
        self._is_single_objective = len(y[0]) == 1

        # Normalize the input data to the unit cube
        self.scaler_x = _MinMaxScaler()
        train_x = self.scaler_x.fit_transform(train_x)

        # Standardize the output data
        self.scaler_y = _StandardScaler()
        train_y = self.scaler_y.fit_transform(train_y)

        # Fit a Gaussian Process model using the extracted data
        self.gp = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)

    def predict(self, x: np.ndarray) -> list[np.ndarray, np.ndarray]:
        x = torch.tensor(x).double()
        self.gp.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # normalized
            scaled_x = self.scaler_x.transform(x)
            # predict
            pred = self.gp(scaled_x)
            if self._is_single_objective:
                scaled_mean = pred.mean.reshape((-1, 1))
                scaled_var = pred.variance.reshape((-1, 1))
            else:
                scaled_mean = torch.permute(pred.mean, (1, 0))
                scaled_var = torch.permute(pred.variance, (1, 0))
            # unscaling
            mean = self.scaler_y.inverse_transform_mean(scaled_mean).numpy()
            var = self.scaler_y.inverse_transform_var(scaled_var).numpy()
        std = np.sqrt(var)
        return mean, std
