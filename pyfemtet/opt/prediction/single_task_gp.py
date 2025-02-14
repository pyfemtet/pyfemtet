import numpy as np
import torch
import gpytorch

from botorch.models import SingleTaskGP, SingleTaskMultiFidelityGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from pyfemtet.opt.prediction._base import PredictionModelBase


DEVICE = 'cpu'
DTYPE = torch.float64


def tensor(x_):
    return torch.tensor(x_, dtype=DTYPE, device=DEVICE)


class SingleTaskGPModel(PredictionModelBase):
    """Simple interface surrogate model using ```SingleTaskGP```.

    See Also:
        https://botorch.org/api/models.html#botorch.models.gp_regression.SingleTaskGP
    """

    def __init__(self, bounds=None, is_noise_free=True):
        if bounds is not None:
            if isinstance(bounds, np.ndarray):
                self.bounds = tensor(bounds).T
            elif isinstance(bounds, list) or isinstance(bounds, tuple):
                self.bounds = tensor(np.array(bounds)).T
            else:
                raise NotImplementedError('Bounds must be a np.ndarray or list or tuple.')
        else:
            self.bounds = None
        self.is_noise_free = is_noise_free
        self._standardizer: Standardize = None

    def _set_bounds(self, bounds: np.ndarray):
        self.bounds = tensor(bounds).T

    # noinspection PyAttributeOutsideInit
    def fit(self, x: np.ndarray, y: np.ndarray):
        X = tensor(x)
        Y = tensor(y)

        # Standardize を SingleTaskGP に任せると
        # 小さい Variance を勝手に 1e-10 に丸めるので
        # 外で Standardize してから渡す
        standardizer = Standardize(m=Y.shape[-1],)
        std_Y, _ = standardizer.forward(Y)
        YVar = torch.full_like(Y, 1e-6)
        self._standardizer = standardizer

        # Fit a Gaussian Process model using the extracted data
        self.gp = self.get_gp()(
            train_X=X,  # Must pass as a kwargs
            train_Y=std_Y,
            train_Yvar=YVar if self.is_noise_free else None,
            input_transform=Normalize(d=X.shape[-1], bounds=self.bounds),
            # BoTorch 0.13 前後で None を渡すと
            # Standardize しない挙動は変わらないので None を渡せばよい
            outcome_transform=None,
        )
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)

    def get_gp(self):
        return SingleTaskGP

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert len(x.shape) >= 2

        X = tensor(x)

        post = self.gp.posterior(X)

        # fit() で Standardize してから SingleTaskGP に渡したので
        # posterior は手動で un-standardize する必要がある
        M, V = self._standardizer.untransform(post.mean, post.variance)

        mean = M.detach().numpy()
        var = V.detach().numpy()
        std = np.sqrt(var)

        return mean, std


class SingleTaskMultiFidelityGPModel(SingleTaskGPModel):

    def _set_bounds(self, bounds: np.ndarray):
        bounds = np.concatenate([[[0, 1]], bounds], axis=0)
        SingleTaskGPModel._set_bounds(self, bounds)

    def get_gp(self):

        def _get_gp(**kwargs):

            assert 'train_X' in kwargs
            train_x = kwargs.pop('train_X')

            kwargs['train_X'] = train_x

            if 'input_transform' in kwargs:
                kwargs.pop('input_transform')

                normalizer = Normalize(
                    d=train_x.shape[-1],
                    indices=list(range(1, train_x.shape[-1])),
                    bounds=self.bounds,
                )

                kwargs['input_transform'] = normalizer

            gp = SingleTaskMultiFidelityGP(
                data_fidelities=[0],
                **kwargs,
            )
            return gp

        return _get_gp

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate(
            [
                np.ones((len(x), 1)),
                x
            ],
            axis=1
        )
        return super().predict(x)


if __name__ == '__main__':
    dim = 3
    N = 20
    bounds = (np.arange(dim*2)**2).reshape((-1, 2))
    x = np.random.rand(N, dim)
    x = x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    y = (x ** 2).sum(axis=1, keepdims=True) * 1e-7

    model = SingleTaskGPModel()
    model.fit(x, y)
    print(model.predict(np.array([[(b[1] + b[0])/2 for b in bounds]])))

    # 外挿
    print(model.predict(np.array([[b[1] for b in bounds]])))
    print(model.predict(np.array([[b[1] * 2 for b in bounds]])))
