import numpy as np
import torch
import gpytorch

from botorch.models import SingleTaskGP
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

    def set_bounds_from_history(self, history, df=None):
        from pyfemtet.opt._femopt_core import History
        history: History
        metadata: str

        if df is None:
            df = history.get_df()

        columns = df.columns
        metadata_columns = history.metadata
        target_columns = [
            col for col, metadata in zip(columns, metadata_columns)
            if metadata == 'prm_lb' or metadata == 'prm_ub'
        ]

        bounds_buff = df.iloc[0][target_columns].values  # 2*len(prm_names) array
        bounds = bounds_buff.reshape(-1, 2).astype(float)
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
        self.gp = SingleTaskGP(
            train_X=X,
            train_Y=std_Y,
            train_Yvar=YVar if self.is_noise_free else None,
            input_transform=Normalize(d=X.shape[-1], bounds=self.bounds),
            # BoTorch 0.13 前後で None を渡すと
            # Standardize しない挙動は変わらないので None を渡せばよい
            outcome_transform=None,
        )
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = tensor(x)

        post = self.gp.posterior(X)

        # fit() で Standardize してから SingleTaskGP に渡したので
        # posterior は手動で un-standardize する必要がある
        M, V = self._standardizer.untransform(post.mean, post.variance)

        mean = M.detach().numpy()
        var = V.detach().numpy()
        std = np.sqrt(var)

        return mean, std


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
