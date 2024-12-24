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
        self.bounds = tensor(np.array(bounds)).T
        self.is_noise_free = is_noise_free

    # noinspection PyAttributeOutsideInit
    def fit(self, x: np.ndarray, y: np.ndarray):
        X = tensor(x)
        Y = tensor(y)

        # Fit a Gaussian Process model using the extracted data
        standardizer = Standardize(m=Y.shape[-1],)
        standardizer.forward(Y)
        _, YVar = standardizer.untransform(Y, torch.full_like(Y, 1e-6))
        self.gp = SingleTaskGP(
            train_X=X,
            train_Y=Y,
            train_Yvar=YVar if self.is_noise_free else None,
            input_transform=Normalize(d=X.shape[-1], bounds=self.bounds),
            outcome_transform=standardizer,
        )
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)

    def predict(self, x: np.ndarray) -> list[np.ndarray, np.ndarray]:
        X = tensor(x)

        self.gp.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # predict
            post = self.gp.posterior(X)
            mean = post.mean.detach().numpy()
            var = post.variance.detach().numpy()
        std = np.sqrt(var)
        return mean, std


if __name__ == '__main__':
    dim = 3
    N = 20
    bounds = (np.arange(dim*2)**2).reshape((-1, 2))
    x = np.random.rand(N, dim)
    x = x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    y = (x ** 2).sum(axis=1, keepdims=True)

    model = SingleTaskGPModel()
    model.fit(x, y)
    print(model.predict(np.array([(b[1] + b[0])/2 for b in bounds])))


