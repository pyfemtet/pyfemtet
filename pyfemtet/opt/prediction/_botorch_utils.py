# import
from __future__ import annotations

from packaging import version

import torch

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel  # , RBFKernel
from gpytorch.priors.torch_priors import GammaPrior  # , LogNormalPrior
# from gpytorch.constraints.constraints import GreaterThan

from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize, Normalize

# import fit_gpytorch_mll
import botorch.version
if version.parse(botorch.version.version) < version.parse("0.8.0"):
    # noinspection PyUnresolvedReferences
    from botorch.fit import fit_gpytorch_model as fit_gpytorch_mll

else:
    from botorch.fit import fit_gpytorch_mll


__all__ = [
    'get_standardizer_and_no_noise_train_yvar',
    'setup_yvar_and_standardizer',
    'setup_gp',
    'get_matern_kernel_with_gamma_prior_as_covar_module',
]


def get_standardizer_and_no_noise_train_yvar(Y: torch.Tensor):
    import gpytorch

    standardizer = Standardize(m=Y.shape[-1])
    min_noise = gpytorch.settings.min_fixed_noise.value(Y.dtype)
    standardizer.forward(Y)  # require to un-transform
    _, YVar = standardizer.untransform(Y, min_noise * torch.ones_like(Y))

    return YVar, standardizer


def setup_yvar_and_standardizer(
        Y_: torch.Tensor,
        observation_noise_: str | float | None,
) -> tuple[torch.Tensor | None, Standardize]:

    standardizer_ = None
    train_yvar_ = None
    if isinstance(observation_noise_, str):
        if observation_noise_.lower() == 'no':
            train_yvar_, standardizer_ = get_standardizer_and_no_noise_train_yvar(Y_)
        else:
            raise NotImplementedError
    elif isinstance(observation_noise_, float):
        train_yvar_ = torch.full_like(Y_, observation_noise_)

    standardizer_ = standardizer_ or Standardize(m=Y_.shape[-1])

    return train_yvar_, standardizer_


def _get_matern_kernel_with_gamma_prior(
        ard_num_dims: int, batch_shape=None
) -> ScaleKernel:
    r"""Constructs the Scale-Matern kernel that is used by default by
    several models. This uses a Gamma(3.0, 6.0) prior for the lengthscale
    and a Gamma(2.0, 0.15) prior for the output scale.
    """

    # PoFBoTorch の要請: 観測のない点は std を大きくしたい

    return ScaleKernel(
        base_kernel=MaternKernel(
            nu=2.5,
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            # lengthscale_prior=GammaPrior(3.0, 6.0),
            lengthscale_prior=GammaPrior(1, 9.0),
        ),
        batch_shape=batch_shape,
        # outputscale_prior=GammaPrior(2.0, 0.15),
        outputscale_prior=GammaPrior(1.0, 0.15),
    )


def get_matern_kernel_with_gamma_prior_as_covar_module(
        X: torch.Tensor,
        Y: torch.Tensor,
        nu: float = 2.5,
        lengthscale_prior: GammaPrior = None,
        outputscale_prior: GammaPrior = None,
):

    _input_batch_shape, _aug_batch_shape = SingleTaskGP.get_batch_dimensions(X, Y)
    ard_num_dims = X.shape[-1]
    batch_shape = _aug_batch_shape

    return ScaleKernel(
        base_kernel=MaternKernel(
            nu=nu,
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            lengthscale_prior=lengthscale_prior or GammaPrior(3.0, 6.0),
        ),
        batch_shape=batch_shape,
        outputscale_prior=outputscale_prior or GammaPrior(2.0, 0.15),
    )


def setup_gp(X, Y, bounds, observation_noise, lh_class=None, covar_module=None):

    lh_class = lh_class or ExactMarginalLogLikelihood

    train_yvar_, standardizer_ = setup_yvar_and_standardizer(
        Y, observation_noise
    )

    model_ = SingleTaskGP(
        X,
        Y,
        train_Yvar=train_yvar_,
        input_transform=Normalize(d=X.shape[-1], bounds=bounds),
        outcome_transform=standardizer_,
        covar_module=covar_module,
    )

    mll_ = lh_class(model_.likelihood, model_)
    fit_gpytorch_mll(mll_)

    return model_
