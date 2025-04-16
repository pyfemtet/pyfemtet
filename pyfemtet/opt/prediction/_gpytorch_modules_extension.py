#!/usr/bin/env python3

# This file is derived from the one of botorch.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Pre-packaged kernels for bayesian optimization, including a Scale/Matern
kernel that is well-suited to low-dimensional high-noise problems, and
a dimension-agnostic RBF kernel without outputscale.

References:

.. [Hvarfner2024vanilla]
    C. Hvarfner, E. O. Hellsten, L. Nardi,
    Vanilla Bayesian Optimization Performs Great in High Dimensions.
    In International Conference on Machine Learning, 2024.
"""

from math import log, sqrt
from typing import Optional, Sequence, Union

import torch
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior

MIN_INFERRED_NOISE_LEVEL = 1e-4
SQRT2 = sqrt(2)
SQRT3 = sqrt(3)


# def get_matern_kernel_with_gamma_prior(
#     ard_num_dims: int, batch_shape: Optional[torch.Size] = None
# ) -> ScaleKernel:
#     r"""Constructs the Scale-Matern kernel that is used by default by
#     several models. This uses a Gamma(3.0, 6.0) prior for the lengthscale
#     and a Gamma(2.0, 0.15) prior for the output scale.
#     """
#     return ScaleKernel(
#         base_kernel=MaternKernel(
#             nu=2.5,
#             ard_num_dims=ard_num_dims,
#             batch_shape=batch_shape,
#             lengthscale_prior=GammaPrior(3.0, 6.0),
#         ),
#         batch_shape=batch_shape,
#         outputscale_prior=GammaPrior(2.0, 0.15),
#     )
#
#
# def get_gaussian_likelihood_with_gamma_prior(
#     batch_shape: Optional[torch.Size] = None,
# ) -> GaussianLikelihood:
#     r"""Constructs the GaussianLikelihood that is used by default by
#     several models. This uses a Gamma(1.1, 0.05) prior and constrains the
#     noise level to be greater than MIN_INFERRED_NOISE_LEVEL (=1e-4).
#     """
#     batch_shape = torch.Size() if batch_shape is None else batch_shape
#     noise_prior = GammaPrior(1.1, 0.05)
#     noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
#     return GaussianLikelihood(
#         noise_prior=noise_prior,
#         batch_shape=batch_shape,
#         noise_constraint=GreaterThan(
#             MIN_INFERRED_NOISE_LEVEL,
#             transform=None,
#             initial_value=noise_prior_mode,
#         ),
#     )
#
#
# def get_gaussian_likelihood_with_lognormal_prior(
#     batch_shape: Optional[torch.Size] = None,
# ) -> GaussianLikelihood:
#     """Return Gaussian likelihood with a LogNormal(-4.0, 1.0) prior.
#     This prior is based on [Hvarfner2024vanilla]_.
#
#     Args:
#         batch_shape: Batch shape for the likelihood.
#
#     Returns:
#         GaussianLikelihood with LogNormal(-4.0, 1.0) prior and constrains the
#         noise level to be greater than MIN_INFERRED_NOISE_LEVEL (=1e-4).
#     """
#     batch_shape = torch.Size() if batch_shape is None else batch_shape
#     noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
#     return GaussianLikelihood(
#         noise_prior=noise_prior,
#         batch_shape=batch_shape,
#         noise_constraint=GreaterThan(
#             MIN_INFERRED_NOISE_LEVEL,
#             transform=None,
#             initial_value=noise_prior.mode,
#         ),
#     )


def get_covar_module_with_dim_scaled_prior_extension(
    ard_num_dims: int,
    batch_shape: Optional[torch.Size] = None,
    use_rbf_kernel: bool = True,
    active_dims: Optional[Sequence[int]] = None,
    loc_coef: float = 1.,
    scale_coef: float = 1.,
) -> Union[MaternKernel, RBFKernel]:
    """Returns an RBF or Matern kernel with priors
    from  [Hvarfner2024vanilla]_.

    Args:
        ard_num_dims: Number of feature dimensions for ARD.
        batch_shape: Batch shape for the covariance module.
        use_rbf_kernel: Whether to use an RBF kernel. If False, uses a Matern kernel.
        active_dims: The set of input dimensions to compute the covariances on.
            By default, the covariance is computed using the full input tensor.
            Set this if you'd like to ignore certain dimensions.

        loc_coef (float): The coefficient of loc of LogNormalPrior.
        scale_coef (float): The coefficient of scale of LogNormalPrior.


    Returns:
        A Kernel constructed according to the given arguments. The prior is constrained
        to have lengthscales larger than 0.025 for numerical stability.
    """
    base_class = RBFKernel if use_rbf_kernel else MaternKernel
    lengthscale_prior = LogNormalPrior(loc=(SQRT2 + log(ard_num_dims) * 0.5) * loc_coef, scale=SQRT3 * scale_coef)
    base_kernel = base_class(
        ard_num_dims=ard_num_dims,
        batch_shape=batch_shape,
        lengthscale_prior=lengthscale_prior,
        lengthscale_constraint=GreaterThan(
            2.5e-2, transform=None, initial_value=lengthscale_prior.mode
        ),
        # pyre-ignore[6] GPyTorch type is unnecessarily restrictive.
        active_dims=active_dims,
    )
    return base_kernel
