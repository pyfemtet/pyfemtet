"""This algorithm is based on BoTorchSampler of optuna_integration[1] and the paper[2].


** LICENSE NOTICE OF [1] **

MIT License

Copyright (c) 2018 Preferred Networks, Inc.
Copyright (c) 2024 Kazuma NAITO.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


** reference of [2] **
LEE, H., et al. Optimization subject to hidden constraints via statistical
emulation. Pacific Journal of Optimization, 2011, 7.3: 467-478



"""


from __future__ import annotations

# ===== constant =====
_USE_FIXED_NOISE = True
def _get_use_fixed_noise() -> bool:
    return _USE_FIXED_NOISE
def _set_use_fixed_noise(value: bool):
    global _USE_FIXED_NOISE
    _USE_FIXED_NOISE = value

# ignore warnings
import warnings
from botorch.exceptions.warnings import InputDataWarning
from optuna.exceptions import ExperimentalWarning

warnings.filterwarnings('ignore', category=InputDataWarning)
warnings.filterwarnings('ignore', category=ExperimentalWarning)

from pyfemtet.opt.optimizer._optuna._botorch_patch.enable_nonlinear_constraint import NonlinearInequalityConstraints

from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
import random

from dataclasses import dataclass

import numpy
from optuna import logging
from optuna._experimental import experimental_class
from optuna._experimental import experimental_func
from optuna._imports import try_import
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import RandomSampler
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._base import _process_constraints_after_trial
from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from packaging import version

with try_import() as _imports:
    from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
    from botorch.acquisition.monte_carlo import qExpectedImprovement
    from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
    from botorch.acquisition.multi_objective import monte_carlo
    from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
    from botorch.acquisition.multi_objective.objective import (
        FeasibilityWeightedMCMultiOutputObjective,
    )
    from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
    from botorch.acquisition.objective import ConstrainedMCObjective
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.models import ModelListGP
    from botorch.models import SingleTaskGP
    from botorch.models import MultiTaskGP
    from botorch.models import SingleTaskMultiFidelityGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.models.transforms.input import Normalize
    from botorch.optim import optimize_acqf
    from botorch.sampling import SobolQMCNormalSampler
    from botorch.sampling.list_sampler import ListSampler
    import botorch.version

    if version.parse(botorch.version.version) < version.parse("0.8.0"):
        from botorch.fit import fit_gpytorch_model as fit_gpytorch_mll


        def _get_sobol_qmc_normal_sampler(num_samples: int) -> SobolQMCNormalSampler:
            return SobolQMCNormalSampler(num_samples)

    else:
        from botorch.fit import fit_gpytorch_mll


        def _get_sobol_qmc_normal_sampler(num_samples: int) -> SobolQMCNormalSampler:
            return SobolQMCNormalSampler(torch.Size((num_samples,)))

    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
    import torch

    from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
    from botorch.utils.sampling import manual_seed
    from botorch.utils.sampling import sample_simplex
    from botorch.utils.transforms import normalize
    from botorch.utils.transforms import unnormalize

_logger = logging.get_logger(__name__)

with try_import() as _imports_logei:
    from botorch.acquisition.analytic import LogConstrainedExpectedImprovement
    from botorch.acquisition.analytic import LogExpectedImprovement

with try_import() as _imports_qhvkg:
    from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
        qHypervolumeKnowledgeGradient,
    )

from pyfemtet.opt.optimizer._optuna._pof_botorch import (
    PoFBoTorchSampler,
    PoFConfig,
    _validate_botorch_version_for_constrained_opt,
    _get_constraint_funcs,
    get_minimum_YVar_and_standardizer,
    acqf_patch_factory,
)

from contextlib import nullcontext


# noinspection PyIncorrectDocstring
@experimental_func("3.3.0")
def logei_candidates_func(
        train_x: "torch.Tensor",
        train_obj: "torch.Tensor",
        train_con: "torch.Tensor" | None,
        bounds: "torch.Tensor",
        pending_x: "torch.Tensor" | None,
        model_c: "SingleTaskGP",
        _constraints,
        _study,
        _opt,
        pof_config,
        gp_model_name,
) -> "torch.Tensor":
    """Log Expected Improvement (LogEI).

    The default value of ``candidates_func`` in :class:`~optuna_integration.BoTorchSampler`
    with single-objective optimization.

    Args:
        train_x:
            Previous parameter configurations. A ``torch.Tensor`` of shape
            ``(n_trials, n_params)``. ``n_trials`` is the number of already observed trials
            and ``n_params`` is the number of parameters. ``n_params`` may be larger than the
            actual number of parameters if categorical parameters are included in the search
            space, since these parameters are one-hot encoded.
            Values are not normalized.
        train_obj:
            Previously observed objectives. A ``torch.Tensor`` of shape
            ``(n_trials, n_objectives)``. ``n_trials`` is identical to that of ``train_x``.
            ``n_objectives`` is the number of objectives. Observations are not normalized.
        train_con:
            Objective constraints. A ``torch.Tensor`` of shape ``(n_trials, n_constraints)``.
            ``n_trials`` is identical to that of ``train_x``. ``n_constraints`` is the number of
            constraints. A constraint is violated if strictly larger than 0. If no constraints are
            involved in the optimization, this argument will be :obj:`None`.
        bounds:
            Search space bounds. A ``torch.Tensor`` of shape ``(2, n_params)``. ``n_params`` is
            identical to that of ``train_x``. The first and the second rows correspond to the
            lower and upper bounds for each parameter respectively.
        pending_x:
            Pending parameter configurations. A ``torch.Tensor`` of shape
            ``(n_pending, n_params)``. ``n_pending`` is the number of the trials which are already
            suggested all their parameters but have not completed their evaluation, and
            ``n_params`` is identical to that of ``train_x``.
        model_c:
            Feasibility model.

    Returns:
        Next set of candidates. Usually the return value of BoTorch's ``optimize_acqf``.

    """

    # Validation and process arguments
    with nullcontext():

        # We need botorch >=0.8.1 for LogExpectedImprovement.
        if not _imports_logei.is_successful():
            raise ImportError(
                "logei_candidates_func requires botorch >=0.8.1. "
                "Please upgrade botorch or use qei_candidates_func as candidates_func instead."
            )

        if train_obj.size(-1) != 1:
            raise ValueError("Objective may only contain single values with logEI.")

        n_constraints = train_con.size(1) if train_con is not None else 0
        if n_constraints > 0:
            assert train_con is not None
            train_y = torch.cat([train_obj, train_con], dim=-1)

            is_feas = (train_con <= 0).all(dim=-1)
            train_obj_feas = train_obj[is_feas]

            if train_obj_feas.numel() == 0:
                _logger.warning(
                    "No objective values are feasible. Using 0 as the best objective in logEI."
                )
                best_f = train_obj.min()
            else:
                best_f = train_obj_feas.max()

        else:
            train_y = train_obj
            best_f = train_obj.max()

    # Select GP Model by the fidelity parameter.
    fixed_features, exclude_first_feature, model = get_gp(
        gp_model_name, train_x, train_y, bounds
    )

    # ACQF setup
    with nullcontext():

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        if n_constraints > 0:
            ACQF = acqf_patch_factory(LogConstrainedExpectedImprovement, pof_config)
            acqf = ACQF(
                model=model,
                best_f=best_f,
                objective_index=0,
                constraints={i: (None, 0.0) for i in range(1, n_constraints + 1)},
            )
        else:
            ACQF = acqf_patch_factory(
                LogExpectedImprovement,
                pof_config,
                # exclude_first_feature  # TODO: 第一引数を考慮したほうがよいかどうかは切り替えられるようにする
            )
            acqf = ACQF(
                model=model,
                best_f=best_f,
            )
        acqf.set_model_c(model_c)

    # Add nonlinear_constraint and run optimize_acqf
    with nullcontext():
        # optimize_acqf の探索に parameter constraints を追加します。
        if len(_constraints) > 0:
            nc = NonlinearInequalityConstraints(_study, _constraints, _opt)

            # 1, batch_limit, nonlinear_..., ic_generator
            kwargs = nc.create_kwargs()
            q = kwargs.pop('q')
            batch_limit = kwargs.pop('options')["batch_limit"]

            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=q,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": batch_limit, "maxiter": 200},
                sequential=True,
                fixed_features=fixed_features,
                **kwargs
            )

        else:
            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
                sequential=True,
                fixed_features=fixed_features,
            )

        ret = candidates.detach()

    # Remove fidelity feature if necessary
    if exclude_first_feature:
        ret = ret[:, 1:]

    return ret


# noinspection PyIncorrectDocstring
@experimental_func("2.4.0")
def qei_candidates_func(
        train_x: "torch.Tensor",
        train_obj: "torch.Tensor",
        train_con: "torch.Tensor" | None,
        bounds: "torch.Tensor",
        pending_x: "torch.Tensor" | None,
        model_c: "SingleTaskGP",
        _constraints,
        _study,
        _opt,
        pof_config,
        gp_model_name,
) -> "torch.Tensor":
    """Quasi MC-based batch Expected Improvement (qEI).

    Args:
        train_x:
            Previous parameter configurations. A ``torch.Tensor`` of shape
            ``(n_trials, n_params)``. ``n_trials`` is the number of already observed trials
            and ``n_params`` is the number of parameters. ``n_params`` may be larger than the
            actual number of parameters if categorical parameters are included in the search
            space, since these parameters are one-hot encoded.
            Values are not normalized.
        train_obj:
            Previously observed objectives. A ``torch.Tensor`` of shape
            ``(n_trials, n_objectives)``. ``n_trials`` is identical to that of ``train_x``.
            ``n_objectives`` is the number of objectives. Observations are not normalized.
        train_con:
            Objective constraints. A ``torch.Tensor`` of shape ``(n_trials, n_constraints)``.
            ``n_trials`` is identical to that of ``train_x``. ``n_constraints`` is the number of
            constraints. A constraint is violated if strictly larger than 0. If no constraints are
            involved in the optimization, this argument will be :obj:`None`.
        bounds:
            Search space bounds. A ``torch.Tensor`` of shape ``(2, n_params)``. ``n_params`` is
            identical to that of ``train_x``. The first and the second rows correspond to the
            lower and upper bounds for each parameter respectively.
        pending_x:
            Pending parameter configurations. A ``torch.Tensor`` of shape
            ``(n_pending, n_params)``. ``n_pending`` is the number of the trials which are already
            suggested all their parameters but have not completed their evaluation, and
            ``n_params`` is identical to that of ``train_x``.
        model_c:
            Feasibility model.
    Returns:
        Next set of candidates. Usually the return value of BoTorch's ``optimize_acqf``.

    """

    # Validation and process arguments
    with nullcontext():

        if train_obj.size(-1) != 1:
            raise ValueError("Objective may only contain single values with qEI.")

        if train_con is not None:
            _validate_botorch_version_for_constrained_opt("qei_candidates_func")
            train_y = torch.cat([train_obj, train_con], dim=-1)

            is_feas = (train_con <= 0).all(dim=-1)
            train_obj_feas = train_obj[is_feas]

            if train_obj_feas.numel() == 0:
                # TODO(hvy): Do not use 0 as the best observation.
                _logger.warning(
                    "No objective values are feasible. Using 0 as the best objective in qEI."
                )
                best_f = torch.zeros(())
            else:
                best_f = train_obj_feas.max()

            n_constraints = train_con.size(1)
            additonal_qei_kwargs = {
                "objective": GenericMCObjective(lambda Z, X: Z[..., 0]),
                "constraints": _get_constraint_funcs(n_constraints),
            }
        else:
            train_y = train_obj

            best_f = train_obj.max()

            additonal_qei_kwargs = {}

    # Select GP Model by the fidelity parameter.
    fixed_features, exclude_first_feature, model = get_gp(
        gp_model_name, train_x, train_y, bounds
    )

    # ACQF setup
    with nullcontext():

        ACQF = acqf_patch_factory(
            qExpectedImprovement,
            pof_config,
            # exclude_first_feature,
        )
        acqf = ACQF(
            model=model,
            best_f=best_f,
            sampler=_get_sobol_qmc_normal_sampler(256),
            X_pending=pending_x,
            **additonal_qei_kwargs,
        )
        acqf.set_model_c(model_c)

    # Add nonlinear_constraint and run optimize_acqf
    with nullcontext():

        if len(_constraints) > 0:
            nc = NonlinearInequalityConstraints(_study, _constraints, _opt)

            # 1, batch_limit, nonlinear_..., ic_generator
            kwargs = nc.create_kwargs()
            q = kwargs.pop('q')
            batch_limit = kwargs.pop('options')["batch_limit"]

            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=q,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": batch_limit, "maxiter": 200},
                sequential=True,
                **kwargs
            )

        else:

            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
                sequential=True,
            )

        ret = candidates.detach()

    # Remove fidelity feature if necessary
    if exclude_first_feature:
        ret = ret[:, 1:]

    return ret


# noinspection PyIncorrectDocstring
@experimental_func("3.3.0")
def qnei_candidates_func(
        train_x: "torch.Tensor",
        train_obj: "torch.Tensor",
        train_con: "torch.Tensor" | None,
        bounds: "torch.Tensor",
        pending_x: "torch.Tensor" | None,
        model_c: "SingleTaskGP",
        _constraints,
        _study,
        _opt,
        pof_config,
) -> "torch.Tensor":
    """Quasi MC-based batch Noisy Expected Improvement (qNEI).

    This function may perform better than qEI (`qei_candidates_func`) when
    the evaluated values of objective function are noisy.

    .. seealso::
        :func:`~optuna_integration.botorch.qei_candidates_func` for argument and return value
        descriptions.
    """
    if train_obj.size(-1) != 1:
        raise ValueError("Objective may only contain single values with qNEI.")
    if train_con is not None:
        _validate_botorch_version_for_constrained_opt("qnei_candidates_func")
        train_y = torch.cat([train_obj, train_con], dim=-1)

        n_constraints = train_con.size(1)
        additional_qnei_kwargs = {
            "objective": GenericMCObjective(lambda Z, X: Z[..., 0]),
            "constraints": _get_constraint_funcs(n_constraints),
        }
    else:
        train_y = train_obj

        additional_qnei_kwargs = {}

    train_x = normalize(train_x, bounds=bounds)
    if pending_x is not None:
        pending_x = normalize(pending_x, bounds=bounds)

    train_yvar, standardizer = get_minimum_YVar_and_standardizer(train_y)

    model = SingleTaskGP(
        train_x,
        train_y,
        train_Yvar=train_yvar,
        outcome_transform=standardizer,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    ACQF = acqf_patch_factory(qNoisyExpectedImprovement, pof_config)
    acqf = ACQF(
        model=model,
        X_baseline=train_x,
        sampler=_get_sobol_qmc_normal_sampler(256),
        X_pending=pending_x,
        **additional_qnei_kwargs,
    )
    acqf.set_model_c(model_c)

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    # optimize_acqf の探索に parameter constraints を追加します。
    if len(_constraints) > 0:
        nc = NonlinearInequalityConstraints(_study, _constraints, _opt)

        # 1, batch_limit, nonlinear_..., ic_generator
        kwargs = nc.create_kwargs()
        q = kwargs.pop('q')
        batch_limit = kwargs.pop('options')["batch_limit"]

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=standard_bounds,
            q=q,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": batch_limit, "maxiter": 200},
            sequential=True,
            **kwargs
        )

    else:
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=standard_bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


# noinspection PyIncorrectDocstring
@experimental_func("2.4.0")
def qehvi_candidates_func(
        train_x: "torch.Tensor",
        train_obj: "torch.Tensor",
        train_con: "torch.Tensor" | None,
        bounds: "torch.Tensor",
        pending_x: "torch.Tensor" | None,
        model_c: "SingleTaskGP",
        _constraints,
        _study,
        _opt,
        pof_config,
) -> "torch.Tensor":
    """Quasi MC-based batch Expected Hypervolume Improvement (qEHVI).

    The default value of ``candidates_func`` in :class:`~optuna_integration.BoTorchSampler`
    with multi-objective optimization when the number of objectives is three or less.

    .. seealso::
        :func:`~optuna_integration.botorch.qei_candidates_func` for argument and return value
        descriptions.
    """

    n_objectives = train_obj.size(-1)

    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        is_feas = (train_con <= 0).all(dim=-1)
        train_obj_feas = train_obj[is_feas]

        n_constraints = train_con.size(1)
        additional_qehvi_kwargs = {
            "objective": IdentityMCMultiOutputObjective(outcomes=list(range(n_objectives))),
            "constraints": _get_constraint_funcs(n_constraints),
        }
    else:
        train_y = train_obj

        train_obj_feas = train_obj

        additional_qehvi_kwargs = {}

    train_x = normalize(train_x, bounds=bounds)
    if pending_x is not None:
        pending_x = normalize(pending_x, bounds=bounds)

    train_yvar, standardizer = get_minimum_YVar_and_standardizer(train_y)

    model = SingleTaskGP(
        train_x,
        train_y,
        train_Yvar=train_yvar,
        outcome_transform=standardizer,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Approximate box decomposition similar to Ax when the number of objectives is large.
    # https://github.com/pytorch/botorch/blob/36d09a4297c2a0ff385077b7fcdd5a9d308e40cc/botorch/acquisition/multi_objective/utils.py#L46-L63
    if n_objectives > 4:
        alpha = 10 ** (-8 + n_objectives)
    else:
        alpha = 0.0

    ref_point = train_obj.min(dim=0).values - 1e-8

    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_obj_feas, alpha=alpha)

    ref_point_list = ref_point.tolist()

    ACQF = acqf_patch_factory(monte_carlo.qExpectedHypervolumeImprovement, pof_config)
    acqf = ACQF(
        model=model,
        ref_point=ref_point_list,
        partitioning=partitioning,
        sampler=_get_sobol_qmc_normal_sampler(256),
        X_pending=pending_x,
        **additional_qehvi_kwargs,
    )
    acqf.set_model_c(model_c)

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    # optimize_acqf の探索に parameter constraints を追加します。
    if len(_constraints) > 0:
        nc = NonlinearInequalityConstraints(_study, _constraints, _opt)

        # 1, batch_limit, nonlinear_..., ic_generator
        kwargs = nc.create_kwargs()
        q = kwargs.pop('q')
        batch_limit = kwargs.pop('options')["batch_limit"]

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=standard_bounds,
            q=q,
            num_restarts=20,
            raw_samples=1024,
            options={"batch_limit": batch_limit, "maxiter": 200, "nonnegative": True},
            sequential=True,
            **kwargs
        )

    else:
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=standard_bounds,
            q=1,
            num_restarts=20,
            raw_samples=1024,
            options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
            sequential=True,
        )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


# noinspection PyIncorrectDocstring
@experimental_func("3.5.0")
def ehvi_candidates_func(
        train_x: "torch.Tensor",
        train_obj: "torch.Tensor",
        train_con: "torch.Tensor" | None,
        bounds: "torch.Tensor",
        pending_x: "torch.Tensor" | None,
        model_c: "SingleTaskGP",
        _constraints,
        _study,
        _opt,
        pof_config,
) -> "torch.Tensor":
    """Expected Hypervolume Improvement (EHVI).

    The default value of ``candidates_func`` in :class:`~optuna_integration.BoTorchSampler`
    with multi-objective optimization without constraints.

    .. seealso::
        :func:`~optuna_integration.botorch.qei_candidates_func` for argument and return value
        descriptions.
    """

    n_objectives = train_obj.size(-1)
    if train_con is not None:
        raise ValueError("Constraints are not supported with ehvi_candidates_func.")

    train_y = train_obj
    train_x = normalize(train_x, bounds=bounds)

    train_yvar, standardizer = get_minimum_YVar_and_standardizer(train_y)

    model = SingleTaskGP(
        train_x,
        train_y,
        train_Yvar=train_yvar,
        outcome_transform=standardizer,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Approximate box decomposition similar to Ax when the number of objectives is large.
    # https://github.com/pytorch/botorch/blob/36d09a4297c2a0ff385077b7fcdd5a9d308e40cc/botorch/acquisition/multi_objective/utils.py#L46-L63
    if n_objectives > 4:
        alpha = 10 ** (-8 + n_objectives)
    else:
        alpha = 0.0

    ref_point = train_obj.min(dim=0).values - 1e-8

    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_y, alpha=alpha)

    ref_point_list = ref_point.tolist()

    ACQF = acqf_patch_factory(ExpectedHypervolumeImprovement)
    acqf = ACQF(
        model=model,
        ref_point=ref_point_list,
        partitioning=partitioning,
    )
    acqf.set_model_c(model_c)
    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    # optimize_acqf の探索に parameter constraints を追加します。
    if len(_constraints) > 0:
        nc = NonlinearInequalityConstraints(_study, _constraints, _opt)

        # 1, batch_limit, nonlinear_..., ic_generator
        kwargs = nc.create_kwargs()
        q = kwargs.pop('q')
        batch_limit = kwargs.pop('options')["batch_limit"]

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=standard_bounds,
            q=q,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": batch_limit, "maxiter": 200},
            sequential=True,
            **kwargs
        )

    else:
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=standard_bounds,
            q=1,
            num_restarts=20,
            raw_samples=1024,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


# noinspection PyIncorrectDocstring
@experimental_func("3.1.0")
def qnehvi_candidates_func(
        train_x: "torch.Tensor",
        train_obj: "torch.Tensor",
        train_con: "torch.Tensor" | None,
        bounds: "torch.Tensor",
        pending_x: "torch.Tensor" | None,
        model_c: "SingleTaskGP",
        _constraints,
        _study,
        _opt,
        pof_config,
) -> "torch.Tensor":
    """Quasi MC-based batch Noisy Expected Hypervolume Improvement (qNEHVI).

    According to Botorch/Ax documentation,
    this function may perform better than qEHVI (`qehvi_candidates_func`).
    (cf. https://botorch.org/tutorials/constrained_multi_objective_bo )

    .. seealso::
        :func:`~optuna_integration.botorch.qei_candidates_func` for argument and return value
        descriptions.
    """

    n_objectives = train_obj.size(-1)

    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        n_constraints = train_con.size(1)
        additional_qnehvi_kwargs = {
            "objective": IdentityMCMultiOutputObjective(outcomes=list(range(n_objectives))),
            "constraints": _get_constraint_funcs(n_constraints),
        }
    else:
        train_y = train_obj

        additional_qnehvi_kwargs = {}

    train_x = normalize(train_x, bounds=bounds)
    if pending_x is not None:
        pending_x = normalize(pending_x, bounds=bounds)

    train_yvar, standardizer = get_minimum_YVar_and_standardizer(train_y)

    model = SingleTaskGP(
        train_x,
        train_y,
        train_Yvar=train_yvar,
        outcome_transform=standardizer,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Approximate box decomposition similar to Ax when the number of objectives is large.
    # https://github.com/pytorch/botorch/blob/36d09a4297c2a0ff385077b7fcdd5a9d308e40cc/botorch/acquisition/multi_objective/utils.py#L46-L63
    if n_objectives > 4:
        alpha = 10 ** (-8 + n_objectives)
    else:
        alpha = 0.0

    ref_point = train_obj.min(dim=0).values - 1e-8

    ref_point_list = ref_point.tolist()

    # prune_baseline=True is generally recommended by the documentation of BoTorch.
    # cf. https://botorch.org/api/acquisition.html (accessed on 2022/11/18)
    ACQF = acqf_patch_factory(monte_carlo.qNoisyExpectedHypervolumeImprovement, pof_config)
    acqf = ACQF(
        model=model,
        ref_point=ref_point_list,
        X_baseline=train_x,
        alpha=alpha,
        prune_baseline=True,
        sampler=_get_sobol_qmc_normal_sampler(256),
        X_pending=pending_x,
        **additional_qnehvi_kwargs,
    )
    acqf.set_model_c(model_c)

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    # optimize_acqf の探索に parameter constraints を追加します。
    if len(_constraints) > 0:
        nc = NonlinearInequalityConstraints(_study, _constraints, _opt)

        # 1, batch_limit, nonlinear_..., ic_generator
        kwargs = nc.create_kwargs()
        q = kwargs.pop('q')
        batch_limit = kwargs.pop('options')["batch_limit"]

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=standard_bounds,
            q=q,
            num_restarts=20,
            raw_samples=1024,
            options={"batch_limit": batch_limit, "maxiter": 200, "nonnegative": True},
            sequential=True,
            **kwargs
        )

    else:
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=standard_bounds,
            q=1,
            num_restarts=20,
            raw_samples=1024,
            options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
            sequential=True,
        )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


# noinspection PyIncorrectDocstring
@experimental_func("2.4.0")
def qparego_candidates_func(
        train_x: "torch.Tensor",
        train_obj: "torch.Tensor",
        train_con: "torch.Tensor" | None,
        bounds: "torch.Tensor",
        pending_x: "torch.Tensor" | None,
        model_c: "SingleTaskGP",
        _constraints,
        _study,
        _opt,
        pof_config,
) -> "torch.Tensor":
    """Quasi MC-based extended ParEGO (qParEGO) for constrained multi-objective optimization.

    The default value of ``candidates_func`` in :class:`~optuna_integration.BoTorchSampler`
    with multi-objective optimization when the number of objectives is larger than three.

    .. seealso::
        :func:`~optuna_integration.botorch.qei_candidates_func` for argument and return value
        descriptions.
    """

    n_objectives = train_obj.size(-1)

    weights = sample_simplex(n_objectives).squeeze()
    scalarization = get_chebyshev_scalarization(weights=weights, Y=train_obj)

    if train_con is not None:
        _validate_botorch_version_for_constrained_opt("qparego_candidates_func")
        train_y = torch.cat([train_obj, train_con], dim=-1)
        n_constraints = train_con.size(1)
        objective = GenericMCObjective(lambda Z, X: scalarization(Z[..., :n_objectives]))
        additional_qei_kwargs = {
            "constraints": _get_constraint_funcs(n_constraints),
        }
    else:
        train_y = train_obj

        objective = GenericMCObjective(scalarization)
        additional_qei_kwargs = {}

    train_x = normalize(train_x, bounds=bounds)
    if pending_x is not None:
        pending_x = normalize(pending_x, bounds=bounds)

    train_yvar, standardizer = get_minimum_YVar_and_standardizer(train_y)

    model = SingleTaskGP(
        train_x,
        train_y,
        train_Yvar=train_yvar,
        outcome_transform=standardizer,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    ACQF = acqf_patch_factory(qExpectedImprovement, pof_config)
    acqf = ACQF(
        model=model,
        best_f=objective(train_y).max(),
        sampler=_get_sobol_qmc_normal_sampler(256),
        objective=objective,
        X_pending=pending_x,
        **additional_qei_kwargs,
    )
    acqf.set_model_c(model_c)

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    # optimize_acqf の探索に parameter constraints を追加します。
    if len(_constraints) > 0:
        nc = NonlinearInequalityConstraints(_study, _constraints, _opt)

        # 1, batch_limit, nonlinear_..., ic_generator
        kwargs = nc.create_kwargs()
        q = kwargs.pop('q')
        batch_limit = kwargs.pop('options')["batch_limit"]

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=standard_bounds,
            q=q,
            num_restarts=20,
            raw_samples=1024,
            options={"batch_limit": batch_limit, "maxiter": 200},
            sequential=True,
            **kwargs
        )

    else:
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=standard_bounds,
            q=1,
            num_restarts=20,
            raw_samples=1024,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


@experimental_func("4.0.0")
def qkg_candidates_func(
        train_x: "torch.Tensor",
        train_obj: "torch.Tensor",
        train_con: "torch.Tensor" | None,
        bounds: "torch.Tensor",
        pending_x: "torch.Tensor" | None,
        model_c: "SingleTaskGP",
        _constraints,
        _study,
        _opt,
        pof_config,
) -> "torch.Tensor":
    """Quasi MC-based batch Knowledge Gradient (qKG).

    According to Botorch/Ax documentation,
    this function may perform better than qEI (`qei_candidates_func`).
    (cf. https://botorch.org/tutorials/one_shot_kg )

    .. seealso::
        :func:`~optuna_integration.botorch.qei_candidates_func` for argument and return value
        descriptions.

    """

    if train_obj.size(-1) != 1:
        raise ValueError("Objective may only contain single values with qKG.")
    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)
        n_constraints = train_con.size(1)
        objective = ConstrainedMCObjective(
            objective=lambda Z, X: Z[..., 0],
            constraints=_get_constraint_funcs(n_constraints),
        )
    else:
        train_y = train_obj
        objective = None  # Using the default identity objective.

    train_x = normalize(train_x, bounds=bounds)
    if pending_x is not None:
        pending_x = normalize(pending_x, bounds=bounds)

    train_yvar, standardizer = get_minimum_YVar_and_standardizer(train_y)

    model = SingleTaskGP(
        train_x,
        train_y,
        train_Yvar=train_yvar,
        outcome_transform=standardizer,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    ACQF = acqf_patch_factory(qKnowledgeGradient, pof_config)
    acqf = ACQF(
        model=model,
        num_fantasies=256,
        objective=objective,
        X_pending=pending_x,
    )
    acqf.set_model_c(model_c)

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    # optimize_acqf の探索に parameter constraints を追加します。
    if len(_constraints) > 0:
        nc = NonlinearInequalityConstraints(_study, _constraints, _opt)

        # 1, batch_limit, nonlinear_..., ic_generator
        kwargs = nc.create_kwargs()
        q = kwargs.pop('q')
        batch_limit = kwargs.pop('options')["batch_limit"]

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=standard_bounds,
            q=q,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": batch_limit, "maxiter": 200},
            sequential=True,
            **kwargs
        )

    else:
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=standard_bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 8, "maxiter": 200},
            sequential=True,
        )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


# noinspection PyIncorrectDocstring,SpellCheckingInspection
@experimental_func("4.0.0")
def qhvkg_candidates_func(
        train_x: "torch.Tensor",
        train_obj: "torch.Tensor",
        train_con: "torch.Tensor" | None,
        bounds: "torch.Tensor",
        pending_x: "torch.Tensor" | None,
        model_c: "SingleTaskGP",
        _constraints,
        _study,
        _opt,
        pof_config,
) -> "torch.Tensor":
    """Quasi MC-based batch Hypervolume Knowledge Gradient (qHVKG).

    According to Botorch/Ax documentation,
    this function may perform better than qEHVI (`qehvi_candidates_func`).
    (cf. https://botorch.org/tutorials/decoupled_mobo )

    .. seealso::
        :func:`~optuna_integration.botorch.qei_candidates_func` for argument and return value
        descriptions.
    """

    # We need botorch >=0.9.5 for qHypervolumeKnowledgeGradient.
    if not _imports_qhvkg.is_successful():
        raise ImportError(
            "qhvkg_candidates_func requires botorch >=0.9.5. "
            "Please upgrade botorch or use qehvi_candidates_func as candidates_func instead."
        )

    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)
    else:
        train_y = train_obj

    train_x = normalize(train_x, bounds=bounds)
    if pending_x is not None:
        pending_x = normalize(pending_x, bounds=bounds)

    models = [
        SingleTaskGP(
            train_x,
            train_y[..., [i]],
            train_Yvar=get_minimum_YVar_and_standardizer(train_y[..., [i]])[0],
            outcome_transform=Standardize(m=1)
        )
        for i in range(train_y.shape[-1])
    ]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    n_constraints = train_con.size(1) if train_con is not None else 0
    objective = FeasibilityWeightedMCMultiOutputObjective(
        model,
        X_baseline=train_x,
        constraint_idcs=[-n_constraints + i for i in range(n_constraints)],
    )

    ref_point = train_obj.min(dim=0).values - 1e-8

    ACQF = acqf_patch_factory(qHypervolumeKnowledgeGradient, pof_config)
    acqf = ACQF(
        model=model,
        ref_point=ref_point,
        num_fantasies=16,
        X_pending=pending_x,
        objective=objective,
        sampler=ListSampler(
            *[
                SobolQMCNormalSampler(sample_shape=torch.Size([16]))
                for _ in range(model.num_outputs)
            ]
        ),
        inner_sampler=SobolQMCNormalSampler(sample_shape=torch.Size([32])),
    )
    acqf.set_model_c(model_c)

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    # optimize_acqf の探索に parameter constraints を追加します。
    if len(_constraints) > 0:
        nc = NonlinearInequalityConstraints(_study, _constraints, _opt)

        # 1, batch_limit, nonlinear_..., ic_generator
        kwargs = nc.create_kwargs()
        q = kwargs.pop('q')
        batch_limit = kwargs.pop('options')["batch_limit"]

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=standard_bounds,
            q=q,
            num_restarts=1,
            raw_samples=1024,
            options={"batch_limit": batch_limit, "maxiter": 200, "nonnegative": True},
            sequential=True,
            **kwargs
        )

    else:
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=standard_bounds,
            q=1,
            num_restarts=1,
            raw_samples=1024,
            options={"batch_limit": 4, "maxiter": 200, "nonnegative": True},
            sequential=False,
        )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


def _get_default_candidates_func(
        n_objectives: int,
        has_constraint: bool,
        consider_running_trials: bool,
) -> Callable[
    [
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor" | None,
        "torch.Tensor",
        "torch.Tensor" | None,
        "SingleTaskGP",
        "list[Constraint]",
        "Study",
        "OptunaOptimizer",
        "PoFConfig",
        str,
    ],
    "torch.Tensor",
]:
    if n_objectives > 3 and not has_constraint and not consider_running_trials:
        # return ehvi_candidates_func
        raise NotImplementedError
    elif n_objectives > 3:
        # return qparego_candidates_func
        raise NotImplementedError
    elif n_objectives > 1:
        # return qehvi_candidates_func
        raise NotImplementedError
    elif consider_running_trials:
        # return qei_candidates_func
        raise NotImplementedError
    else:
        return logei_candidates_func


# ===== main re-implementation of BoTorchSampler =====

def get_gp(
        gp_model_name: str,
        train_x,
        train_y,
        bounds,
) -> tuple[dict | None, bool, 'ExactGP']:
    train_yvar, standardizer = get_minimum_YVar_and_standardizer(train_y)

    fixed_features = None
    exclude_first_feature = False
    if gp_model_name == 'SingleTaskMultiFidelityGP':
        fixed_features = {0: 1.}
        exclude_first_feature = True
        model = SingleTaskMultiFidelityGP(
            train_x,
            train_y,
            train_yvar,
            data_fidelities=[0],
            outcome_transform=standardizer,
            input_transform=Normalize(
                train_x.shape[-1],
                indices=list(range(1, train_x.shape[-1])),
                bounds=bounds[:, 1:]
            ),
        )

    elif gp_model_name == 'MultiTaskGP':
        model = MultiTaskGP(
            train_x,
            train_y,
            train_Yvar=train_yvar,
            task_feature=0,
            output_tasks=[1],
            outcome_transform=standardizer,
            input_transform=Normalize(
                train_x.shape[-1],
                indices=list(range(1, train_x.shape[-1])),
                bounds=bounds
            ),
        )

    else:
        raise NotImplementedError

    return fixed_features, exclude_first_feature, model


@experimental_class("2.4.0")
class MultiFidelityPoFBoTorchSampler(PoFBoTorchSampler):
    """A sampler that forked from BoTorchSampler.

    This sampler improves the BoTorchSampler to account
    for known/hidden constraints and repeated penalties.

    See Also:
        https://optuna.readthedocs.io/en/v3.0.0-b1/reference/generated/optuna.integration.BoTorchSampler.html

    Args:
        candidates_func:
            An optional function that suggests the next candidates. It must take the training
            data, the objectives, the constraints, the search space bounds and return the next
            candidates. The arguments are of type ``torch.Tensor``. The return value must be a
            ``torch.Tensor``. However, if ``constraints_func`` is omitted, constraints will be
            :obj:`None`. For any constraints that failed to compute, the tensor will contain
            NaN.

            If omitted, it is determined automatically based on the number of objectives and
            whether a constraint is specified. If the
            number of objectives is one and no constraint is specified, log-Expected Improvement
            is used. If constraints are specified, quasi MC-based batch Expected Improvement
            (qEI) is used.
            If the number of objectives is either two or three, Quasi MC-based
            batch Expected Hypervolume Improvement (qEHVI) is used. Otherwise, for a larger number
            of objectives, analytic Expected Hypervolume Improvement is used if no constraints
            are specified, or the faster Quasi MC-based extended ParEGO (qParEGO) is used if
            constraints are present.

            The function should assume *maximization* of the objective.

            .. seealso::
                See :func:`optuna_integration.botorch.qei_candidates_func` for an example.
        constraints_func:
            An optional function that computes the objective constraints. It must take a
            :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
            be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
            constraint is violated. A value equal to or smaller than 0 is considered feasible.

            If omitted, no constraints will be passed to ``candidates_func`` nor taken into
            account during suggestion.
        n_startup_trials:
            Number of initial trials, that is the number of trials to resort to independent
            sampling.
        consider_running_trials:
            If True, the acquisition function takes into consideration the running parameters
            whose evaluation has not completed. Enabling this option is considered to improve the
            performance of parallel optimization.

            .. note::
                Added in v3.2.0 as an experimental argument.
        independent_sampler:
            An independent sampler to use for the initial trials and for parameters that are
            conditional.
        seed:
            Seed for random number generator.
        device:
            A ``torch.device`` to store input and output data of BoTorch. Please set a CUDA device
            if you fasten sampling.
        pof_config (PoFConfig or None):
            Sampler settings.
    """

    def sample_relative(
            self,
            study: Study,
            trial: FrozenTrial,
            search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        assert isinstance(search_space, dict)

        if len(search_space) == 0:
            return {}

        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        running_trials = [t for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)) if t != trial]
        all_trials = study.get_trials(deepcopy=False)

        trials = completed_trials + running_trials
        n_trials = len(trials)
        if n_trials < self._n_startup_trials:
            return {}

        trans = _SearchSpaceTransform(search_space)
        n_objectives = len(study.directions)

        con: numpy.ndarray | torch.Tensor | None = None
        bounds: numpy.ndarray | torch.Tensor = trans.bounds

        # ===== completed trials =====
        n_completed_trials = len(completed_trials)
        completed_values: numpy.ndarray | torch.Tensor = numpy.empty((n_completed_trials, n_objectives), dtype=numpy.float64)
        completed_params: numpy.ndarray | torch.Tensor
        completed_params = numpy.empty((n_completed_trials, 1 + trans.bounds.shape[0]), dtype=numpy.float64)  # +1 is task index

        # Task index (int starts with 1)
        # or fidelity parameter (main is 1.0)
        completed_params[:, 0] = 1

        for trial_idx, trial in enumerate(completed_trials):
            completed_params[trial_idx, 1:] = trans.transform(trial.params)
            assert len(study.directions) == len(trial.values)

            for obj_idx, (direction, value) in enumerate(zip(study.directions, trial.values)):
                assert value is not None
                if (
                        direction == StudyDirection.MINIMIZE
                ):  # BoTorch always assumes maximization.
                    value *= -1
                completed_values[trial_idx, obj_idx] = value

            if self._constraints_func is not None:
                constraints = study._storage.get_trial_system_attrs(trial._trial_id).get(
                    _CONSTRAINTS_KEY
                )
                if constraints is not None:
                    n_constraints = len(constraints)

                    if con is None:
                        con = numpy.full(
                            (n_completed_trials, n_constraints), numpy.nan, dtype=numpy.float64
                        )
                    elif n_constraints != con.shape[1]:
                        raise RuntimeError(
                            f"Expected {con.shape[1]} constraints "
                            f"but received {n_constraints}."
                        )
                    con[trial_idx] = constraints

        # ===== Add Task information to the completed_params for MultiTaskGP. =====
        # Add fidelity values
        # TODO:
        #   SingleTaskMultiFidelityGP の bounds の扱いが複雑すぎるので根本から分ける
        #   gp_model_name を str ではなく enum にする
        #   SingleTaskMultiFidelityGP が MultiTaskGP より極端に悪いことの原因調査
        #   ★ _pof_botorch module との統合と継承
        #   GUI (fidelity 別の進捗)
        #   History ( history の形で保存するか、又は main history に入れる)
        #       restart 性も考えれば、main history には入れたい。けど閲覧性が悪いか？
        _transformed_x: numpy.ndarray
        _sub_fid_y: dict[int, list[float]]
        _sub_y: list[float]
        fidelity_int_map: dict[str, int] = {}
        gp_model_name: str = None
        for trial_idx, trial in enumerate(all_trials):

            # Get x
            if len(trial.params) != len(trans.bounds):
                # unknown FAIL or RUNNING
                continue
            _transformed_x = trans.transform(trial.params)

            # If sub-fidelity-values are recorded, process them.
            if 'sub-fidelity-values' in trial.user_attrs.keys():
                _sub_fid_y = trial.user_attrs['sub-fidelity-values']

                # For each fidelity values, add them to x and y.
                for fidelity, _sub_y in _sub_fid_y.items():

                    # FrozenTrial の際に user_attr の key は
                    # str にされてしまう模様
                    try:
                        fidelity = float(fidelity)
                    except (TypeError, ValueError):
                        pass

                    # If fidelity is str, re-index as int
                    # for MultiTaskGP.
                    if isinstance(fidelity, str):

                        if gp_model_name is None:
                            gp_model_name = 'MultiTaskGP'
                        else:
                            assert gp_model_name == 'MultiTaskGP', 'Mixed fidelity specification detected.'

                        if fidelity not in fidelity_int_map:
                            fidelity_int_map[fidelity] = len(fidelity_int_map) + 2  # 1 is for main
                        fidelity = fidelity_int_map[fidelity]
                        assert isinstance(fidelity, int)

                    # elif float and 0<= fid < 1,
                    # as SingleTaskMultiFidelityGP.
                    elif isinstance(fidelity, float):

                        if 0 <= fidelity < 1:
                            if gp_model_name is None:
                                gp_model_name = 'SingleTaskMultiFidelityGP'

                            else:
                                assert gp_model_name == 'SingleTaskMultiFidelityGP', 'Mixed fidelity specification detected.'

                        else:
                            raise NotImplementedError('Invalid fidelity specification.')

                    else:
                        raise NotImplementedError('Invalid fidelity specification.')

                    # Add to x.
                    _row = numpy.array([fidelity, *_transformed_x], dtype=numpy.float64)
                    completed_params = numpy.concatenate([completed_params, [_row]], axis=0)

                    # Add to y.
                    for obj_idx, (direction, value) in enumerate(zip(study.directions, _sub_y)):
                        assert value is not None
                        if (
                                direction == StudyDirection.MINIMIZE
                        ):  # BoTorch always assumes maximization.
                            _sub_y[obj_idx] = -1 * value

                    completed_values = numpy.concatenate([completed_values, [_sub_y]], axis=0)

        # No sub-fidelity specification
        if gp_model_name is None:
            gp_model_name = 'MultiTaskGP'

        if gp_model_name == 'SingleTaskMultiFidelityGP':
            bounds = numpy.concatenate(
                [[[0, 1]], bounds], axis=0
            )

        if self._constraints_func is not None:
            if con is None:
                warnings.warn(
                    "`constraints_func` was given but no call to it correctly computed "
                    "constraints. Constraints passed to `candidates_func` will be `None`."
                )
            elif numpy.isnan(con).any():
                warnings.warn(
                    "`constraints_func` was given but some calls to it did not correctly compute "
                    "constraints. Constraints passed to `candidates_func` will contain NaN."
                )

        completed_values = torch.from_numpy(completed_values).to(self._device)
        completed_params = torch.from_numpy(completed_params).to(self._device)

        if con is not None:
            con = torch.from_numpy(con).to(self._device)
            if con.dim() == 1:
                con.unsqueeze_(-1)

        bounds = torch.from_numpy(bounds).to(self._device)
        bounds.transpose_(0, 1)

        if self._candidates_func is None:
            self._candidates_func = _get_default_candidates_func(
                n_objectives=n_objectives,
                has_constraint=con is not None,
                consider_running_trials=self._consider_running_trials,
            )

        if self._consider_running_trials:
            n_running_trials = len(running_trials)
            running_params: numpy.ndarray | torch.Tensor = numpy.empty(
                (n_running_trials, 1 + trans.bounds.shape[0]), dtype=numpy.float64
            )
            running_params[:, 0] = 0

            for trial_idx, trial in enumerate(running_trials):
                running_params[trial_idx, 1:] = trans.transform(trial.params)
                assert len(study.directions) == len(trial.values)

                if all(p in trial.params for p in search_space):
                    running_params[trial_idx] = trans.transform(trial.params)
                else:
                    running_params[trial_idx] = numpy.nan

        else:
            running_params = None

        # 一時的に取り消し：TPESampler と整合性が取れない
        # if self._seed is not None:
        #     random.seed(self._seed)
        #     numpy.random.seed(self._seed)
        #     torch.manual_seed(self._seed)
        #     torch.backends.cudnn.benchmark = False
        #     torch.backends.cudnn.deterministic = True

        with manual_seed(self._seed):

            model_c = self.process_constraint(
                study,
                search_space,
                gp_model_name,
            )

            # ===== NonlinearConstraints の実装に必要なクラスを渡す =====
            # PyFemtet 専用関数が前提になっているからこの実装をせざるを得ない。
            # 将来的に optuna の拘束関数の取り扱いの実装が変わったら
            # そちらに実装を変更する（Constraints の変換をして optuna 単体でも使えるようにする）
            # これらは Optimizer の中でセットする

            # noinspection PyUnresolvedReferences
            _constraints = self._pyfemtet_constraints
            # noinspection PyUnresolvedReferences
            _opt = self._pyfemtet_optimizer

            # `manual_seed` makes the default candidates functions reproducible.
            # `SobolQMCNormalSampler`'s constructor has a `seed` argument, but its behavior is
            # deterministic when the BoTorch's seed is fixed.
            candidates = self._candidates_func(
                completed_params,
                completed_values,
                con,
                bounds,
                running_params,
                model_c,
                _constraints,
                study,
                _opt,
                self.pof_config,
                gp_model_name,
            )
            if gp_model_name == 'SingleTaskMultiFidelityGP':
                bounds = bounds[:, 1:]
            if self._seed is not None:
                self._seed += 1

        if not isinstance(candidates, torch.Tensor):
            raise TypeError("Candidates must be a torch.Tensor.")
        if candidates.dim() == 2:
            if candidates.size(0) != 1:
                raise ValueError(
                    "Candidates batch optimization is not supported and the first dimension must "
                    "have size 1 if candidates is a two-dimensional tensor. Actual: "
                    f"{candidates.size()}."
                )
            # Batch size is one. Get rid of the batch dimension.
            candidates = candidates.squeeze(0)
        if candidates.dim() != 1:
            raise ValueError("Candidates must be one or two-dimensional.")
        if candidates.size(0) != bounds.size(1):
            raise ValueError(
                "Candidates size must match with the given bounds. Actual candidates: "
                f"{candidates.size(0)}, bounds: {bounds.size(1)}."
            )

        return trans.untransform(candidates.cpu().numpy())
