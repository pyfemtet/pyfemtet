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
    from botorch.models.transforms.outcome import Standardize
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


def _validate_botorch_version_for_constrained_opt(func_name: str) -> None:
    if version.parse(botorch.version.version) < version.parse("0.9.0"):
        raise ImportError(
            f"{func_name} requires botorch>=0.9.0 for constrained problems, but got "
            f"botorch={botorch.version.version}.\n"
            "Please run ``pip install botorch --upgrade``."
        )


def _get_constraint_funcs(n_constraints: int) -> list[Callable[["torch.Tensor"], "torch.Tensor"]]:
    return [lambda Z: Z[..., -n_constraints + i] for i in range(n_constraints)]


# helper function
def symlog(x):
    """Symmetric logarithm function.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: The symlog of the input tensor.
    """
    # Apply the symlog transformation
    return torch.where(
        x >= 0,
        torch.log(x + 1),
        -torch.log(1 - x)
    )


# ベースとなる獲得関数クラスに pof 係数を追加したクラスを作成する関数
def acqf_patch_factory(acqf_class, pof_config=None):
    """ベース acqf クラスに pof 係数の計算を追加したクラスを作成します。

    出力されたクラスは、 set_model_c() メソッドで学習済みの
    feasibility を評価するための SingleTaskGP オブジェクトを
    指定する必要があります。
    """
    from torch.distributions import Normal

    if pof_config is None:
        pof_config = PoFConfig()

    # optuna_integration.botorch.botorch.qExpectedImprovement
    class ACQFWithPOF(acqf_class):
        """Introduces PoF coefficients for a given class of acquisition functions."""
        model_c: SingleTaskGP

        enable_pof: bool = pof_config.enable_pof  # PoF を考慮するかどうかを規定します。
        gamma: float or torch.Tensor = pof_config.gamma  # PoF に対する指数です。大きいほど feasibility を重視します。0 だと PoF を考慮しません。
        threshold: float or torch.Tensor = pof_config.threshold  # PoF を cdf で計算する際の境界値です。0 ~ 1 が基本で、 0.5 が推奨です。大きいほど feasibility を重視します。

        enable_log: bool = pof_config.enable_log  # ベース獲得関数値に symlog を適用します。
        enable_positive_only_pof: bool = pof_config.enable_positive_only_pof  # ベース獲得関数が正のときのみ PoF を乗じます。

        enable_dynamic_pof: bool = pof_config.enable_dynamic_pof  # gamma を動的に変更します。 True のとき、gamma は無視されます。
        enable_dynamic_threshold: bool = pof_config.enable_dynamic_threshold  # threshold を動的に変更します。 True のとき、threshold は無視されます。

        enable_repeat_penalty: bool = pof_config.enable_repeat_penalty  # サンプル済みの点の近傍のベース獲得関数値にペナルティ係数を適用します。
        _repeat_penalty: float or torch.Tensor = pof_config._repeat_penalty  # enable_repeat_penalty が True のときに使用される内部変数です。

        enable_dynamic_repeat_penalty: bool = pof_config.enable_dynamic_repeat_penalty  # 同じ値が繰り返された場合にペナルティ係数を強化します。True の場合、enable_repeat_penalty は True として振舞います。
        repeat_watch_window: int = pof_config.repeat_watch_window  # enable_dynamic_repeat_penalty が True のとき、直近いくつの提案値を参照してペナルティの大きさを決めるかを既定します。
        repeat_watch_norm_distance: float = pof_config.repeat_watch_norm_distance  # [0, 1] で正規化されたパラメータ空間においてパラメータの提案同士のノルムがどれくらいの大きさ以下であればペナルティを強くするかを規定します。極端な値は数値不安定性を引き起こす可能性があります。
        _repeat_penalty_gamma: float or torch.Tensor = pof_config._repeat_penalty_gamma  # _repeat_penalty の指数で、内部変数です。


        def set_model_c(self, model_c: SingleTaskGP):
            self.model_c = model_c

        def pof(self, X: torch.Tensor):
            # 予測点の平均と標準偏差をもとにした正規分布の関数を作る
            _X = X.squeeze(1)
            posterior = self.model_c.posterior(_X)
            mean = posterior.mean
            sigma = posterior.variance.sqrt()

            # 積分する
            normal = Normal(mean, sigma)
            # ここの閾値を true に近づけるほど厳しくなる
            # true の値を超えて大きくしすぎると、多分 true も false も
            # 差が出なくなる
            if isinstance(self.threshold, float):
                cdf = 1. - normal.cdf(torch.tensor(self.threshold, device='cpu').double())
            else:
                cdf = 1. - normal.cdf(self.threshold)

            return cdf.squeeze(1)

        def forward(self, X: torch.Tensor) -> torch.Tensor:
            # ===== ベース目的関数 =====
            base_acqf = super().forward(X)

            # ===== 各種 dynamic 手法を使う際の共通処理 =====
            if (
                    self.enable_dynamic_pof
                    or self.enable_dynamic_threshold
                    or self.enable_dynamic_threshold
                    or self.enable_repeat_penalty
                    or self.enable_dynamic_repeat_penalty
            ):
                # ===== 正規化不確実性の計算 =====
                _X = X.squeeze(1)  # batch x 1 x dim -> batch x dim
                # X の予測標準偏差を取得する
                post = self.model_c.posterior(_X)
                current_stddev = post.variance.sqrt()  # batch x dim
                # 既知のポイントの標準偏差を取得する
                post = self.model_c.posterior(self.model_c.train_inputs[0])
                known_stddev = post.variance.sqrt().mean(dim=0)
                # known_stddev: サンプル済みポイントの標準偏差なので小さいはず。
                # current_stddev: 未知の点の標準偏差なので大きいはず。逆に、小さければ既知の点に近い。
                # 既知のポイントの標準偏差で規格化し、平均を取って一次元にする
                buff = current_stddev / known_stddev
                norm_stddev = buff.mean(dim=1)  # (batch, ), 1 ~ 100 くらいの値

                # ===== 動的 gamma =====
                if self.enable_dynamic_pof:
                    buff = 1000. / norm_stddev  # 1 ~ 100 くらいの値
                    buff = symlog(buff)  # 1 ~ 4 くらいの値?
                    self.gamma = buff

                # ===== 動的 threshold =====
                if self.enable_dynamic_threshold:
                    # 効きすぎる傾向？
                    self.threshold = (1 - torch.sigmoid(norm_stddev - 1 - 4) / 2).unsqueeze(1)

                # ===== 繰り返しペナルティ =====
                if self.enable_repeat_penalty:
                    # ベースペナルティは不確実性
                    # stddev が小さい
                    # = サンプル済み付近
                    # = 獲得関数を小さくしたい
                    # = stddev をそのまま係数にする
                    self._repeat_penalty = norm_stddev

                # ===== 動的繰り返しペナルティ =====
                if self.enable_dynamic_repeat_penalty:
                    # 計算コストが多くないので念のためベースペナルティを(再)定義
                    self._repeat_penalty = norm_stddev
                    # サンプル数が watch_window 以下なら何もできない
                    if len(self.model_c.train_inputs[0]) > self.repeat_watch_window:
                        # 直近 N サンプルの x のばらつきが小さいほど
                        # その optimize_scipy 全体でペナルティを強化する
                        monitor_window = self.model_c.train_inputs[0][-self.repeat_watch_window:]
                        g = monitor_window.mean(dim=0)
                        distance = torch.norm(monitor_window - g, dim=1).mean()
                        self._repeat_penalty_gamma = self.repeat_watch_norm_distance / distance

            # ===== PoF 計算 =====
            if self.enable_pof:
                pof = self.pof(X)
            else:
                pof = 1.

            # ===== その他 =====
            if self.enable_log:
                base_acqf = symlog(base_acqf)

            if self.enable_positive_only_pof:
                pof = torch.where(
                    base_acqf >= 0,
                    pof,
                    torch.ones_like(pof)
                )

            ret = -torch.log(1 - torch.sigmoid(base_acqf)) * pof ** self.gamma * self._repeat_penalty ** self._repeat_penalty_gamma
            return ret

    return ACQFWithPOF


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

    train_x = normalize(train_x, bounds=bounds)

    model = SingleTaskGP(
        train_x,
        train_y,
        train_Yvar=1e-4*torch.ones_like(train_y) if _get_use_fixed_noise() else None,
        outcome_transform=Standardize(m=train_y.size(-1))
    )
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
        ACQF = acqf_patch_factory(LogExpectedImprovement, pof_config)
        acqf = ACQF(
            model=model,
            best_f=best_f,
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

    train_x = normalize(train_x, bounds=bounds)
    if pending_x is not None:
        pending_x = normalize(pending_x, bounds=bounds)

    model = SingleTaskGP(
        train_x,
        train_y,
        train_Yvar=1e-4*torch.ones_like(train_y) if _get_use_fixed_noise() else None,
        outcome_transform=Standardize(m=train_y.size(-1))
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    ACQF = acqf_patch_factory(qExpectedImprovement, pof_config)
    acqf = ACQF(
        model=model,
        best_f=best_f,
        sampler=_get_sobol_qmc_normal_sampler(256),
        X_pending=pending_x,
        **additonal_qei_kwargs,
    )
    acqf.set_model_c(model_c)

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

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

    model = SingleTaskGP(
        train_x,
        train_y,
        train_Yvar=1e-4*torch.ones_like(train_y) if _get_use_fixed_noise() else None,
        outcome_transform=Standardize(m=train_y.size(-1))
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

    model = SingleTaskGP(
        train_x,
        train_y,
        train_Yvar=1e-4*torch.ones_like(train_y) if _get_use_fixed_noise() else None,
        outcome_transform=Standardize(m=train_y.size(-1))
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

    model = SingleTaskGP(
        train_x,
        train_y,
        train_Yvar=1e-4*torch.ones_like(train_y) if _get_use_fixed_noise() else None,
        outcome_transform=Standardize(m=train_y.size(-1))
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

    model = SingleTaskGP(
        train_x,
        train_y,
        train_Yvar=1e-4*torch.ones_like(train_y) if _get_use_fixed_noise() else None,
        outcome_transform=Standardize(m=train_y.size(-1))
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

    model = SingleTaskGP(
        train_x,
        train_y,
        train_Yvar=1e-4*torch.ones_like(train_y) if _get_use_fixed_noise() else None,
        outcome_transform=Standardize(m=train_y.size(-1))
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

    model = SingleTaskGP(
        train_x,
        train_y,
        train_Yvar=1e-4*torch.ones_like(train_y) if _get_use_fixed_noise() else None,
        outcome_transform=Standardize(m=train_y.size(-1))
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
            train_Yvar=1e-4*torch.ones_like(train_y[..., [i]]) if _get_use_fixed_noise() else None,
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
    ],
    "torch.Tensor",
]:
    if n_objectives > 3 and not has_constraint and not consider_running_trials:
        return ehvi_candidates_func
    elif n_objectives > 3:
        return qparego_candidates_func
    elif n_objectives > 1:
        return qehvi_candidates_func
    elif consider_running_trials:
        return qei_candidates_func
    else:
        return logei_candidates_func


# ===== main re-implementation of BoTorchSampler =====
@dataclass
class PoFConfig:
    """Configuration of PoFBoTorchSampler

    Args:
        enable_pof (bool):
            Whether to consider Probability of Feasibility.
            Defaults to True.

        gamma (float or torch.Tensor):
            Exponent for Probability of Feasibility. A larger value places more emphasis on feasibility.
            If 0, Probability of Feasibility is not considered.
            Defaults to 1.

        threshold (float or torch.Tensor):
            Boundary value for calculating Probability of Feasibility with CDF.
            Generally between 0 and 1, with 0.5 being recommended. A larger value places more emphasis on feasibility.
            Defaults to 0.5.

        enable_log (bool):
            Whether to apply symlog to the base acquisition function values.
            Defaults to True.

        enable_positive_only_pof (bool):
            Whether to apply Probability of Feasibility only when the base acquisition function is positive.
            Defaults to False.

        enable_dynamic_pof (bool):
            Whether to change gamma dynamically. When True, ```gamma``` argument is ignored.
            Defaults to True.

        enable_dynamic_threshold (bool):
            Whether to change threshold dynamically. When True, ```threshold``` argument is ignored.
            Defaults to False.

        enable_repeat_penalty (bool):
            Whether to apply a penalty coefficient on the base acquisition function values near sampled points.
            Defaults to True.

        enable_dynamic_repeat_penalty (bool):
            Enhances the penalty coefficient if the same value is repeated. When True, it behaves as if enable_repeat_penalty is set to True.
            Defaults to True.

        repeat_watch_window (int):
           Specifies how many recent proposal values are referenced when determining the magnitude of penalties when enable_dynamic_repeat_penalty is True.
           Defaults to 3.

        repeat_watch_norm_distance (float):
           Defines how small the norm distance between proposed parameters needs to be in normalized parameter space [0, 1]
           for a stronger penalty effect. Extreme values may cause numerical instability.
           Defaults to 0.1.

        enable_no_noise (bool):
            Whether to treat observation errors as non-existent
            when training the regression model with the objective
            function value. The default is True because there is
            essentially no observational error in a FEM analysis.
            This is different from the original BoTorchSampler
            implementation.

    """
    enable_pof: bool = True  # PoF を考慮するかどうかを規定します。
    gamma: float or torch.Tensor = 1.0  # PoF に対する指数です。大きいほど feasibility を重視します。0 だと PoF を考慮しません。
    threshold: float or torch.Tensor = 0.5  # PoF を cdf で計算する際の境界値です。0 ~ 1 が基本で、 0.5 が推奨です。大きいほど feasibility を重視します。

    enable_log: bool = True  # ベース獲得関数値に symlog を適用します。
    enable_positive_only_pof: bool = False  # ベース獲得関数が正のときのみ PoF を乗じます。

    enable_dynamic_pof: bool = False  # gamma を動的に変更します。 True のとき、gamma は無視されます。
    enable_dynamic_threshold: bool = False  # threshold を動的に変更します。 True のとき、threshold は無視されます。

    enable_repeat_penalty: bool = False  # サンプル済みの点の近傍のベース獲得関数値にペナルティ係数を適用します。
    _repeat_penalty: float or torch.Tensor = 1.  # enable_repeat_penalty が True のときに使用される内部変数です。

    enable_dynamic_repeat_penalty: bool = False  # 同じ値が繰り返された場合にペナルティ係数を強化します。True の場合、enable_repeat_penalty は True として振舞います。
    repeat_watch_window: int = 3  # enable_dynamic_repeat_penalty が True のとき、直近いくつの提案値を参照してペナルティの大きさを決めるかを既定します。
    repeat_watch_norm_distance: float = 0.1  # [0, 1] で正規化されたパラメータ空間においてパラメータの提案同士のノルムがどれくらいの大きさ以下であればペナルティを強くするかを規定します。極端な値は数値不安定性を引き起こす可能性があります。
    _repeat_penalty_gamma: float or torch.Tensor = 1.  # _repeat_penalty の指数で、内部変数です。

    enable_no_noise: bool = True

    def _disable_all_features(self):
        # 拘束以外のすべてを disable にすることで、
        # BoTorchSampler の実装と同じにします。
        self.enable_pof = False
        self.enable_log = False
        self.enable_positive_only_pof = False
        self.enable_dynamic_pof = False
        self.enable_dynamic_threshold = False
        self.enable_repeat_penalty = False
        self.enable_dynamic_repeat_penalty = False
        self.enable_no_noise = False


@experimental_class("2.4.0")
class PoFBoTorchSampler(BaseSampler):
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

    def __init__(
            self,
            *,
            candidates_func: (
                    Callable[
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
                        ],
                        "torch.Tensor",
                    ]
                    | None
            ) = None,
            constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
            n_startup_trials: int = 10,
            consider_running_trials: bool = False,
            independent_sampler: BaseSampler | None = None,
            seed: int | None = None,
            device: "torch.device" | None = None,
            pof_config: PoFConfig or None = None,
    ):
        _imports.check()

        self._candidates_func = candidates_func
        self._constraints_func = constraints_func
        self._consider_running_trials = consider_running_trials
        self._independent_sampler = independent_sampler or RandomSampler(seed=seed)
        self._n_startup_trials = n_startup_trials
        self._seed = seed

        self._study_id: int | None = None
        self._search_space = IntersectionSearchSpace()
        self._device = device or torch.device("cpu")

        self.pof_config = pof_config or PoFConfig()
        _set_use_fixed_noise(self.pof_config.enable_no_noise)


    @property
    def use_fixed_noise(self) -> bool:
        return _get_use_fixed_noise()

    @use_fixed_noise.setter
    def use_fixed_noise(self, value: bool):
        _set_use_fixed_noise(value)

    def infer_relative_search_space(
            self,
            study: Study,
            trial: FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        if self._study_id is None:
            self._study_id = study._study_id
        if self._study_id != study._study_id:
            # Note that the check below is meaningless when `InMemoryStorage` is used
            # because `InMemoryStorage.create_new_study` always returns the same study ID.
            raise RuntimeError("BoTorchSampler cannot handle multiple studies.")

        search_space: dict[str, BaseDistribution] = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                # built-in `candidates_func` cannot handle distributions that contain just a
                # single value, so we skip them. Note that the parameter values for such
                # distributions are sampled in `Trial`.
                continue
            search_space[name] = distribution

        return search_space

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
        running_trials = [
            t for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)) if t != trial
        ]
        trials = completed_trials + running_trials

        n_trials = len(trials)
        n_completed_trials = len(completed_trials)
        if n_trials < self._n_startup_trials:
            return {}

        trans = _SearchSpaceTransform(search_space)
        n_objectives = len(study.directions)
        values: numpy.ndarray | torch.Tensor = numpy.empty(
            (n_trials, n_objectives), dtype=numpy.float64
        )
        params: numpy.ndarray | torch.Tensor
        con: numpy.ndarray | torch.Tensor | None = None
        bounds: numpy.ndarray | torch.Tensor = trans.bounds
        params = numpy.empty((n_trials, trans.bounds.shape[0]), dtype=numpy.float64)
        for trial_idx, trial in enumerate(trials):
            if trial.state == TrialState.COMPLETE:
                params[trial_idx] = trans.transform(trial.params)
                assert len(study.directions) == len(trial.values)
                for obj_idx, (direction, value) in enumerate(zip(study.directions, trial.values)):
                    assert value is not None
                    if (
                            direction == StudyDirection.MINIMIZE
                    ):  # BoTorch always assumes maximization.
                        value *= -1
                    values[trial_idx, obj_idx] = value
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
            elif trial.state == TrialState.RUNNING:
                if all(p in trial.params for p in search_space):
                    params[trial_idx] = trans.transform(trial.params)
                else:
                    params[trial_idx] = numpy.nan
            else:
                assert False, "trail.state must be TrialState.COMPLETE or TrialState.RUNNING."

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

        values = torch.from_numpy(values).to(self._device)
        params = torch.from_numpy(params).to(self._device)
        if con is not None:
            con = torch.from_numpy(con).to(self._device)
        bounds = torch.from_numpy(bounds).to(self._device)

        if con is not None:
            if con.dim() == 1:
                con.unsqueeze_(-1)
        bounds.transpose_(0, 1)

        if self._candidates_func is None:
            self._candidates_func = _get_default_candidates_func(
                n_objectives=n_objectives,
                has_constraint=con is not None,
                consider_running_trials=self._consider_running_trials,
            )

        completed_values = values[:n_completed_trials]
        completed_params = params[:n_completed_trials]
        if self._consider_running_trials:
            running_params = params[n_completed_trials:]
            running_params = running_params[~torch.isnan(running_params).any(dim=1)]
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

            # ===== model_c 構築 =====
            # ===== model_c を作成する =====
            # ----- bounds, train_x, train_y を準備する -----
            # train_x, train_y は元実装にあわせないと
            # ACQF.forward(X) の引数と一致しなくなる。

            # strict constraint 違反またはモデル破綻で prune された trial
            pruned_trials = study.get_trials(deepcopy=False, states=(TrialState.PRUNED,))
            # 元実装と違い、このモデルを基に次の点を提案するわけではないので running は考えなくてよい
            trials = completed_trials + pruned_trials
            n_trials = len(trials)

            # ----- train_x, train_y (completed_params, completed_values) を作る -----
            # trials から x, y(=feasibility) を収集する
            trans = _SearchSpaceTransform(search_space)
            bounds: numpy.ndarray | torch.Tensor = trans.bounds
            params: numpy.ndarray | torch.Tensor = numpy.empty((n_trials, trans.bounds.shape[0]), dtype=numpy.float64)
            values: numpy.ndarray | torch.Tensor = numpy.empty((n_trials, 1), dtype=numpy.float64)
            for trial_idx, trial in enumerate(trials):
                params[trial_idx] = trans.transform(trial.params)
                if trial.state == TrialState.COMPLETE:
                    values[trial_idx, 0] = 1.  # feasible
                elif trial.state == TrialState.PRUNED:
                    values[trial_idx, 0] = 0.  # infeasible
                else:
                    assert False, "trial.state must be TrialState.COMPLETE or TrialState.PRUNED."
            bounds = torch.from_numpy(bounds).to(self._device)
            params = torch.from_numpy(params).to(self._device)  # 未正規化, n_points x n_parameters Tensor
            values = torch.from_numpy(values).to(self._device)  # 0 or 1, n_points x 1 Tensor
            bounds.transpose_(0, 1)  # 未正規化, 2 x n_parameters Tensor

            # ----- model_c を作る -----
            # with manual_seed(self._seed):
            train_x_c = normalize(params, bounds=bounds)
            train_y_c = values
            model_c = SingleTaskGP(
                train_x_c,  # n_data x n_prm
                train_y_c,  # n_data x n_obj
                # train_Yvar=1e-4 + torch.zeros_like(train_y_c),
                outcome_transform=Standardize(
                    m=train_y_c.shape[-1],  # The output dimension.
                )
            )
            mll_c = ExactMarginalLogLikelihood(
                model_c.likelihood,
                model_c
            )
            fit_gpytorch_mll(mll_c)

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
            )
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

    def sample_independent(
            self,
            study: Study,
            trial: FrozenTrial,
            param_name: str,
            param_distribution: BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()
        if self._seed is not None:
            self._seed = numpy.random.RandomState().randint(numpy.iinfo(numpy.int32).max)

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
            self,
            study: Study,
            trial: FrozenTrial,
            state: TrialState,
            values: Sequence[float] | None,
    ) -> None:
        if self._constraints_func is not None:
            _process_constraints_after_trial(self._constraints_func, study, trial, state)
        self._independent_sampler.after_trial(study, trial, state, values)
