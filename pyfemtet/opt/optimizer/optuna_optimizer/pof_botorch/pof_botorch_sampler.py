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

# import
from __future__ import annotations

import warnings
import dataclasses
from packaging import version
from typing import Callable, Sequence, Any, TYPE_CHECKING

# import optuna
from optuna.logging import get_logger
from optuna._imports import try_import
from optuna._transform import _SearchSpaceTransform
from optuna.trial import FrozenTrial, TrialState
from optuna._experimental import experimental_class
from optuna._experimental import experimental_func
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
# from optuna.samplers import RandomSampler
from optuna.samplers._base import _CONSTRAINTS_KEY
# from optuna.samplers._base import _process_constraints_after_trial
# from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study, StudyDirection

from optuna_integration.botorch import BoTorchSampler

# import others
import numpy
import torch
from torch.distributions import Normal

with try_import() as _imports:
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Standardize, Normalize

    # from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
    # from botorch.acquisition.monte_carlo import qExpectedImprovement
    # from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
    # from botorch.acquisition.multi_objective import monte_carlo
    # from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
    # from botorch.acquisition.multi_objective.objective import (
    #     FeasibilityWeightedMCMultiOutputObjective,
    # )
    # from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
    # from botorch.acquisition.objective import ConstrainedMCObjective
    # from botorch.acquisition.objective import GenericMCObjective
    # from botorch.models import ModelListGP
    from botorch.optim import optimize_acqf
    from botorch.sampling import SobolQMCNormalSampler
    # from botorch.sampling.list_sampler import ListSampler
    import botorch.version

    if version.parse(botorch.version.version) < version.parse("0.8.0"):
        # noinspection PyUnresolvedReferences
        from botorch.fit import fit_gpytorch_model as fit_gpytorch_mll

        def _get_sobol_qmc_normal_sampler(num_samples: int) -> SobolQMCNormalSampler:
            # noinspection PyTypeChecker
            return SobolQMCNormalSampler(num_samples)

    else:
        from botorch.fit import fit_gpytorch_mll

        def _get_sobol_qmc_normal_sampler(num_samples: int) -> SobolQMCNormalSampler:
            return SobolQMCNormalSampler(torch.Size((num_samples,)))

    from gpytorch.mlls import ExactMarginalLogLikelihood
    # from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
    #
    # from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
    # from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
    from botorch.utils.sampling import manual_seed
    # from botorch.utils.sampling import sample_simplex
    # from botorch.utils.transforms import normalize
    # from botorch.utils.transforms import unnormalize

with try_import() as _imports_logei:
    from botorch.acquisition.analytic import LogConstrainedExpectedImprovement
    from botorch.acquisition.analytic import LogExpectedImprovement

# with try_import() as _imports_qhvkg:
#     from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
#         qHypervolumeKnowledgeGradient,
#     )

# import pyfemtet
from pyfemtet.opt.history.history import TrialState as PFTrialState
from pyfemtet.opt.exceptions import *
from pyfemtet.opt.optimizer.optuna_optimizer.optuna_attribute import OptunaAttribute
from pyfemtet.opt.optimizer.optuna_optimizer.pof_botorch.enable_nonlinear_constraint import (
    NonlinearInequalityConstraints
)

# warnings to filter
from botorch.exceptions.warnings import InputDataWarning
from optuna.exceptions import ExperimentalWarning

if TYPE_CHECKING:
    from pyfemtet.opt.optimizer import AbstractOptimizer

# noinspection PyTypeChecker
_logger = get_logger(False)

warnings.filterwarnings('ignore', category=InputDataWarning)
warnings.filterwarnings('ignore', category=ExperimentalWarning)

CandidateFunc = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        SingleTaskGP,
        NonlinearInequalityConstraints | None,
        'PoFConfig',
        float | str | None,  # observation noise
        'PartialOptimizeACQFInput',
    ],
    tuple[torch.Tensor, SingleTaskGP],
]


__all__ = [
    'PartialOptimizeACQFConfig',
    'PoFConfig',
    'PoFBoTorchSampler',
]


def log_sigmoid(X: torch.Tensor) -> torch.Tensor:
    return torch.log(1 - torch.sigmoid(-X))


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


def setup_gp(X, Y, bounds, observation_noise, lh_class=None):

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
    )

    mll_ = lh_class(model_.likelihood, model_)
    fit_gpytorch_mll(mll_)

    return model_


class PartialOptimizeACQFConfig:

    def __init__(
            self,
            *,
            gen_candidates: str = None,  # 'scipy' or 'torch'
            timeout_sec: float = None,
            # scipy
            method: str = None,  # 'COBYLA, COBYQA, SLSQP or trust-constr
            tol: float = None,
            scipy_minimize_kwargs: dict = None,
            # torch
    ):

        if gen_candidates:
            if gen_candidates == 'scipy':
                from botorch.generation.gen import gen_candidates_scipy
                gen_candidates = gen_candidates_scipy
            elif gen_candidates == 'torch':
                # from botorch.generation.gen import gen_candidates_torch
                # gen_candidates = gen_candidates_torch
                raise NotImplementedError('`gen_candidates_torch` cannot handle '
                                          'nonlinear inequality constraint.')
            else:
                raise ValueError('`gen_candidates` must be "scipy" or "torch".')

        self.kwargs = dict(
            gen_candidates=gen_candidates,
            options=dict(
                # scipy
                method=method,
                tol=tol,
                options=scipy_minimize_kwargs,  # For method-specific options, see :func:`show_options()`.
                # torch
                #   ExpMAStoppingCriterion
                #   lr
            ),
            timeout_sec=timeout_sec,  # ?
        )


# noinspection PyUnusedLocal,PyIncorrectDocstring
@experimental_func("3.3.0")
def logei_candidates_func(
        train_x: torch.Tensor,
        train_obj: torch.Tensor,
        train_con: torch.Tensor | None,
        bounds: torch.Tensor,
        pending_x: torch.Tensor | None,
        model_c: SingleTaskGP,
        botorch_nlc: NonlinearInequalityConstraints | None,
        pof_config: 'PoFConfig',
        observation_noise: str | float | None,
        partial_optimize_acqf_kwargs: PartialOptimizeACQFConfig,
) -> tuple[torch.Tensor, SingleTaskGP]:
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

    Returns:
        Next set of candidates. Usually the return value of BoTorch's ``optimize_acqf``.

    """

    # ===== ここから変更なし =====

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
    # ===== ここまで変更なし =====

    model = setup_gp(train_x, train_y, bounds, observation_noise)

    if n_constraints > 0:
        ACQF = acqf_patch_factory(
            LogConstrainedExpectedImprovement,
            is_log_acqf=True,
        )
        acqf = ACQF(
            model=model,
            best_f=best_f,
            objective_index=0,
            constraints={i: (None, 0.0) for i in range(1, n_constraints + 1)},
        )
    else:
        ACQF = acqf_patch_factory(
            LogExpectedImprovement,
            is_log_acqf=True,
        )
        acqf = ACQF(
            model=model,
            best_f=best_f,
        )
    acqf.set(model_c, pof_config)

    # optimize_acqf の探索に parameter constraints を追加します。
    if botorch_nlc is not None:

        nlc_kwargs = botorch_nlc.create_kwargs()
        q = nlc_kwargs.pop('q')
        batch_limit = nlc_kwargs.pop('options_batch_limit')

        kwargs = partial_optimize_acqf_kwargs.kwargs.copy()

        options = {
            "batch_limit": batch_limit, "maxiter": 200
        }.update(kwargs.pop('options'))

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=q,
            num_restarts=10,
            raw_samples=512,
            options=options,
            sequential=True,
            **nlc_kwargs,
            **kwargs,
        )

    else:

        kwargs = partial_optimize_acqf_kwargs.kwargs.copy()

        # noinspection PyTypeChecker
        options = {
            "batch_limit": 5, "maxiter": 200
        }.update(kwargs.pop('options'))

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
            options=options,
            sequential=True,
            **kwargs,
        )

    return candidates.detach(), model


def _get_default_candidates_func(
        n_objectives: int,
        has_constraint: bool,
        consider_running_trials: bool,
) -> CandidateFunc:
    if n_objectives > 3 and not has_constraint and not consider_running_trials:
        raise NotImplementedError
        # return ehvi_candidates_func
    elif n_objectives > 3:
        raise NotImplementedError
        # return qparego_candidates_func
    elif n_objectives > 1:
        raise NotImplementedError
        # return qehvi_candidates_func
    elif consider_running_trials:
        raise NotImplementedError
        # return qei_candidates_func
    else:
        return logei_candidates_func


@dataclasses.dataclass
class PoFConfig:
    consider_pof: bool = True
    states_to_consider_pof: list[PFTrialState] = None
    feasibility_threshold: float | str = 0.5  # or 'mean'
    feasibility_noise: float | str | None = None  # 'no' to fixed minimum noise


def acqf_patch_factory(acqf_class, is_log_acqf=False):

    class ACQFWithPoF(acqf_class):

        model_c: SingleTaskGP
        config: PoFConfig

        def set(self, model_c: SingleTaskGP, config: PoFConfig):
            self.model_c = model_c
            self.config = config

        def pof(self, X: torch.Tensor):

            # 予測点の平均と標準偏差をもとにした正規分布を作る
            _X = X.squeeze(1)
            posterior = self.model_c.posterior(_X)
            mean = posterior.mean
            sigma = posterior.variance.sqrt()
            normal = Normal(mean, sigma)

            # threshold を決める
            if isinstance(self.config.feasibility_threshold, float):
                threshold = self.config.feasibility_threshold
            elif isinstance(self.config.feasibility_threshold, str):
                if self.config.feasibility_threshold == 'mean':
                    # train_y: torch.Tensor = self.model_c.train_targets
                    # train_y.mean()
                    threshold = self.model_c.mean_module.constant
                else:
                    raise ValueError
            else:
                raise ValueError

            # 積分する
            if isinstance(threshold, float):
                threshold = torch.tensor(threshold, dtype=X.dtype, device=X.device)
            elif isinstance(threshold, torch.Tensor):
                pass
            else:
                raise ValueError
            cdf = 1. - normal.cdf(threshold)

            return cdf.squeeze(1)

        def forward(self, X: torch.Tensor) -> torch.Tensor:

            # ===== ベース目的関数 =====
            base_acqf: torch.Tensor = super().forward(X)

            # ===== pof =====
            pof = self.pof(X)\
                if self.config.consider_pof \
                else (0. if is_log_acqf else 1.)

            if is_log_acqf:
                return base_acqf + pof

            else:
                return -log_sigmoid(-base_acqf) * pof

    return ACQFWithPoF


@experimental_class("2.4.0")
class PoFBoTorchSampler(BoTorchSampler):

    observation_noise: float | str | None
    pof_config: PoFConfig
    pyfemtet_optimizer: AbstractOptimizer
    partial_optimize_acqf_kwargs: PartialOptimizeACQFConfig
    current_gp_model: SingleTaskGP | None

    _candidates_func: CandidateFunc | None

    def __init__(
            self,
            *,
            candidates_func: CandidateFunc = None,
            constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
            n_startup_trials: int = 10,
            consider_running_trials: bool = False,
            independent_sampler: BaseSampler | None = None,
            seed: int | None = None,
            device: torch.device | None = None,
            # common
            observation_noise: float | str | None = None,  # 'no' to minimum fixed noise
            # acqf_input
            partial_optimize_acqf_kwargs: PartialOptimizeACQFConfig = None,
            # pof_config
            pof_config: PoFConfig = None,
    ):
        super().__init__(candidates_func=candidates_func, constraints_func=constraints_func,
                         n_startup_trials=n_startup_trials, consider_running_trials=consider_running_trials,
                         independent_sampler=independent_sampler, seed=seed, device=device)

        self.partial_optimize_acqf_kwargs = \
            partial_optimize_acqf_kwargs or PartialOptimizeACQFConfig()

        self.observation_noise = observation_noise
        self.pof_config = pof_config or PoFConfig()
        self.current_gp_model = None

    def train_model_c(
            self,
            study,
            search_space,
            feature_dim_bound: list[float] = None,
            feature_param: float | numpy.ndarray = None
    ):

        # ===== 準備 =====
        if feature_dim_bound is not None:
            assert feature_param is not None
        elif feature_param is not None:
            assert feature_dim_bound is not None

        trans = _SearchSpaceTransform(search_space, transform_0_1=False)


        # ===== trials の整理 =====
        # 正常に終了した trial
        completed_trials: list[FrozenTrial] = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

        # hard constraint 違反またはモデル破綻で prune された trial
        pruned_trials: list[FrozenTrial] = study.get_trials(deepcopy=False, states=(TrialState.PRUNED,))

        # 分割
        succeeded_trials: list[FrozenTrial] = []
        hard_v_trials: list[FrozenTrial] = []
        soft_v_trials: list[FrozenTrial] = []
        hidden_v_trials: list[FrozenTrial] = []
        for trial in (completed_trials + pruned_trials):
            state: PFTrialState = OptunaAttribute.get_pf_state_from_trial_attr(
                trial.user_attrs[OptunaAttribute.main_fidelity_key()]
            )
            if state == PFTrialState.succeeded:
                succeeded_trials.append(trial)
            elif state == PFTrialState.hard_constraint_violation:
                hard_v_trials.append(trial)
            elif state == PFTrialState.soft_constraint_violation:
                soft_v_trials.append(trial)
            elif isinstance(
                    PFTrialState.get_corresponding_exception_from_state(state),
                    HiddenConstraintViolation):
                hidden_v_trials.append(trial)
            else:
                warnings.warn(f'Unknown trial state {state} is detected and ignored in {self.__class__.__name__}.')

        # 統合
        Feasibility = int
        trials: list[FrozenTrial] = succeeded_trials
        corresponding_feas: list[Feasibility] = [1 for _ in range(len(succeeded_trials))]
        if self.pof_config.states_to_consider_pof is None:
            trials.extend(hard_v_trials)
            corresponding_feas.extend([0 for _ in range(len(hard_v_trials))])
            trials.extend(soft_v_trials)
            corresponding_feas.extend([0 for _ in range(len(soft_v_trials))])
            trials.extend(hidden_v_trials)
            corresponding_feas.extend([0 for _ in range(len(hidden_v_trials))])
        else:
            if PFTrialState.hard_constraint_violation in self.pof_config.states_to_consider_pof:
                trials.extend(hard_v_trials)
                corresponding_feas.extend([0 for _ in range(len(hard_v_trials))])
            if PFTrialState.soft_constraint_violation in self.pof_config.states_to_consider_pof:
                trials.extend(soft_v_trials)
                corresponding_feas.extend([0 for _ in range(len(soft_v_trials))])
            if any([state_ in self.pof_config.states_to_consider_pof
                    for state_ in PFTrialState.get_hidden_constraint_violation_states()]):
                trials.extend(hidden_v_trials)
                corresponding_feas.extend([0 for _ in range(len(hidden_v_trials))])

        # ===== bounds, params, feasibility の作成 =====
        # bounds: (d(+1), 2) shaped array
        # params: (n, d) shaped array
        # values: (n, m(=1)) shaped array
        bounds: numpy.ndarray = trans.bounds
        if feature_dim_bound is not None:
            bounds: numpy.ndarray = numpy.concatenate([[feature_dim_bound], bounds], axis=0)
        params: numpy.ndarray = numpy.empty((len(trials), bounds.shape[0]), dtype=numpy.float64)
        values: numpy.ndarray = numpy.empty((len(trials), 1), dtype=numpy.float64)

        # 元実装と違い、このモデルを基に次の点を提案する
        # わけではないので running は考えなくてよい
        feasibility: Feasibility
        for trial_idx, (trial, feasibility) in enumerate(zip(trials, corresponding_feas)):

            # train_x
            if feature_param is not None:
                params[trial_idx, 0] = feature_param
                params[trial_idx, 1:] = trans.transform(trial.params)

            else:
                params[trial_idx] = trans.transform(trial.params)

            # train_y
            values[trial_idx, 0] = feasibility


        # ===== Tensor の作成 =====
        # bounds: (2, d(+1)) shaped Tensor
        # train_x: (n, d) shaped Tensor
        # train_y: (n, m(=1)) shaped Tensor
        # train_y_var: (n, m(=1)) shaped Tensor or None
        bounds: torch.Tensor = torch.from_numpy(bounds).to(self._device).transpose(0, 1)
        train_x_c: torch.Tensor = torch.from_numpy(params).to(self._device)
        train_y_c: torch.Tensor = torch.from_numpy(values).to(self._device)
        # yvar
        train_yvar_c, standardizer = setup_yvar_and_standardizer(
            train_y_c, self.observation_noise
        )


        # ===== model_c を作る =====
        # train_x, train_y は元実装にあわせないと
        # ACQF.forward(X) の引数と一致しなくなる。
        with manual_seed(self._seed):
            model_c = SingleTaskGP(
                train_X=train_x_c,
                train_Y=train_y_c,
                train_Yvar=train_yvar_c,
                input_transform=Normalize(d=train_x_c.shape[-1], bounds=bounds),
                outcome_transform=standardizer
            )
            mll_c = ExactMarginalLogLikelihood(
                model_c.likelihood,
                model_c
            )
            fit_gpytorch_mll(mll_c)

            return model_c

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        
        assert hasattr(self, 'pyfemtet_optimizer')
        
        # ===== ここから変更なし =====

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

        # ===== ここまで変更なし =====

        # TODO: ミーゼスなどの場合にこれらのシード固定法も試す
        # if self._seed is not None:
        #     random.seed(self._seed)
        #     numpy.random.seed(self._seed)
        #     torch.manual_seed(self._seed)
        #     torch.backends.cudnn.benchmark = False
        #     torch.backends.cudnn.deterministic = True

        with manual_seed(self._seed):
            # `manual_seed` makes the default candidates functions reproducible.
            # `SobolQMCNormalSampler`'s constructor has a `seed` argument, but its behavior is
            # deterministic when the BoTorch's seed is fixed.

            # feasibility model
            model_c = self.train_model_c(study, search_space)

            # hard constraints
            hard_constraints = [
                cns for cns in self.pyfemtet_optimizer.constraints.values()
                if cns.hard
            ]
            if len(hard_constraints) > 0:
                botorch_nli_cons = NonlinearInequalityConstraints(
                    hard_constraints,
                    self.pyfemtet_optimizer,
                    trans
                )
            else:
                botorch_nli_cons = None

            candidates, model = self._candidates_func(
                completed_params, completed_values, con, bounds, running_params,
                model_c, botorch_nli_cons, self.pof_config, self.observation_noise,
                self.partial_optimize_acqf_kwargs
            )
            if self._seed is not None:
                self._seed += 1

            self.current_gp_model = model


        # ===== ここから変更なし =====

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

        # ===== ここまで変更なし =====

        return trans.untransform(candidates.cpu().numpy())
