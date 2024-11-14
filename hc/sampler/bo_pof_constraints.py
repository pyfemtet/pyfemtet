import numpy as np
import torch
from torch import Tensor

from hc.sampler._base import AbstractSampler
from hc.problem._base import Floats

# from packaging import version
# import botorch.version
# from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
# from botorch.acquisition.monte_carlo import qExpectedImprovement
# from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective import monte_carlo
# from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
# from botorch.acquisition.multi_objective.objective import FeasibilityWeightedMCMultiOutputObjective
# from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
# from botorch.acquisition.objective import ConstrainedMCObjective
# from botorch.acquisition.objective import GenericMCObjective
# from botorch.models import ModelListGP
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
# from botorch.sampling.list_sampler import ListSampler
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
# from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
# from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
# from botorch.utils.sampling import manual_seed
# from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import normalize
from botorch.utils.transforms import unnormalize
# from botorch.acquisition.analytic import LogConstrainedExpectedImprovement
# from botorch.acquisition.analytic import LogExpectedImprovement
# from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import qHypervolumeKnowledgeGradient
# from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement



def _get_sobol_qmc_normal_sampler(num_samples: int) -> SobolQMCNormalSampler:
    return SobolQMCNormalSampler(torch.Size((num_samples,)))


def tensor(x: Floats) -> Tensor:
    return Tensor(x, device='cpu').double()


threshold = 0.5  # mean(0, 1)
gamma = 1.  # 大きい程 feasibility を重視する


def edited_acqf_factory(BaseACQFClass):
    # from botorch.acquisition import AcquisitionFunction

    class EditedACQF(BaseACQFClass):
        model_c: object

        def set_constraint_model(self, constraint_model):
            self.model_c: SingleTaskGP = constraint_model

        def calc_pof(self, X: Tensor):
            # 予測点の平均と標準偏差をもとにした正規分布の関数を作る
            _X = X.squeeze(1)
            posterior = self.model_c.posterior(_X)
            mean = posterior.mean
            sigma = posterior.variance.sqrt()

            # 積分する
            from torch.distributions import Normal
            normal = Normal(mean, sigma)
            # ここの閾値を true に近づけるほど厳しくなる
            # true の値を超えて大きくしすぎると、多分 true も false も
            # 差が出なくなる
            cdf = 1. - normal.cdf(torch.tensor(threshold, device='cpu').double())
            return (cdf.squeeze(1)) ** gamma

        def forward(self, X: Tensor):
            pof = self.calc_pof(X)
            org = super().forward(X)
            return pof * org

    return EditedACQF


class BayesSampler(AbstractSampler):
    return_wrong_if_infeasible: bool = False
    wrong_rate: float = 1.

    def setup(self):
        # global optimize_acqf

        target_fun = optimize_acqf

        new_fun: callable = OptimizeReplacedACQF(target_fun)
        new_fun.set_constraints(self.constraints)

        optimize_acqf = new_fun

    def candidate_x(self) -> Floats:
        # ===== setup tensor =====
        # common
        bounds = tensor(self.problem.bounds).T

        # for constraint model
        train_x_c = normalize(tensor(self.x), bounds=bounds)
        train_y_c = tensor(self.df['feasibility'].values.astype(float)).unsqueeze(1)  # True: 1, False: 0 => feasible: 1, infeasible: 0

        # for model
        if self.return_wrong_if_infeasible:
            # set infeasible value to y
            idx_infeasible = np.where(self.df['feasibility'] == False)
            wrongest = self.y_feasible.min(axis=0)  # botorch solve optimization as maximization problem
            best = self.y_feasible.max(axis=0)  # botorch solve optimization as maximization problem
            wrong_values = best - (best - wrongest) * self.wrong_rate
            buff = self.y.copy()
            buff[idx_infeasible] = wrong_values

            train_x_z = normalize(tensor(self.x), bounds=bounds)
            train_y_z = tensor(buff)

        else:
            train_x_z = normalize(tensor(self.x_feasible), bounds=bounds)
            train_y_z = tensor(self.y_feasible)

        # ===== fit =====
        # constraint
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

        # model
        model_z = SingleTaskGP(
            train_x_z,  # n_data x n_prm
            train_y_z,  # n_data x n_obj
            train_Yvar=1e-4 + torch.zeros_like(train_y_z),
            outcome_transform=Standardize(
                m=train_y_z.shape[-1],  # The output dimension.
            )
        )
        mll_z = ExactMarginalLogLikelihood(
            model_z.likelihood,
            model_z
        )
        fit_gpytorch_mll(mll_z)

        # set reference points for hypervolume
        ref_point = train_y_z.min(dim=0).values - 1e-8
        ref_point = ref_point.squeeze()
        ref_point_list = ref_point.tolist()

        # Approximate box decomposition similar to Ax when the number of objectives is large.
        # https://github.com/pytorch/botorch/blob/36d09a4297c2a0ff385077b7fcdd5a9d308e40cc/botorch/acquisition/multi_objective/utils.py#L46-L63
        if train_y_z.shape[-1] > 4:
            alpha = 10 ** (-8 + train_y_z.shape[-1])
        else:
            alpha = 0.0
        partitioning = NondominatedPartitioning(
            ref_point=ref_point,
            Y=train_y_z,
            alpha=alpha
        )

        # set acqf
        ACQFClass = edited_acqf_factory(monte_carlo.qExpectedHypervolumeImprovement)
        acqf = ACQFClass(
            model=model_z,
            ref_point=ref_point_list,
            partitioning=partitioning,
            sampler=_get_sobol_qmc_normal_sampler(256),
            # X_pending=pending_x,
            # **additional_qehvi_kwargs,
        )
        standard_bounds = torch.zeros_like(bounds)
        standard_bounds[1] = 1

        # set constraint model to acqf
        acqf.set_constraint_model(model_c)

        # calc candidates
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=standard_bounds,
            q=1,
            num_restarts=20,
            raw_samples=1024,
            options={"batch_limit": 1, "maxiter": 200, "nonnegative": True},
            sequential=True,
        )

        candidates = unnormalize(candidates.detach(), bounds=bounds)

        return candidates.numpy()[0]


from typing import Callable
from functools import partial

import numpy as np

import torch
from torch import Tensor

from optuna.study import Study
from optuna.trial import Trial
from optuna._transform import _SearchSpaceTransform

from botorch.acquisition import AcquisitionFunction
from botorch.optim.initializers import gen_batch_initial_conditions


from time import time



class GeneralFunctionWithForwardDifference(torch.autograd.Function):
    """自作関数を pytorch で自動微分するためのクラスです。

    ユーザー定義関数を botorch 形式に変換する過程で微分の計算ができなくなるのでこれが必要です。
    """

    @staticmethod
    def forward(ctx, f, xs):
        ys = f(xs)
        ctx.save_for_backward(xs, ys)
        ctx.f = f
        return ys

    @staticmethod
    def backward(ctx, grad_output):
        xs, ys = ctx.saved_tensors
        f = ctx.f
        dx = 0.001  # 入力は normalized なので決め打ちでよい
        diff = []
        xs = xs.detach()  # xs に余計な計算履歴を残さないために、detachする。
        for i in range(len(xs)):
            xs[i] += dx
            diff.append(torch.sum(grad_output * (f(xs) - ys)))
            xs[i] -= dx
        diff = torch.tensor(diff) / dx
        return None, diff


class NonlinearInequalityConstraints:
    """botorch の optimize_acqf に parameter constraints を設定するための引数を作成します。"""

    def __init__(self, constraints: list[callable]):
        self._constraints = constraints

        self._nonlinear_inequality_constraints = []

        for cns_func in constraints:
            item = (
                lambda x: GeneralFunctionWithForwardDifference.apply(cns_func, x),
                True
            )
            self._nonlinear_inequality_constraints.append(item)

    def _filter_feasible_conditions(self, ic_batch):
        # List to store feasible initial conditions
        feasible_ic_list = []

        for each_num_restarts in ic_batch:
            feasible_q_list = []
            for each_q in each_num_restarts:
                norm_x: np.ndarray = each_q.numpy()  # normalized parameters

                if is_feasible(self._study, self._constraints, norm_x, self._opt):
                    feasible_q_list.append(each_q)  # Keep only feasible rows

            if feasible_q_list:  # Only add if there are feasible rows
                feasible_ic_list.append(torch.stack(feasible_q_list))

        # Stack feasible conditions back into tensor format
        if feasible_ic_list:
            return torch.stack(feasible_ic_list)
        else:
            return None  # Return None if none are feasible

    @staticmethod
    def _generate_random_initial_conditions(shape):
        # Generates random initial conditions with the same shape as ic_batch
        return torch.rand(shape)

    def _generate_feasible_initial_conditions(self, *args, **kwargs):
        # A `num_restarts x q x d` tensor of initial conditions.
        ic_batch = gen_batch_initial_conditions(*args, **kwargs)
        feasible_ic_batch = self._filter_feasible_conditions(ic_batch)

        while feasible_ic_batch is None:
            # Generate new random ic_batch with the same shape
            print('警告: gen_batch_initial_conditions() は feasible な初期値を提案しませんでした。'
                  'パラメータ提案を探索するための初期値をランダムに選定します。')
            random_ic_batch = self._generate_random_initial_conditions(ic_batch.shape)
            feasible_ic_batch = self._filter_feasible_conditions(random_ic_batch)

        return feasible_ic_batch

    def create_kwargs(self) -> dict:
        """
        nonlinear_inequality_constraints:
            非線形不等式制約を表すタプルのリスト。
            タプルの最初の要素は、`callable(x) >= 0` という形式の制約を表す呼び出し可能オブジェクトです。
            2 番目の要素はブール値で、点内制約の場合は `True`
            制約は後で scipy ソルバーに渡されます。
            この場合、`batch_initial_conditions` を渡す必要があります。
            非線形不等式制約を使用するには、`batch_limit` を 1 に設定する必要もあります。
                これは、`options` で指定されていない場合は自動的に行われます。
        """
        return dict(
            q=1,
            options=dict(
                batch_limit=1,
            ),
            nonlinear_inequality_constraints=self._nonlinear_inequality_constraints,
            ic_generator=self._generate_feasible_initial_conditions,
        )


class OptimizeReplacedACQF(partial):
    """optimize_acqf をこの partial 関数に置き換えます。"""

    # noinspection PyAttributeOutsideInit
    def set_constraints(self, constraints: list[callable]):
        self._constraints: list[callable] = constraints

    # noinspection PyAttributeOutsideInit
    def set_study(self, study: Study):
        self._study: Study = study

    def __call__(self, *args, **kwargs):
        """置き換え先の関数の処理内容です。

        kwargs を横入りして追記することで拘束を実現します。
        """

        # optimize_acqf の探索に parameter constraints を追加します。
        nlic = NonlinearInequalityConstraints(self._constraints)
        kwargs.update(nlic.create_kwargs())

        # replace other arguments
        ...

        return super().__call__(*args, **kwargs)


