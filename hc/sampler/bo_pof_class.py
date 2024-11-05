# import numpy as np
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

from botorch.models.gpytorch import GPyTorchModel
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


class GPClassificationModel(ApproximateGP, GPyTorchModel):
    def __init__(self, train_x, train_y):
        self.train_inputs = (train_x,)
        self.train_targets = train_y

        variational_distribution = CholeskyVariationalDistribution(train_x.size(0)).double()
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution).double()
        super(GPClassificationModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean().double()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel().double()).double()
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood().double()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



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
            # # 予測点の平均と標準偏差をもとにした正規分布の関数を作る
            # _X = X.squeeze(1)
            # posterior = self.model_c.posterior(_X)
            # mean = posterior.mean
            # sigma = posterior.variance.sqrt()
            #
            # # 積分する
            # from torch.distributions import Normal
            # normal = Normal(mean, sigma)
            # # ここの閾値を true に近づけるほど厳しくなる
            # # true の値を超えて大きくしすぎると、多分 true も false も
            # # 差が出なくなる
            # cdf = 1. - normal.cdf(torch.tensor(threshold, device='cpu').double())

            test_output_distn = self.model_c(X)
            pof = self.model_c.likelihood(test_output_distn).probs

            return (pof.squeeze(1)) ** gamma

        def forward(self, X: Tensor):
            pof = self.calc_pof(X)
            org = super().forward(X)
            return pof * org

    return EditedACQF


class BayesSampler(AbstractSampler):
    return_wrong_if_infeasible: bool = False
    wrong_rate: float = 1.

    def setup(self):
        pass

    def candidate_x(self) -> Floats:
        # ===== setup tensor =====
        # common
        bounds = tensor(self.problem.bounds).T

        # for constraint model
        train_x_c = normalize(tensor(self.x), bounds=bounds)
        train_y_c = tensor(self.df['feasibility'].values.astype(float))  # True: 1, False: 0 => feasible: 1, infeasible: 0

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
        # model_c = SingleTaskGP(
        #     train_x_c,  # n_data x n_prm
        #     train_y_c,  # n_data x n_obj
        #     # train_Yvar=1e-4 + torch.zeros_like(train_y_c),
        #     outcome_transform=Standardize(
        #         m=train_y_c.shape[-1],  # The output dimension.
        #     )
        # )
        # mll_c = ExactMarginalLogLikelihood(
        #     model_c.likelihood,
        #     model_c
        # )
        # fit_gpytorch_mll(mll_c)

        model_c = GPClassificationModel(train_x_c, train_y_c)
        mll_c = gpytorch.mlls.VariationalELBO(model_c.likelihood, model_c, len(train_x_c))

        # _parameters, _bounds = get_parameters_and_bounds(mll)
        # parameters = {n: p for n, p in _parameters.items() if p.requires_grad}
        # closure = get_loss_closure_with_grads(mll, parameters=parameters)
        # closure = NdarrayOptimizationClosure(closure, parameters)

        fit_gpytorch_mll(
            mll_c,
            # closure=closure  # メモリ問題を解決するため
            full_batch_limit=0,
        )

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
            options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
            sequential=True,
        )

        candidates = unnormalize(candidates.detach(), bounds=bounds)

        return candidates.numpy()[0]
