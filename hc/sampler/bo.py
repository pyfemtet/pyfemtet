# import numpy as np
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


class BayesSampler(AbstractSampler):

    def setup(self):
        pass

    def candidate_x(self) -> Floats:
        # setup Tensor
        n_objectives = self.problem.n_obj

        bounds = tensor(self.problem.bounds).T

        train_x = tensor(self.x)
        train_x = normalize(train_x, bounds=bounds)

        train_y = tensor(self.y)

        # fit
        model = SingleTaskGP(
            train_x,  # n_data x n_prm
            train_y,  # n_data x n_obj
            # train_Yvar=
            outcome_transform=Standardize(
                m=train_y.shape[-1],  # The output dimension.
            )
        )
        mll = ExactMarginalLogLikelihood(
            model.likelihood,
            model
        )
        fit_gpytorch_mll(mll)

        # set reference points for hypervolume
        ref_point = train_y.min(dim=0).values - 1e-8
        ref_point = ref_point.squeeze()
        ref_point_list = ref_point.tolist()

        # Approximate box decomposition similar to Ax when the number of objectives is large.
        # https://github.com/pytorch/botorch/blob/36d09a4297c2a0ff385077b7fcdd5a9d308e40cc/botorch/acquisition/multi_objective/utils.py#L46-L63
        if n_objectives > 4:
            alpha = 10 ** (-8 + n_objectives)
        else:
            alpha = 0.0
        partitioning = NondominatedPartitioning(
            ref_point=ref_point,
            Y=train_y,
            alpha=alpha
        )

        # set acqf
        # acqf = qLogExpectedHypervolumeImprovement(
        acqf = monte_carlo.qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point_list,
            partitioning=partitioning,
            sampler=_get_sobol_qmc_normal_sampler(256),
            # X_pending=pending_x,
            # **additional_qehvi_kwargs,
        )
        standard_bounds = torch.zeros_like(bounds)
        standard_bounds[1] = 1

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
