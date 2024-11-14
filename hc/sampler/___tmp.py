import botorch
import gpytorch
# import matplotlib.pyplot as plt
import torch
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.fit import fit_gpytorch_model, fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from functools import partial
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


def tensor(value):
    return torch.Tensor(value, device='cpu').double()


"""
     ^
     |
  feasible
     |
-----+----->
     |
  infeasible
     |

"""


train_x = tensor([
    [-1, -1],
    [ 1, -1],
    [-1,  1],
    [ 1,  1],
])

train_y = tensor([
    0,
    0,
    1,
    1,
])



class GPClassificationModel(ApproximateGP, GPyTorchModel):
    def __init__(self, _train_x, _train_y):
        self.train_inputs = (_train_x,)
        self.train_targets = _train_y

        variational_distribution = CholeskyVariationalDistribution(_train_x.size(0)).double()
        variational_strategy = VariationalStrategy(self, _train_x, variational_distribution).double()
        super(GPClassificationModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean().double()
        self.covar_module = gpytorch.kernels.MaternKernel(nu=1/2).double()
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood().double()
        # self.likelihood = gpytorch.likelihoods.BetaLikelihood().double()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


from gpytorch.distributions import MultivariateNormal

con_model = GPClassificationModel(train_x, train_y)
con_mll = gpytorch.mlls.VariationalELBO(con_model.likelihood, con_model, len(train_x))
# con_mll = gpytorch.mlls.ExactMarginalLogLikelihood(con_model.likelihood, con_model)
fit_gpytorch_mll(con_mll)



# ===== Bernoulli likelihood =====
# test
test_x = tensor([[-1, -1]])
ret: MultivariateNormal = con_model(test_x)
print(f'Model returns {ret.mean.detach().numpy()[0]:.3f}±{ret.stddev.detach().numpy()[0]:.3f} from {test_x.detach().numpy()[0]}')

# likelihood
p = con_model.likelihood(ret)
print(f'Feasible probability is {p.probs.detach().numpy()[0]:.4f}')
