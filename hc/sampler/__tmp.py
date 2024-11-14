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
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

train_y = tensor([
    0,
    0,
    1,
    1,
])



class GPClassificationModel(ApproximateGP):
    def __init__(self, _train_x):
        self.train_inputs = (_train_x,)

        variational_distribution = CholeskyVariationalDistribution(_train_x.size(0)).double()
        variational_strategy = VariationalStrategy(self, _train_x, variational_distribution).double()
        super(GPClassificationModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean().double()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel().double()).double()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


from gpytorch.distributions import MultivariateNormal

model = GPClassificationModel(train_x)
likelihood = gpytorch.likelihoods.BernoulliLikelihood().double()

# ===== Find optimal model hyperparameters =====
# train mode
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood

# !! VariationalELBO はデータ数が少ないと弱いらしい
# num_data refers to the amount of training data
# mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y.numel())
# mll = gpytorch.mlls.VariationalMarginalLogLikelihood(likelihood, model, len(train_y))

# !! RuntimeError: Likelihood must be Gaussian for exact inference
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


class 無理やりMarginalLogLikelihood(gpytorch.mlls.ExactMarginalLogLikelihood, gpytorch.mlls.MarginalLogLikelihood):
    def __init__(self, likelihood, model):
        gpytorch.mlls.MarginalLogLikelihood.__init__(self, likelihood, model)
mll = 無理やりMarginalLogLikelihood(likelihood, model)

# train
training_iter = 100
for i in range(training_iter):
    # Zero back-propped gradients from previous iteration
    optimizer.zero_grad()
    # Get predictive output
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()


# ===== Go into eval mode =====
# eval mode
model.eval()
likelihood.eval()

# calc parameter of Ber. dist. (=prob)
test_x = train_x[0].unsqueeze(0)
ret: MultivariateNormal = model(test_x)
p = likelihood(ret)
print(f'Feasible probability is {p.probs.detach().numpy()[0]:.4f}')

#
# #
#
# # test
# test_x = tensor([[-1, -1]])
# ret: MultivariateNormal = con_model(test_x)
# print(f'Model returns {ret.mean.detach().numpy()[0]:.3f}±{ret.stddev.detach().numpy()[0]:.3f} from {test_x.detach().numpy()[0]}')
#
# test_x = tensor([[1, 1]])
# ret: MultivariateNormal = con_model(test_x)
# print(f'Model returns {ret.mean.detach().numpy()[0]:.3f}±{ret.stddev.detach().numpy()[0]:.3f} from {test_x.detach().numpy()[0]}')
#
# # likelihood
# from torch.distributions import Bernoulli
# _p = Bernoulli(0.3)
# _p.probs
# _p.mean
# _p.stddev
# _p.enumerate_support(True)
#
# test_x = tensor([[-1, -1]])
# ret: MultivariateNormal = con_model(test_x)
# p = con_model.likelihood(ret)
# print(f'Feasible probability is {p.probs.detach().numpy()[0]:.4f}')
#
#
# # posterior
# test_x = tensor([[-1, -1]])
# post = con_model.posterior(test_x)
# print(f'Posterior of model is {post.mean.detach().numpy()[0][0]:.3f}±{post.stddev.detach().numpy()[0]:.3f} from {test_x.detach().numpy()[0]}')
# # ret と一致

