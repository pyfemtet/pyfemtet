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


N = 10  # initial observations
problem = botorch.test_functions.multi_objective.BraninCurrin(noise_std=0)


def feasibility_constraint(X):
    return ((X ** 2).sum(axis=-1) < 0.5).to(dtype=torch.float32)


train_x = botorch.utils.sampling.draw_sobol_samples(
    problem.bounds, n=N, q=1, seed=42
).squeeze()

train_y = problem(train_x)
train_y_mean = train_y.mean(dim=0)
train_y_std = train_y.std(dim=0)
train_yn = (train_y - train_y_mean) / train_y_std

train_f = feasibility_constraint(train_x)


class GPClassificationModel(ApproximateGP, GPyTorchModel):
    def __init__(self, train_x, train_y):
        self.train_inputs = (train_x,)
        self.train_targets = train_y

        variational_distribution = CholeskyVariationalDistribution(train_x.size(0)).double()
        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution
        ).double()
        super(GPClassificationModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean().double()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel().double()).double()
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood().double()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


obj1_model = SingleTaskGP(train_x, train_yn[:, [0]])
obj1_mll = ExactMarginalLogLikelihood(obj1_model.likelihood, obj1_model)
fit_gpytorch_model(obj1_mll)

obj2_model = SingleTaskGP(train_x, train_yn[:, [1]])
obj2_mll = ExactMarginalLogLikelihood(obj2_model.likelihood, obj2_model)
fit_gpytorch_model(obj2_mll)

constr_model = GPClassificationModel(train_x.double(), train_f.double())
constr_mll = gpytorch.mlls.VariationalELBO(constr_model.likelihood, constr_model, N)
# fit_gpytorch_model(constr_mll)
fit_gpytorch_mll(constr_mll)

models = ModelListGP(obj1_model, obj2_model, constr_model)

# propose next point
train_yn_min = train_yn.min(dim=0).values
train_yn_max = train_yn.max(dim=0).values
train_data = torch.cat([train_yn, (train_f.view(-1, 1) * 200 - 100)], dim=1)

def objective_func(samples, weights, X):
    Yn, F = samples[..., :-1], samples[..., -1]
    # place outcomes on a scale from 0 (worst) to 1 (best)
    Yu = 1 - (Yn - train_yn_min) / (train_yn_max - train_yn_min)
    scalarization = (weights * Yu).min(dim=-1).values
    p = constr_model.likelihood(F).probs
    return p * scalarization

objective = botorch.acquisition.GenericMCObjective(
    partial(objective_func, weights=torch.tensor([1, 1]))
)
acquisition = qExpectedImprovement(
    model=models, objective=objective, best_f=objective(train_data).max()
)
candidate, acq_value = botorch.optim.optimize_acqf(
    acquisition, bounds=problem.bounds, q=1, num_restarts=5, raw_samples=256
)
print(candidate, acq_value)

# # plot
# n_grid = 51
# dx = torch.linspace(0, 1, n_grid)
# X1_grid, X2_grid = torch.meshgrid(dx, dx)
# X = torch.stack([X1_grid.reshape(-1), X2_grid.reshape(-1)], dim=1)
#
# with torch.no_grad():
#     posterior_mean = models.posterior(X).mean.reshape(n_grid, n_grid, 3)
#
# weights = torch.tensor([1, 1])
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(
#     ncols=4, figsize=(15, 4), sharex=True, sharey=True
# )
# ax1.contourf(X1_grid, X2_grid, posterior_mean[..., 0], levels=16)
# ax2.contourf(X1_grid, X2_grid, posterior_mean[..., 1], levels=16)
# ax3.contourf(X1_grid, X2_grid, constr_model.likelihood(posterior_mean[..., 2]).probs)
# ax4.contourf(X1_grid, X2_grid, objective_func(posterior_mean, weights))
# ax1.scatter(*train_x.T, c="r")
# ax2.scatter(*train_x.T, c="r")
# ax3.scatter(*train_x.T, c="r")
# ax1.set(xlabel="x1", title="output 1 (minimize)", ylabel="x2")
# ax2.set(xlabel="x1", title="output 2 (minimize)")
# ax3.set(xlabel="x1", title="feasibility constraint (maximize)")
# ax4.set(xlabel="x1", title=f"objective, w={weights.numpy()} (maximize)")
# fig.suptitle("Model posterior")
# fig.tight_layout()
# plt.show()
