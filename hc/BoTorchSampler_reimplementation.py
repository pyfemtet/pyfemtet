# ===== botorch の獲得関数に pof を乗ずるための準備 =====

import torch
from torch import Tensor
from torch.distributions import Normal
from botorch.models import SingleTaskGP


# botorch の獲得関数クラスに pof 係数を追加する関数
def acqf_patch_factory(acqf_class):

    # optuna_integration.botorch.botorch.qExpectedImprovement
    class ACQFWithPOF(acqf_class):
        gamma: float = 1.0
        threshold: float = 0.5
        model_c: SingleTaskGP

        def set_model_c(self, model_c: SingleTaskGP):
            self.model_c = model_c

        def pof(self, X: Tensor):
            # 予測点の平均と標準偏差をもとにした正規分布の関数を作る
            _X = X.squeeze(1)
            posterior = self.model_c.posterior(_X)
            mean = posterior.mean
            sigma = posterior.variance.sqrt()

            # 積分する
            normal = Normal(mean, sigma)
            # ここの閾値を true に近づけるほど厳しくなる
            # true の値を超えて大きくしすぎると、多分 true も false も
            cdf = 1. - normal.cdf(torch.tensor(self.threshold, device='cpu').double())
            # 差が出なくなる
            return (cdf.squeeze(1)) ** self.gamma

        def forward(self, X: Tensor) -> Tensor:
            base_acqf = super().forward(X)
            return base_acqf * self.pof(X)

    return ACQFWithPOF


# optimize_acqf の実行時に獲得関数に拘束モデルを渡す関数
def optimize_acqf_patch(original_fun, *args, model_c: SingleTaskGP = None, **kwargs):
    # acqf を検出
    if 'acq_function' in kwargs.keys():
        acqf = kwargs['acq_function']
    else:
        acqf = args[0]

    # acqf に学習済み model_c を渡す
    acqf.set_model_c(model_c)

    # optimize_acqf を実行する
    original_fun(*args, **kwargs)


# ===== optuna_integration で使われている獲得関数 =====
# 獲得関数の編集
import optuna_integration

optuna_integration.botorch.botorch.LogConstrainedExpectedImprovement = acqf_patch_factory()
optuna_integration.botorch.botorch.LogExpectedImprovement = object()
optuna_integration.botorch.botorch.qExpectedImprovement = object()
optuna_integration.botorch.botorch.qNoisyExpectedImprovement = object()
optuna_integration.botorch.botorch.monte_carlo.qExpectedHypervolumeImprovement = object()
optuna_integration.botorch.botorch.ExpectedHypervolumeImprovement = object()
optuna_integration.botorch.botorch.monte_carlo.qNoisyExpectedHypervolumeImprovement = object()
optuna_integration.botorch.botorch.qExpectedImprovement = object()
optuna_integration.botorch.botorch.qKnowledgeGradient = object()
optuna_integration.botorch.botorch.qHypervolumeKnowledgeGradient = object()

# optimize_acqf の編集
from functools import partial

optuna_integration.botorch.botorch.optimize_acqf = partial(
    optimize_acqf_patch,
    optuna_integration.botorch.botorch.optimize_acqf,
)


import optuna
from optuna import Trial
from hc.problem.hyper_sphere import HyperSphere
from hc.problem._base import InfeasibleException


class OptunaHyperSphere(HyperSphere):

    def objective(self, trial: Trial):
        # r
        r = trial.suggest_float('r', *(self.bounds[0]))
        # fai
        fai = []
        for i in range(self.n_prm - 2):
            fai.append(trial.suggest_float(f'fai{i}', *(self.bounds[i+1])))
        # theta
        theta = trial.suggest_float('theta', *(self.bounds[-1]))

        x = list()
        x.append(r)
        x.extend(fai)
        x.append(theta)

        # evaluate hidden constraint
        # is_feasible = self._hidden_constraint(x)
        # if not is_feasible:
            # raise InfeasibleException

        y = self._raw_objective(x)
        return tuple(y)


if __name__ == '__main__':
    dim = 3
    p = OptunaHyperSphere(dim)

    s = optuna.create_study(
        directions=['minimize']*dim,
        sampler=BoTorchSampler(n_startup_trials=dim**2),
        load_if_exists=True,
    )

    s.optimize(p.objective, n_trials=dim**2+5)

    print("set breakpoint here")
