from typing import Any, Callable, Sequence

import optuna

import warnings
from botorch.exceptions.warnings import InputDataWarning
warnings.filterwarnings('ignore', category=InputDataWarning)
from optuna.exceptions import ExperimentalWarning
warnings.filterwarnings('ignore', category=ExperimentalWarning)


# ヘルパー関数
def symlog(x):
    """
    Symmetric logarithm function.

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

# ===== botorch の獲得関数に pof を乗ずるための準備 =====
import torch
from torch import Tensor
from torch.distributions import Normal
from botorch.models import SingleTaskGP


# botorch の獲得関数クラスに pof 係数を追加したクラスを作る関数
def acqf_patch_factory(acqf_class):

    # optuna_integration.botorch.botorch.qExpectedImprovement
    class ACQFWithPOF(acqf_class):
        model_c: SingleTaskGP

        gamma: float = 1.0  # 大きいほど feasibility を重視する。
        # gamma: float = 10.0  # 大きいほど feasibility を重視する。
        threshold: float = 0.5  # 0 ~ 1。大きいほど feasibility を重視する。
        # threshold: float = 0.8  # 0 ~ 1。大きいほど feasibility を重視する。

        enable_log: bool = True  # 獲得関数に symlog を適用する。
        enable_positive_only_pof: bool = True  # 獲得関数が正のときのみ pof を乗ずる。

        # 関数が微分不可能になるので下記の実用化はとりやめ
        enable_dynamic_gamma: bool = False  # gamma を動的に変更します。 True のとき、gamma は無視されます。
        # eps: float = 10.  # dynamic_gamma の場合の s_bar の閾値です。目的関数の予想される最大値の 10% 程度がよいとされます。
        # alpha_0: float = 0.3  # dynamic_gamma の場合の gamma の初期値です。

        def set_model_c(self, model_c: SingleTaskGP):
            self.model_c = model_c

        def s_hat(self, x):
            if hasattr(self, '_mean_and_sigma'):
                mean, sigma = self._mean_and_sigma(x)
            else:
                sigma = self.model.posterior(x).stddev
            return sigma

        def s_bar(self, x):
            # "Bayesian optimization with hidden constraints for aircraft design"
            # if x in A:  # すでに評価されている場合
            if False:
                return 0.
            else:
                return self.s_hat(x)

        def alpha(self, X):
            s_bar = self.s_bar(X)
            x = X.flatten()
            ret = torch.where(
                s_bar < self.eps,
                torch.ones_like(x),
                torch.where(
                    self.pof(X) < 0.01,
                    self.alpha_0 * torch.ones_like(x),
                    torch.zeros_like(x),
                )
            )
            return ret

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
            return cdf.squeeze(1)

        def forward(self, X: Tensor) -> Tensor:
            base_acqf = super().forward(X)
            pof = self.pof(X)

            if self.enable_log:
                base_acqf = symlog(base_acqf)

            if self.enable_positive_only_pof:
                pof = torch.where(
                    base_acqf >= 0,
                    pof,
                    torch.ones_like(pof)
                )

            if self.enable_dynamic_gamma:
                self.gamma = self.alpha(X)

            return base_acqf * pof ** self.gamma


    return ACQFWithPOF


# ===== 獲得関数クラスを挿げ替える関数 =====
# Notes: 代替 optimize_acqf() は引数に model_c を持つので sample_relative の中で毎回挿げ替える
from functools import partial
from packaging.version import Version
import optuna_integration
from optuna_integration import version as optuna_integration_version
if Version(optuna_integration_version.__version__) < Version('4.0.0'):
    target_package = optuna_integration.botorch
else:
    target_package = optuna_integration.botorch.botorch


def replace_acqf():
    from botorch.acquisition.analytic import LogConstrainedExpectedImprovement
    target_package.LogConstrainedExpectedImprovement = acqf_patch_factory(
        LogConstrainedExpectedImprovement
    )

    from botorch.acquisition.analytic import LogExpectedImprovement
    target_package.LogExpectedImprovement = acqf_patch_factory(
        LogExpectedImprovement
    )

    from botorch.acquisition.monte_carlo import qExpectedImprovement
    target_package.qExpectedImprovement = acqf_patch_factory(
        qExpectedImprovement
    )

    from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
    target_package.qNoisyExpectedImprovement = acqf_patch_factory(
        qNoisyExpectedImprovement
    )

    from botorch.acquisition.multi_objective import monte_carlo
    target_package.monte_carlo.qExpectedHypervolumeImprovement = acqf_patch_factory(
        monte_carlo.qExpectedHypervolumeImprovement
    )
    target_package.monte_carlo.qNoisyExpectedHypervolumeImprovement = acqf_patch_factory(
        monte_carlo.qNoisyExpectedHypervolumeImprovement
    )

    from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
    target_package.ExpectedHypervolumeImprovement = acqf_patch_factory(
        ExpectedHypervolumeImprovement
    )

    from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
    target_package.qKnowledgeGradient = acqf_patch_factory(
        qKnowledgeGradient
    )

    from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import qHypervolumeKnowledgeGradient
    target_package.qHypervolumeKnowledgeGradient = acqf_patch_factory(
        qHypervolumeKnowledgeGradient
    )


# ===== optimize_acqf の前に model_c を acqf に set する処理を挿入した関数 =====
from botorch.optim import optimize_acqf


def optimize_acqf_with_model_c_set(
        *args,  # optimize_acqf の位置引数
        model_c: SingleTaskGP = None,  # set する model_c. キーワード引数で与えること。
        **kwargs,  # optimize_acqf のキーワード引数
):
    # acqf を検出
    if 'acq_function' in kwargs.keys():
        # acqf に学習済み model_c を渡す
        acqf = kwargs['acq_function']
        acqf.set_model_c(model_c)
        kwargs['acq_function'] = acqf

    else:
        # acqf に学習済み model_c を渡す
        acqf = args[0]
        acqf.set_model_c(model_c)
        list(args)[0] = acqf
        args = tuple(args)

    # optimize_acqf を実行する
    return optimize_acqf(*args, **kwargs)


# ===== PoF を考慮した BoTorchSampler の実装 =====
import numpy
from optuna._transform import _SearchSpaceTransform
from optuna import Study
from optuna.distributions import BaseDistribution
from optuna.trial import FrozenTrial, TrialState
from optuna_integration import BoTorchSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.utils.sampling import manual_seed


class PoFBoTorchSampler(BoTorchSampler):
    ACQF_REPLACED = False

    def sample_relative(
            self,
            study: Study,
            trial: FrozenTrial,
            search_space: dict[str, BaseDistribution]) -> dict[str, Any]:

        # n_startup_trials までは {} を返す
        if len(search_space) == 0:
            return {}
        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        running_trials = [t for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)) if t != trial]
        n_trials = len(completed_trials + running_trials)
        if n_trials < self._n_startup_trials:
            return {}

        # ===== 必要ならば獲得関数を置換 =====
        if not self.ACQF_REPLACED:
            replace_acqf()
            self.ACQF_REPLACED = True

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

        # model_c を作る
        with manual_seed(self._seed):
            train_x_c = normalize(params, bounds=bounds)
            train_y_c = values
            model_c = SingleTaskGP(
                train_x_c,  # n_data x n_prm
                train_y_c,  # n_data x n_obj
                train_Yvar=1e-4 + torch.zeros_like(train_y_c),
                outcome_transform=Standardize(
                    m=train_y_c.shape[-1],  # The output dimension.
                )
            )
            mll_c = ExactMarginalLogLikelihood(
                model_c.likelihood,
                model_c
            )
            fit_gpytorch_mll(mll_c)

        # ===== optimize_acqf を置換する =====
        new_optimize_acqf = partial(
            optimize_acqf_with_model_c_set,
            model_c=model_c,
        )
        target_package.optimize_acqf = new_optimize_acqf

        # ===== sampling を実行する =====
        return super().sample_relative(study, trial, search_space)


if __name__ == '__main__':
    _count = 0

    def _g_obj(t: optuna.Trial):
        global _count
        print(f'feasible counter: {_count}')
        a = t.suggest_float("a", -10, 10)
        if a < 5:
            print(f'{a=}')
            raise optuna.TrialPruned
        else:
            _count += 1
            return a ** 2

    _g_s = optuna.create_study(
        direction='minimize',
        sampler=PoFBoTorchSampler(
            n_startup_trials=5,
            seed=7,
        )
    )
    _g_s.optimize(_g_obj, n_trials=50)
