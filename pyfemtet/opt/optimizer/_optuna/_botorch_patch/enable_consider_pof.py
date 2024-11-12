from functools import partial
import torch
from torch import Tensor
from torch.distributions import Normal
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf

from pyfemtet.opt.optimizer._optuna._botorch_patch import detect_target

# ignore warnings
import warnings
from botorch.exceptions.warnings import InputDataWarning
warnings.filterwarnings('ignore', category=InputDataWarning)
from optuna.exceptions import ExperimentalWarning
warnings.filterwarnings('ignore', category=ExperimentalWarning)


__all__ = ['PoFBoTorchSampler']


# helper function
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


# ===== optuna_integration.botorch の獲得関数に pof 係数を導入する =====
# ベースとなる獲得関数クラスに pof 係数を追加したクラスを作成する関数
def acqf_patch_factory(acqf_class):
    """ベース acqf クラスに pof 係数の計算を追加したクラスを作成します。

    出力されたクラスは、 set_model_c() メソッドで学習済みの
    feasibility を評価するための SingleTaskGP オブジェクトを
    指定する必要があります。
    """

    # optuna_integration.botorch.botorch.qExpectedImprovement
    class ACQFWithPOF(acqf_class):
        model_c: SingleTaskGP

        gamma: float = 1.0  # 大きいほど feasibility を重視します。
        threshold: float = 0.5  # 0 ~ 1。大きいほど feasibility を重視します。

        enable_log: bool = False  # 獲得関数に symlog を適用します。
        enable_positive_only_pof: bool = False  # 獲得関数が正のときのみ pof を乗じます。

        # 関数が微分不可能になるので下記の実用化はペンディングする。
        # enable_dynamic_gamma は必ず False であること。
        enable_dynamic_gamma: bool = True  # gamma を動的に変更します。 True のとき、gamma は無視されます。
        # eps: float = 10.  # dynamic_gamma の場合の s_bar の閾値です。目的関数の予想される最大値の 10% 程度がよいとされます。
        # alpha_0: float = 0.3  # dynamic_gamma の場合の gamma の初期値です。
        repeat_penalty: float = 1.

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
            # 差が出なくなる
            if isinstance(self.threshold, float):
                cdf = 1. - normal.cdf(torch.tensor(self.threshold, device='cpu').double())
            else:
                cdf = 1. - normal.cdf(self.threshold)

            return cdf.squeeze(1)

        def forward(self, X: Tensor) -> Tensor:
            base_acqf = super().forward(X)

            if self.enable_dynamic_gamma:

                # 戦略 1. stddev が小さくなっていれば acqf を小さくする
                # バッチを外す
                _X = X.squeeze()  # batch x 1 x dim -> batch x dim
                # X の予測標準偏差を取得する
                post = self.model_c.posterior(_X)
                current_stddev = post.variance.sqrt()  # batch x dim
                # 既知のポイントの標準偏差で規格化し、一次元にする
                post = self.model_c.posterior(self.model_c.train_inputs[0])
                known_stddev = post.variance.sqrt().mean(dim=0)
                # 規格化し、一次元にする
                # known_stddev: 小さい。
                # current_stddev: かなり大きい。10~100 倍とか。
                norm_stddev = current_stddev / known_stddev
                norm_stddev = norm_stddev.mean(dim=1)  # (batch, ), 1 ~ 100 くらいの値

                strategy = 7
                # 1. stddev が小さければ、acqf を小さくする
                if strategy == 0:
                    self.repeat_penalty = norm_stddev

                # 2. stddev が小さければ、gamma を大きくする（=acqf を小さくする
                elif strategy == 1:
                    buff = 1000. / norm_stddev  # 1 ~ 100 くらいの値
                    buff = symlog(buff)  # 1 ~ 4 くらいの値?
                    self.gamma = buff

                # 3. 両方
                elif strategy == 2:
                    self.repeat_penalty = norm_stddev
                    buff = 1000. / norm_stddev  # 1 ~ 100 くらいの値
                    buff = symlog(buff)  # 1 ~ 4 くらいの値?
                    self.gamma = buff

                # 4. stddev が小さければ、threshold を 1 に近づける
                elif strategy == 3:
                    # norm_stddev が最小で 1 の時、threshold が 1
                    # norm_stddev が 1 以上の時、threshold は 0.5 に近づく
                    self.threshold = (1 - torch.sigmoid(norm_stddev - 1 - 4) / 2).unsqueeze(1)

                # 5. stddev が閾値を下回れば gamma を 10 にし、そうでなければ 0 にする
                elif strategy == 4:
                    self.gamma = torch.where(
                        norm_stddev < 20,
                        torch.ones_like(base_acqf)*10,
                        torch.zeros_like(base_acqf),
                    )

                # 6. 5 以外全部
                elif strategy == 6:
                    self.repeat_penalty = norm_stddev
                    buff = 1000. / norm_stddev  # 1 ~ 100 くらいの値
                    buff = symlog(buff)  # 1 ~ 4 くらいの値?
                    self.gamma = buff
                    self.threshold = (1 - torch.sigmoid(norm_stddev - 1 - 4) / 2).unsqueeze(1)

                # 同じ提案が続くならば repeat_penalty を大きくする
                elif strategy == 7:
                    self.repeat_penalty = norm_stddev
                    # 直近 3 回の結果を見る
                    if len(self.model_c.train_inputs[0]) > 3:
                        monitor_window = self.model_c.train_inputs[0][-3:]
                        # monitor_window.std(dim=0).mean()
                        g = monitor_window.mean(dim=0)
                        distance = torch.norm(monitor_window - g, dim=1).mean()
                        # distance が小さければ小さいほど repeat_penalty を小さくする
                        self.repeat_penalty = self.repeat_penalty ** (0.1/distance)

            pof = self.pof(X)

            if self.enable_log:
                base_acqf = symlog(base_acqf)

            if self.enable_positive_only_pof:
                pof = torch.where(
                    base_acqf >= 0,
                    pof,
                    torch.ones_like(pof)
                )

            ret = base_acqf * pof ** self.gamma * self.repeat_penalty

            return ret

    return ACQFWithPOF


# optuna_integration.botorch の獲得関数クラスを置換する関数
def override_optuna_integration_acqf():
    target_module = detect_target.get_botorch_sampler_module()

    from botorch.acquisition.analytic import LogConstrainedExpectedImprovement
    target_module.LogConstrainedExpectedImprovement = acqf_patch_factory(
        LogConstrainedExpectedImprovement
    )

    from botorch.acquisition.analytic import LogExpectedImprovement
    target_module.LogExpectedImprovement = acqf_patch_factory(
        LogExpectedImprovement
    )

    from botorch.acquisition.monte_carlo import qExpectedImprovement
    target_module.qExpectedImprovement = acqf_patch_factory(
        qExpectedImprovement
    )

    from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
    target_module.qNoisyExpectedImprovement = acqf_patch_factory(
        qNoisyExpectedImprovement
    )

    from botorch.acquisition.multi_objective import monte_carlo
    target_module.monte_carlo.qExpectedHypervolumeImprovement = acqf_patch_factory(
        monte_carlo.qExpectedHypervolumeImprovement
    )
    target_module.monte_carlo.qNoisyExpectedHypervolumeImprovement = acqf_patch_factory(
        monte_carlo.qNoisyExpectedHypervolumeImprovement
    )

    from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
    target_module.ExpectedHypervolumeImprovement = acqf_patch_factory(
        ExpectedHypervolumeImprovement
    )

    from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
    target_module.qKnowledgeGradient = acqf_patch_factory(
        qKnowledgeGradient
    )

    from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import qHypervolumeKnowledgeGradient
    target_module.qHypervolumeKnowledgeGradient = acqf_patch_factory(
        qHypervolumeKnowledgeGradient
    )


# optuna_integration の candidate function の中で
# 獲得関数クラスに model_c を set するために
# 共通処理である optimize_acqf の前に当該処理を挿入した関数。
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
    target_module = detect_target.get_botorch_sampler_module()
    target_fun = target_module.optimize_acqf
    return target_fun(*args, **kwargs)


# optimize_acqf を上記関数に置換する関数
# ※ model_c は毎 sample 学習しなおす必要があるので
#   毎 sampler_relative で呼ぶのであって、PoF...Sampler を使えば
#   自動で呼び出される。
def override_optimize_acqf(model_c):
    target_module = detect_target.get_botorch_sampler_module()
    new_func = OptimizeACQFWithModelC(target_module.optimize_acqf)
    new_func.set_model_c(model_c)

    target_module.optimize_acqf = new_func


# 渡された関数の実行前に引数から acqf を取得し model_c を set する。
# optimize_acqf に対して使う。
class OptimizeACQFWithModelC(partial):
    model_c: SingleTaskGP

    def set_model_c(self, model_c: SingleTaskGP):
        self.model_c = model_c

    def __call__(self, *args, **kwargs):

        # acqf を検出
        if 'acq_function' in kwargs.keys():
            # acqf に学習済み model_c を渡す
            acqf = kwargs['acq_function']
            acqf.set_model_c(self.model_c)
            kwargs['acq_function'] = acqf

        else:
            # acqf に学習済み model_c を渡す
            acqf = args[0]
            acqf.set_model_c(self.model_c)
            list(args)[0] = acqf
            args = tuple(args)

        # optimize_acqf を実行する
        return super().__call__(*args, **kwargs)


# ===== PoF を考慮した Sampler =====
from typing import Any, Callable, Sequence
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

    @classmethod
    def set_replaced_flag(cls):
        # クラスメソッド経由で set すれば
        # すべてのインスタンスのメンバーも変更になる
        cls.ACQF_REPLACED = True

    def sample_relative(
            self,
            study: Study,
            trial: FrozenTrial,
            search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:

        # n_startup_trials までは {} を返す
        if len(search_space) == 0:
            return {}
        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        running_trials = [t for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)) if t != trial]
        n_trials = len(completed_trials + running_trials)
        if n_trials < self._n_startup_trials:
            return {}

        # デバッグ用
        print('PoFBoTorchSampler start')

        # ===== 必要ならば獲得関数を置換 =====
        if not self.ACQF_REPLACED:
            override_optuna_integration_acqf()
            self.set_replaced_flag()

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
        override_optimize_acqf(model_c)

        # ===== sampling を実行する =====
        return super().sample_relative(study, trial, search_space)
