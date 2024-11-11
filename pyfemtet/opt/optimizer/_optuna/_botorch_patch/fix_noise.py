from typing import Optional

import torch
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch import Module
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from torch import Tensor

from botorch.models import SingleTaskGP

from pyfemtet.opt.optimizer._optuna._botorch_patch import detect_target


class FixedNoiseSingleTaskGP(SingleTaskGP):

    def __init__(
            self,
            train_X: Tensor,
            train_Y: Tensor,
            train_Yvar: Optional[Tensor] = None,
            likelihood: Optional[Likelihood] = None,
            covar_module: Optional[Module] = None,
            mean_module: Optional[Mean] = None,
            outcome_transform: Optional[OutcomeTransform] = None,
            input_transform: Optional[InputTransform] = None
    ) -> None:

        if train_Yvar is None:
            train_Yvar = 1e-4 + torch.zeros_like(train_Y)

        super().__init__(
            train_X,
            train_Y,
            train_Yvar,
            likelihood,
            covar_module,
            mean_module,
            outcome_transform,
            input_transform
        )


def fix_noise():
    """optuna_integration が呼び出す SingleTaskGP を置換します。

    SingleTaskGP の train_YVar を小さい値に指定することで、
    ブラックボックス関数にランダム性がないことを反映し
    サロゲートモデルの生成精度を向上させます。
    """
    target_module = detect_target.get_botorch_sampler_module()
    target_module.SingleTaskGP = FixedNoiseSingleTaskGP
