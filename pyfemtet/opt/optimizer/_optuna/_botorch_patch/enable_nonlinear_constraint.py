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

from pyfemtet.opt._femopt_core import Constraint
from pyfemtet.opt.optimizer import OptunaOptimizer, logger
from pyfemtet._message import Msg



BotorchConstraint = Callable[[Tensor], Tensor]


# 拘束関数に pytorch の自動微分機能を適用するためのクラス
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


# ユーザー定義関数 (pyfemtet.opt.Constraint) を受け取り、
# botorch で処理できる callable オブジェクトを作成するクラス
class ConvertedConstraint:
    """ユーザーが定義した Constraint を botorch で処理できる形式に変換します。

    `callable()` は形状 `d` の 1 次元テンソルを受け取り、スカラーを返します。
    """

    def __init__(self, constraint: Constraint, study: Study, bound: str, opt: OptunaOptimizer):
        self._constraint: Constraint = constraint
        self._study = study
        self._bound = bound
        self._opt = opt

    def __call__(self, x: Tensor) -> Tensor:  # BotorchConstraint
        """optimize_acqf() に渡される非線形拘束関数の処理です。

        Args:
            x (Tensor): Normalized parameters. Its length is d (== len(prm)).

        Returns:
            float Tensor. >= 0 is feasible.

        """

        norm_x = x.detach().numpy()
        c = evaluate_pyfemtet_cns(self._study, self._constraint, norm_x, self._opt)
        if self._bound == 'lb':
            return Tensor([c - self._constraint.lb])
        elif self._bound == 'ub':
            return Tensor([self._constraint.ub - c])


# list[pyfemtet.opt.Constraint] について、正規化された入力に対し、 feasible or not を返す関数
def is_feasible(study: Study, constraints: list[Constraint], norm_x: np.ndarray, opt: OptunaOptimizer) -> bool:
    feasible = True
    cns: Constraint
    for cns in constraints:
        c = evaluate_pyfemtet_cns(study, cns, norm_x, opt)
        if cns.lb is not None:
            if cns.lb > c:
                feasible = False
                break
        if cns.ub is not None:
            if cns.ub < c:
                feasible = False
                break
    return feasible


# 正規化された入力を受けて pyfemtet.opt.Constraint を評価する関数
def evaluate_pyfemtet_cns(study: Study, cns: Constraint, norm_x: np.ndarray, opt: OptunaOptimizer) -> float:
    """Evaluate given constraint function by given NORMALIZED x.

    Args:
        study (Study): Optuna study. Use to detect search space from last trial's Distribution objects.
        cns (Constraint): PyFemtet's format constraint.
        norm_x (np.ndarray): NORMALIZED values of all parameters.
        opt (OptunaOptimizer): PyFemtet's optimizer. Used for update values of `opt` and `fem` who may be used in `cns`.

    Returns:
        bool: feasible or not.
    """
    # ===== unnormalize x =====
    search_space = study.sampler.infer_relative_search_space(study, None)
    trans = _SearchSpaceTransform(search_space, transform_0_1=True, transform_log=False, transform_step=False)
    params = trans.untransform(norm_x)

    # ===== update OptunaOptimizer and FEMInterface who is referenced by cns =====

    # opt
    opt.set_parameter(params)

    # fem
    if cns.using_fem:
        df_to_fem = opt.variables.get_variables(format='df', filter_pass_to_fem=True)
        opt.fem.update_parameter(df_to_fem)

    # ===== calc cns =====
    return cns.calc(opt.fem)


# botorch の optimize_acqf で非線形拘束を使えるようにするクラス。以下を備える。
#   - 渡すパラメータ nonlinear_constraints を作成する
#   - gen_initial_conditions で feasible なものを返すラッパー関数
class NonlinearInequalityConstraints:
    """botorch の optimize_acqf に parameter constraints を設定するための引数を作成します。"""

    def __init__(self, study: Study, constraints: list[Constraint], opt: OptunaOptimizer):
        self._study = study
        self._constraints = constraints
        self._opt = opt

        self._nonlinear_inequality_constraints = []
        cns: Constraint
        for cns in self._constraints:
            if cns.lb is not None:
                cns_botorch = ConvertedConstraint(cns, self._study, 'lb', self._opt)
                item = (lambda x: GeneralFunctionWithForwardDifference.apply(cns_botorch, x), True)
                self._nonlinear_inequality_constraints.append(item)
            if cns.ub is not None:
                cns_botorch = ConvertedConstraint(cns, self._study, 'ub', self._opt)
                item = (lambda x: GeneralFunctionWithForwardDifference.apply(cns_botorch, x), True)
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
