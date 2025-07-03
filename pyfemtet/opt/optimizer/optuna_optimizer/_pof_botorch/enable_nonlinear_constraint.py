import numpy as np

import torch
from torch import Tensor

from optuna._transform import _SearchSpaceTransform

from botorch.optim.initializers import gen_batch_initial_conditions

from pyfemtet._i18n import _
from pyfemtet.logger import get_module_logger
from pyfemtet.opt.problem.problem import *
from pyfemtet.opt.optimizer import AbstractOptimizer

logger = get_module_logger('opt.optimizer')


__all__ = ['NonlinearInequalityConstraints']


# 拘束関数に pytorch の自動微分機能を適用するためのクラス
class _GeneralFunctionWithForwardDifference(torch.autograd.Function):
    """自作関数を pytorch で自動微分するためのクラスです。

    ユーザー定義関数を botorch 形式に変換する過程で微分の計算ができなくなるのでこれが必要です。
    """

    @staticmethod
    def forward(ctx, f, xs):
        ys = f(xs)
        ctx.save_for_backward(xs, ys)
        ctx.f = f
        return ys

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx, grad_output):
        xs, ys = ctx.saved_tensors
        f = ctx.f
        dx = 0.001  # 入力は normalized なので決め打ちでよい
        diff = []
        xs_ = xs.detach()  # xs に余計な計算履歴を残さないために、detachする。
        for i in range(len(xs_)):
            xs_[i] += dx
            diff.append(torch.sum(grad_output * (f(xs) - ys)))
            xs_[i] -= dx
        diff = torch.tensor(diff).to(xs) / dx
        return None, diff


# 入力 x を受けて pyfemtet.opt.Constraint を評価する関数
def _evaluate_pyfemtet_cns(
        cns: Constraint,
        opt: AbstractOptimizer,
        trans: _SearchSpaceTransform,
        x: np.ndarray,
) -> float:

    # un-transform continuous to original
    new_params: dict[str, ...] = trans.untransform(x)
    params: TrialInput = opt.variable_manager.get_variables()

    # update
    for prm_name, prm in params.items():
        if prm_name in new_params:
            prm.value = new_params[prm_name]
    opt.variable_manager.eval_expressions()

    # update fem (very slow!)
    if cns.using_fem:
        logger.warning(_(
            en_message='Accessing FEM API inside hard constraint '
                       'function may be very slow.',
            jp_message='hard constraint 関数の評価中の FEM への'
                       'アクセスは著しく時間がかかりますので'
                       'ご注意ください。'
        ))
        pass_to_fem = opt.variable_manager.get_variables(
            filter='pass_to_fem', format='raw'
        )
        opt.fem.update_parameter(pass_to_fem)

    # eval
    return cns.eval(opt.fem)


# ユーザー定義関数 (pyfemtet.opt.Constraint) を受け取り、
# botorch で処理できる callable オブジェクトを作成するクラス
class _ConvertedConstraint:

    def __init__(
            self,
            cns: Constraint,
            opt: AbstractOptimizer,
            trans: _SearchSpaceTransform,
            ub_or_lb: str,
            constraint_enhancement: float = None,
    ):
        self.cns: Constraint = cns
        self.ub_or_lb = ub_or_lb
        self.opt = opt
        self.trans = trans
        self.ce = constraint_enhancement or 0.

    def __call__(self, X: Tensor) -> Tensor:  # BotorchConstraint
        """optimize_acqf() に渡される非線形拘束関数の処理です。

        Args:
            X (Tensor): Normalized parameters. Its length is d (== len(prm)).

        Returns:
            float Tensor. >= 0 is feasible.

        """

        x = X.detach().cpu().numpy()
        cns_value = _evaluate_pyfemtet_cns(
            self.cns,
            self.opt,
            self.trans,
            x,
        )
        if self.ub_or_lb == 'lb':
            return Tensor([cns_value - self.cns.lower_bound - self.ce])
        elif self.ub_or_lb == 'ub':
            return Tensor([self.cns.upper_bound - cns_value - self.ce])
        else:
            assert False


# list[pyfemtet.opt.Constraint] について、正規化された入力に対し、 feasible or not を返す関数
def _is_feasible(
        constraints: list[Constraint],
        opt: AbstractOptimizer,
        trans: _SearchSpaceTransform,
        x: np.ndarray,
        constraint_enhancement: float = None,
        constraint_scaling: float = None,
) -> bool:
    for cns in constraints:
        cns_value = _evaluate_pyfemtet_cns(cns, opt, trans, x)
        cns_result = ConstraintResult(cns, opt.fem, cns_value, constraint_enhancement, constraint_scaling)
        if cns_result.check_violation() is not None:
            return False
    return True


# botorch の optimize_acqf で非線形拘束を使えるようにするクラス。以下を備える。
#   - 渡すパラメータ nonlinear_constraints を作成する
#   - gen_initial_conditions で feasible なものを返すラッパー関数
class NonlinearInequalityConstraints:
    """botorch の optimize_acqf に parameter constraints を設定するための引数を作成します。"""

    def __init__(
            self,
            constraints: list[Constraint],
            opt: AbstractOptimizer,
            trans: _SearchSpaceTransform,
            constraint_enhancement: float = None,
            constraint_scaling: float = None,
    ):
        self.trans = trans
        self.constraints = constraints
        self.opt = opt
        self.ce = constraint_enhancement or 0.
        self.cs = constraint_scaling or 1.

        self.nonlinear_inequality_constraints = []
        cns: Constraint
        for cns in self.constraints:

            if cns.lower_bound is not None:
                cns_botorch = _ConvertedConstraint(cns, self.opt, self.trans, 'lb', self.ce)
                item = (lambda x: _GeneralFunctionWithForwardDifference.apply(cns_botorch, x), True)
                self.nonlinear_inequality_constraints.append(item)

            if cns.upper_bound is not None:
                cns_botorch = _ConvertedConstraint(cns, self.opt, self.trans, 'ub', self.ce)
                item = (lambda x: _GeneralFunctionWithForwardDifference.apply(cns_botorch, x), True)
                self.nonlinear_inequality_constraints.append(item)

    def _filter_feasible_conditions(self, ic_batch):
        # List to store feasible initial conditions
        feasible_ic_list = []

        for each_num_restarts in ic_batch:
            feasible_q_list = []
            for each_q in each_num_restarts:
                x: np.ndarray = each_q.detach().cpu().numpy()  # normalized parameters
                if _is_feasible(self.constraints, self.opt, self.trans, x, self.ce, self.cs):
                    feasible_q_list.append(each_q)  # Keep only feasible rows

            if feasible_q_list:  # Only add if there are feasible rows
                feasible_ic_list.append(torch.stack(feasible_q_list))

        # Stack feasible conditions back into tensor format
        if feasible_ic_list:
            return torch.stack(feasible_ic_list).to(ic_batch)
        else:
            return None  # Return None if none are feasible

    @staticmethod
    def _generate_random_initial_conditions(ic_batch):
        # Generates random initial conditions with the same shape as ic_batch
        return torch.rand(ic_batch.shape).to(ic_batch)

    def _generate_feasible_initial_conditions(self, *args, **kwargs):
        # A `num_restarts x q x d` tensor of initial conditions.
        ic_batch = gen_batch_initial_conditions(*args, **kwargs)
        feasible_ic_batch = self._filter_feasible_conditions(ic_batch)

        while feasible_ic_batch is None:
            # Generate new random ic_batch with the same shape
            logger.warning(_(
                en_message='gen_batch_initial_conditions() failed to generate '
                           'feasible initial conditions for acquisition '
                           'function optimization sub-problem, '
                           'so trying to use random feasible parameters '
                           'as initial conditions.'
                           'The constraint functions or solutions spaces '
                           'may be too complicated.',
                jp_message='gen_batch_initial_conditions() は獲得関数を最適化する'
                           'サブ最適化問題で実行可能な初期値を生成できませんでした。'
                           'サブ最適化問題の初期値としてランダムなパラメータを設定します。'
                           '拘束が厳しすぎるか、目的関数が複雑すぎるかもしれません。'
            ))
            random_ic_batch = self._generate_random_initial_conditions(ic_batch)
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
            options_batch_limit=1,
            nonlinear_inequality_constraints=self.nonlinear_inequality_constraints,
            ic_generator=self._generate_feasible_initial_conditions,
        )
