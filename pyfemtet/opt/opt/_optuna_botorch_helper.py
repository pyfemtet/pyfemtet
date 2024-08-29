from typing import Optional, List, Tuple, Callable
from functools import partial
import inspect

import numpy as np
import optuna.study
import torch
from torch import Tensor
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.utils.transforms import unnormalize
from optuna.study import Study
from botorch.acquisition import AcquisitionFunction

from pyfemtet.opt.opt import AbstractOptimizer
from pyfemtet.opt.parameter import ExpressionEvaluator

# module to monkey patch
import optuna_integration


# モンキーパッチを実行するため、optimize_acqf の引数を MonkyPatch クラスで定義し optuna に上書きされないようにするためのクラス
class NonOverwritablePartial(partial):
    def __call__(self, /, *args, **keywords):
        stored_kwargs = self.keywords
        keywords.update(stored_kwargs)
        return self.func(*self.args, *args, **keywords)


# prm_name を引数に取る関数を optimize_acqf の nonlinear_inequality_constraints に入れられる形に変換する関数
class ConvertedConstraintFunction:
    def __init__(self, fun, prm_args, kwargs, variables: ExpressionEvaluator, study: optuna.study.Study):
        self.fun = fun
        self.prm_args = prm_args
        self.kwargs = kwargs
        self.variables = variables
        self.study = study

        self.bounds = None
        self.prm_name_seq = None

        # fun の prm として使う引数が指定されていなければ fun の引数を取得
        if self.prm_args is None:
            signature = inspect.signature(fun)
            prm_inputs = set([a.name for a in signature.parameters.values()])
        else:
            prm_inputs = set(self.prm_args)

        # 引数の set から kwargs の key を削除
        self.prm_arg_names = prm_inputs - set(kwargs.keys())

        # 変な引数が残っていないか確認
        assert all([(arg in variables.get_parameter_names()) for arg in self.prm_arg_names])

    def __call__(self, x: Tensor or np.ndarray):
        # x: all of normalized parameters whose sequence is sorted by optuna

        if not isinstance(x, Tensor):
            x = torch.tensor(np.array(x)).double()

        x = unnormalize(x, self.bounds)

        # fun で使うパラメータのみ value を取得
        kwargs = self.kwargs
        kwargs.update(
            {k: v for k, v in zip(self.prm_name_seq, x) if k in self.prm_arg_names}
        )

        return self.fun(**kwargs)


# 与えられた獲得関数に拘束を満たさない場合 0 を返すよう加工された獲得関数
class AcqWithConstraint(AcquisitionFunction):

    # noinspection PyAttributeOutsideInit
    def set(self, _org_acq_function: AcquisitionFunction, nonlinear_constraints):
        self._org_acq_function = _org_acq_function
        self._nonlinear_constraints = nonlinear_constraints

    def forward(self, X: Tensor) -> Tensor:
        base = self._org_acq_function.forward(X)

        is_feasible = all([cons(X[0][0]) > 0 for cons, _ in self._nonlinear_constraints])
        if is_feasible:
            return base
        else:
            # penalty = torch.Tensor(size=base.shape)
            # penalty = torch.fill(penalty, -1e10)
            # return base * penalty
            return base * 0.


def remove_infeasible(_ic_batch, nonlinear_constraints):
    # infeasible なものを削除
    remove_indices = []
    for i, ic in enumerate(_ic_batch):  # ic: 1 x len(params) tensor
        # cons: Callable[["Tensor"], "Tensor"]
        is_feasible = all([cons(ic[0]) > 0 for cons, _ in nonlinear_constraints])
        if not is_feasible:
            # ic_batch[i] = torch.nan  # これで無視にならない
            remove_indices.append(i)
    for i in remove_indices[::-1]:
        _ic_batch = torch.cat((_ic_batch[:i], _ic_batch[i + 1:]))
    return _ic_batch


class OptunaBotorchWithParameterConstraintMonkeyPatch:

    def __init__(self, study: Study, opt: AbstractOptimizer):
        self.num_restarts: int = 20
        self.raw_samples_additional: int = 512
        self.eta: float = 2.0
        self.study = study
        self.opt = opt
        self.nonlinear_inequality_constraints = []
        self.additional_kwargs = dict()
        self.bounds = None
        self.prm_name_seq = None

    def add_nonlinear_constraint(self, fun, prm_args, kwargs):
        f = ConvertedConstraintFunction(
            fun,
            prm_args,
            kwargs,
            self.opt.variables,
            self.study,
        )

        # 初期化
        self.nonlinear_inequality_constraints = self.nonlinear_inequality_constraints or []

        # 自身に追加
        self.nonlinear_inequality_constraints.append((f, True))

        # optimize_acqf() に渡す引数に追加
        self.additional_kwargs.update(
            nonlinear_inequality_constraints=self.nonlinear_inequality_constraints
        )

    def _detect_prm_seq_if_needed(self):
        # study から distribution の情報を復元する。
        if self.bounds is None or self.prm_name_seq is None:
            from optuna._transform import _transform_search_space
            # sample_relative の後に呼ばれているから最後の trial は search_space を持つはず
            search_space: dict = self.study.sampler.infer_relative_search_space(self.study, self.study.trials[-1])
            self.bounds = _transform_search_space(search_space, False, False)[0].T
            self.prm_name_seq = list(search_space.keys())

            for cns in self.nonlinear_inequality_constraints:
                cns[0].bounds = torch.tensor(self.bounds)
                cns[0].prm_name_seq = self.prm_name_seq

    def generate_initial_conditions(self, *args, **kwargs):
        self._detect_prm_seq_if_needed()

        # acqf_function を 上書きし、拘束を満たさないならば 0 を返すようにする
        org_acq_function = kwargs['acq_function']
        new_acqf = AcqWithConstraint(None)
        new_acqf.set(org_acq_function, self.nonlinear_inequality_constraints)
        kwargs['acq_function'] = new_acqf

        # initial condition の提案 batch を作成
        # ic: `num_restarts x q x d` tensor of initial conditions.
        # q = 1, d = len(params)
        ic_batch = gen_batch_initial_conditions(*args, **kwargs)

        # 拘束を満たさないものを削除
        ic_batch = remove_infeasible(ic_batch, self.nonlinear_inequality_constraints)

        # 全部なくなっているならばランダムに生成
        if len(ic_batch) == 0:
            print('拘束を満たす組み合わせがなかったのでランダムサンプリングします')
        while len(ic_batch) == 0:
            size = ic_batch.shape
            ic_batch = torch.rand(size=[100, *size[1:]])  # 正規化された変数の組合せ
            ic_batch = remove_infeasible(ic_batch, self.nonlinear_inequality_constraints)

        return ic_batch

    def do_monkey_patch(self):
        """optuna_integration.botorch には optimize_acqf に constraints を渡す方法が用意されていないので、モンキーパッチして渡す

        モンキーパッチ自体は最適化実行前のどの時点で呼んでも機能するが、additional_kwargs の更新後に
        モンキーパッチを呼ぶ必要があるのでコンストラクタにこの処理は入れない。
        各 add_constraint に入れるのはいいかも。

        """

        # === reconstruct argument ``options`` for optimize_acqf ===
        options = dict()  # initialize

        # for nonlinear-constraint
        options.update(dict(batch_limit=1))

        # for gen_candidates_scipy()
        # use COBYLA or SLSQP only.
        options.update(dict(method='SLSQP'))

        # make partial of optimize_acqf used in optuna_integration.botorch and replace to it.
        original_fun = optuna_integration.botorch.optimize_acqf
        overwritten_fun = NonOverwritablePartial(
            original_fun,
            q=1,  # for nonlinear constraints
            options=options,
            num_restarts=20,  # gen_batch_initial_conditions に渡すべきで、self.generate_initial_conditions に渡される変数。
            raw_samples=512,  # gen_batch_initial_conditions に渡すべきで、self.generate_initial_conditions に渡される変数。
            nonlinear_inequality_constraints=self.nonlinear_inequality_constraints,
            ic_generator=self.generate_initial_conditions,
        )
        optuna_integration.botorch.optimize_acqf = overwritten_fun
