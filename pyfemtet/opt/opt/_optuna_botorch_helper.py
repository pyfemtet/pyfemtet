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

from pyfemtet.opt._femopt_core import History
from pyfemtet.opt.opt import AbstractOptimizer
from pyfemtet.opt.parameter import ExpressionEvaluator

# module to monkey patch
import optuna_integration


class NonOverwritablePartial(partial):
    def __call__(self, /, *args, **keywords):
        stored_kwargs = self.keywords
        keywords.update(stored_kwargs)
        return self.func(*self.args, *args, **keywords)


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


class OptunaBotorchWithParameterConstraintMonkeyPatch:

    def __init__(
            self,
            study: Study,
            opt: AbstractOptimizer,
            inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
            equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
            nonlinear_inequality_constraints: Optional[List[Tuple[Callable, bool]]] = None,
    ):
        self.study = study
        self.opt = opt
        self.nonlinear_inequality_constraints = nonlinear_inequality_constraints
        self.additional_kwargs = dict(
            q=1,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            nonlinear_inequality_constraints=nonlinear_inequality_constraints,
            ic_generator=self.generate_initial_conditions,
        )
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

    def detect_prm_seq_if_needed(self):
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

    def generate_initial_conditions(
        self,
        *args,
        **kwargs,
    ):
        from time import time
        start = time()

        batch_initial_conditions_feasible = []

        self.detect_prm_seq_if_needed()

        if len(batch_initial_conditions_feasible) == 0:

            # n 回 batch_initial_conditions を行い、feasible のもののみ抽出
            counter = 0
            n = 1
            start2 = time()
            # print(kwargs)
            # print(len(self.prm_name_seq))
            num_restarts = max(20, len(self.prm_name_seq))  # デフォルトである 20 以上
            num_random_points_before_concider_heuristic = max(num_restarts, 0)  # 関数が num_restarts 以上を必要とする。デフォルト 1024 は多すぎる。
            kwargs.update(dict(num_restarts=num_restarts))  # 20
            kwargs.update(dict(raw_samples=num_random_points_before_concider_heuristic))  # 1024
            batch_initial_conditions = gen_batch_initial_conditions(*args, **kwargs)
            print(f'  DEBUG: batch_initial_conditions: {int(time() - start2)} sec')
            while True:
                counter += 1
                if counter > n:
                    break

                # 各初期値提案について
                for ic_candidate in batch_initial_conditions:
                    # すべての非線形拘束について
                    for cns in self.nonlinear_inequality_constraints:
                        # ひとつでも拘束を満たしていなければ初期値提案の処理を抜ける
                        f: ConvertedConstraintFunction = cns[0]
                        if f(*ic_candidate) < 0:
                            break
                    else:
                        # 初期値提案がすべての非線形拘束を満たしたら feasible に入れる
                        batch_initial_conditions_feasible.append(ic_candidate.numpy())

        if len(batch_initial_conditions_feasible) == 0:

            # study の履歴から initial conditions を作成。
            # 内部で acquisition function が更新されるので同じ値が提案されてもよい
            for trial in self.study.best_trials:
                # BoTorchSampler 内部実装の並びと trial.params の並びは一致しないので並び替える
                prm_names, values = list(trial.params.keys()), list(trial.params.values())
                indices = [prm_names.index(seq) for seq in self.prm_name_seq]
                sorted_values = [values[idx] for idx in indices]
                normalized_values = (np.array(sorted_values).astype(float) - self.bounds[0]) / (self.bounds[1] - self.bounds[0])

                # すべての非線形拘束について
                for cns in self.nonlinear_inequality_constraints:
                    # ひとつでも拘束を満たしていなければ初期値提案の処理を抜ける
                    f: ConvertedConstraintFunction = cns[0]
                    if f(normalized_values) < 0:
                        break
                else:
                    batch_initial_conditions_feasible.append([normalized_values])

        # もしここで feasible なものがなければ、どうしようもない
        if len(batch_initial_conditions_feasible) == 0:
            raise RuntimeError("candidate 探索のための拘束を満たす初期値を提案できませんでした。")

        end = time()

        print(f'DEBUG: generate_initial_conditions: {int(end-start)} sec')

        return torch.tensor(np.array(batch_initial_conditions_feasible)).double()

    def do_monkey_patch(self):
        """optuna_integration.botorch には optimize_acqf に constraints を渡す方法が用意されていないので、モンキーパッチして渡す

        モンキーパッチ自体は最適化実行前のどの時点で呼んでも機能するが、additional_kwargs の更新後に
        モンキーパッチを呼ぶ必要があるのでコンストラクタにこの処理は入れない。
        各 add_constraint に入れるのはいいかも。

        """

        # reconstruct argument ``options`` for optimize_acqf
        options = dict()  # initialize

        # for nonlinear-constraint
        options.update(dict(batch_limit=1))

        # for gen_candidates_scipy()
        options.update(dict(method='SLSQP'))  # use COBYLA instead of SLSQP. This is the only method that can process dict format constraints. COBYLA is a bit stable but slow

        # make partial of optimize_acqf used in optuna_integration.botorch and replace to it.
        original_fun = optuna_integration.botorch.optimize_acqf
        overwritten_fun = NonOverwritablePartial(
            original_fun,
            options=options,
            **self.additional_kwargs,
        )
        optuna_integration.botorch.optimize_acqf = overwritten_fun
