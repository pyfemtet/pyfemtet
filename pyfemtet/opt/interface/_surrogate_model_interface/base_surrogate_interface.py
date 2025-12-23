from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from pyfemtet.opt.history import *

from pyfemtet.opt.interface import AbstractFEMInterface
from pyfemtet._i18n import _

if TYPE_CHECKING:
    from pyfemtet.opt.optimizer._base_optimizer import (
        AbstractOptimizer,
        GlobalOptimizationData,
        OptimizationDataPerFEM,
        DIRECTION
    )


__all__ = [
    'AbstractSurrogateModelInterfaceBase',
]


class AbstractSurrogateModelInterfaceBase(AbstractFEMInterface):
    current_obj_values: dict[str, float]
    train_history: History

    def __init__(
            self,
            history_path: str | None = None,
            train_history: History | None = None,
            _output_directions: (
                Sequence[DIRECTION]
                | dict[str, DIRECTION]
                | dict[int, DIRECTION]
                | None
            ) = None
    ):

        self._output_directions = _output_directions

        # history_path が与えられた場合、train_history をコンストラクトする
        if history_path is not None:
            train_history = History()
            train_history.load_csv(history_path, with_finalize=True)

        assert train_history is not None

        self.train_history = train_history

        self.current_obj_values = {}

    @property
    def object_pass_to_fun(self):
        return self.current_obj_values
    
    def contact_to_optimizer(
            self,
            opt: AbstractOptimizer,
            global_data: GlobalOptimizationData,
            ctx: OptimizationDataPerFEM,
    ):
        # output_directions で指定された分を
        # ctx に対して add_objective する。

        # directions を正規化
        name_and_directions: dict[str, DIRECTION]
        if self._output_directions is None:
            name_and_directions = {}
        elif isinstance(self._output_directions, dict):
            name_and_directions = {}
            obj_names = self.train_history.obj_names
            for obj_name_or_index, direction in self._output_directions.items():
                if isinstance(obj_name_or_index, int):
                    obj_name = obj_names[obj_name_or_index]
                else:
                    obj_name = obj_name_or_index
                name_and_directions.update({obj_name: direction})
        else:
            obj_names = self.train_history.obj_names
            if len(self._output_directions) != len(obj_names):
                raise ValueError(_(
                    en_message='The length of _output_directions passed as a list '
                            'must be same with that of the history\'s objective '
                            'names.',
                    jp_message='_output_directions をリストで渡す場合は'
                            'その長さが history の目的関数数と一致して'
                            'いなければなりません。'
                ))
            name_and_directions = {
                obj_name: direction
                for obj_name, direction
                in zip(obj_names, self._output_directions)
            }

        # global に紐づいていると objective_pass_to_fun が Sequence になるので
        # global に登録されたものの内 train に含まれるものは ctx に移動
        obj_names_to_remove = set()
        for obj_name, obj in global_data.objectives.items():
            if obj_name in self.train_history.obj_names:
                obj_names_to_remove.add(obj_name)
                ctx.objectives.update({obj_name: obj})
        for obj_name in obj_names_to_remove:
            global_data.objectives.pop(obj_name)

        # directions に含まれるものをすべて ctx に追加または上書き
        for obj_name, direction in name_and_directions.items():

            # validation
            if obj_name not in self.train_history.obj_names:
                raise ValueError(_(
                    en_message="The objective name {obj_name} is "
                               "not in the train_history's objectives: "
                               "{obj_names}",
                    jp_message="目的関数名 {obj_name} は "
                               "train_history の目的関数: "
                               "{obj_names} に含まれていません。",
                    obj_name=obj_name,
                    obj_names=self.train_history.obj_names,
                ))

            def dummy(*args, **kwargs):
                assert False

            ctx.add_objective(obj_name, dummy, direction, supress_duplicated_name_check=True)
        
        # ctx の目的関数のうち train に含まれるものを差し替え
        # directions の目的変数名を validation しているので漏れはないはず
        for obj_name, obj in ctx.objectives.items():
            if obj_name in self.train_history.obj_names:
                # global 由来のものは何が入っているかわからないので
                # 必要な変数を確実に上書き
                obj.direction = name_and_directions.get(obj_name, obj.direction)
                obj.args = tuple()
                obj.kwargs = dict()
                obj.fun = lambda obj_values, obj_name_=obj_name: obj_values[obj_name_]

        # これが呼ばれた後に optimizer の同期はされないので
        # ここで同期する
        opt._initialize_objectives()
        opt.objectives.update(ctx.objectives)
        opt.objectives.update(global_data.objectives)

    def _check_using_fem(self, fun: callable) -> bool:
        return False

    def _check_param_and_raise(self, prm_name) -> None:
        if prm_name not in self.train_history.prm_names:
            raise KeyError(f'Parameter name {prm_name} is not in '
                           f'training input {self.train_history.prm_names}.')
