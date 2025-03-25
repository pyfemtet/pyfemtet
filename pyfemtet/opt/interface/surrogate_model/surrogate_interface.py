from __future__ import annotations

from typing import TYPE_CHECKING

from pyfemtet.opt.history import *

from pyfemtet.opt.interface import AbstractFEMInterface

if TYPE_CHECKING:
    from pyfemtet.opt.optimizer import AbstractOptimizer


class AbstractSurrogateModelInterfaceBase(AbstractFEMInterface):
    current_obj_values: dict[str, float]
    train_history: History

    def __init__(
            self,
            history_path: str = None,
            train_history: History = None,
    ):

        # history_path が与えられた場合、train_history をコンストラクトする
        if history_path is not None:
            train_history = History()
            train_history.load_csv(history_path, with_finalize=True)

        assert train_history is not None

        self.train_history = train_history

        self.current_obj_values = {}

    @property
    def _object_pass_to_fun(self):  # lambda に渡される
        return self

    def load_objectives(self, opt: AbstractOptimizer):
        # add_objective された目的のうち、
        # training data に含まれる名前ならば
        # fun を「その時点の current_obj_values を返す関数」で
        # 上書き
        for obj_name, obj in opt.objectives.items():
            if obj_name in self.train_history.obj_names:
                obj.fun = lambda obj_name_=obj_name: self.current_obj_values[obj_name_]

    def load_variables(self, opt: AbstractOptimizer):
        # opt の変数が充分であるかのチェック
        parameters = opt.variable_manager.get_variables()
        assert len(set(self.train_history.prm_names) - set(parameters.keys())) == 0

    def _check_using_fem(self, fun: callable) -> bool:
        return False

    def _check_param_and_raise(self, prm_name) -> None:
        if prm_name not in self.train_history.prm_names:
            raise KeyError(f'Parameter name {prm_name} is not in '
                           f'training input {self.train_history.prm_names}.')
