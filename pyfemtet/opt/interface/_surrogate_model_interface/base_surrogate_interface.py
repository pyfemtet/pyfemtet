from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from pyfemtet.opt.history import *

from pyfemtet.opt.interface import AbstractFEMInterface
from pyfemtet._i18n import _

if TYPE_CHECKING:
    from pyfemtet.opt.optimizer import AbstractOptimizer


__all__ = [
    'AbstractSurrogateModelInterfaceBase',
]


class AbstractSurrogateModelInterfaceBase(AbstractFEMInterface):
    _load_problem_from_fem = True
    current_obj_values: dict[str, float]
    train_history: History

    def __init__(
            self,
            history_path: str = None,
            train_history: History = None,
            _output_directions: (
                Sequence[str | float]
                | dict[str, str | float]
                | dict[int, str | float]
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

    def load_objectives(self, opt: AbstractOptimizer):

        # output directions が与えられない場合、
        # opt.add_objective との整合をチェックする
        if self._output_directions is None:

            # add_objective された目的のうち、
            # training data に含まれる名前ならば
            # fun を「その時点の current_obj_values を返す関数」で
            # 上書き
            obj_name: str
            for obj_name, obj in opt.objectives.items():
                # あれば上書き、なければ surrogate 最適化の際に
                # 新しく追加した model を使わない目的関数と見做して何もしない
                if obj_name in self.train_history.obj_names:
                    obj.fun = lambda _, obj_name_=obj_name: self.current_obj_values[obj_name_]
                    obj.args = tuple()
                    obj.kwargs = dict()

        # dict で与えられた場合
        elif isinstance(self._output_directions, dict):

            # index 入力か str 入力かで統一されているか確認
            keys = tuple(self._output_directions.keys())
            assert all([isinstance(key, type(keys[0])) for key in keys]), _(
                en_message='The keys of _output_directions must be '
                           'all-int or all-str.',
                jp_message='_output_directions のキーは int または str で'
                           '統一されていなければなりません。',
            )

            # index がキーである場合
            if isinstance(keys[0], int):

                for index, direction in self._output_directions.items():
                    obj_name = self.train_history.obj_names[index]

                    opt.add_objective(
                        name=obj_name,
                        fun=lambda _, obj_name_=obj_name: self.current_obj_values[obj_name_],
                        direction=direction,
                        args=(),
                        kwargs={},
                    )

            # obj_name がキーである場合
            if isinstance(keys[0], str):

                for obj_name, direction in self._output_directions.items():
                    assert obj_name in self.train_history.obj_names, _(
                        en_message='The objective name passed as a key of '
                                   '_output_direction must be one of the history\'s '
                                   'objective names. Passed name: {obj_name} / '
                                   'History\'s names: {obj_names}',
                        jp_message='_output_directions に目的関数名を与える場合は'
                                   'history に含まれる名前を指定しなければなりません。'
                                   '与えられた目的名: {obj_name} / history に含まれる'
                                   '目的名: {obj_names}',
                        obj_name=obj_name,
                        obj_names=', '.join(self.train_history.obj_names)
                    )

                    opt.add_objective(
                        name=obj_name,
                        fun=lambda obj_name_=obj_name: self.current_obj_values[obj_name_],
                        direction=direction,
                        args=(),
                        kwargs={},
                    )

        # tuple で与えられた場合
        elif isinstance(self._output_directions, list) \
                or isinstance(self._output_directions, tuple):

            obj_names = self.train_history.obj_names
            assert len(self._output_directions) == len(obj_names), _(
                en_message='The length of _output_directions passed as a list '
                           'must be same with that of the history\'s objective '
                           'names.',
                jp_message='_output_directions をリストで渡す場合は'
                           'その長さが history の目的関数数と一致して'
                           'いなければなりません。'
            )

            for obj_name, direction in zip(obj_names, self._output_directions):
                opt.add_objective(
                    name=obj_name,
                    fun=lambda _, obj_name_=obj_name: self.current_obj_values[obj_name_],
                    direction=direction,
                    args=(),
                    kwargs={},
                )

    def load_variables(self, opt: AbstractOptimizer):
        # opt の変数が充分であるかのチェックのみ
        parameters = opt.variable_manager.get_variables()
        assert len(set(self.train_history.prm_names) - set(parameters.keys())) == 0

    def _check_using_fem(self, fun: callable) -> bool:
        return False

    def _check_param_and_raise(self, prm_name) -> None:
        if prm_name not in self.train_history.prm_names:
            raise KeyError(f'Parameter name {prm_name} is not in '
                           f'training input {self.train_history.prm_names}.')
