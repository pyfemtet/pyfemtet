from __future__ import annotations

from typing import TYPE_CHECKING

from ._base_interface import AbstractFEMInterface
from pyfemtet.opt.problem.problem import TrialInput


if TYPE_CHECKING:
    from pyfemtet.opt.optimizer._base_optimizer import FEMContext


class MultipleFEMInterface(AbstractFEMInterface):

    def __init__(self):
        # TODO:
        #   list ではなく dict のほうがいいか？
        #   そもそも AbstractFEMInterface が
        #   name を持ったほうがいいか？
        self._fems: list[AbstractFEMInterface] = []
        self._ctxs: list[FEMContext] = []

    def __iter__(self):
        return iter(self._fems)

    def __len__(self):
        return len(self._fems)

    def __getitem__(self, index: int) -> AbstractFEMInterface:
        return self._fems[index]

    @property
    def ordered_contexts(self) -> list['FEMContext']:
        return self._ctxs

    def add(self, fem: AbstractFEMInterface) -> FEMContext:
        from pyfemtet.opt.optimizer._base_optimizer import FEMContext
        ctx = FEMContext(fem=fem)
        self._fems.append(fem)
        self._ctxs.append(ctx)
        return ctx

    def remove(self, fem: AbstractFEMInterface):
        self._fems.remove(fem)

    def pop(self, index: int):
        self._fems.pop(index)

    # TODO: この属性がそもそも AbstractFEMInterface に必要か検討する。
    @property
    def _load_problem_from_fem(self):
        return any(fem._load_problem_from_fem for fem in self._fems)

    def update_parameter(self, x: TrialInput) -> None:
        for fem in self._fems:
            fem.update_parameter(x)

    def update(self):
        for fem in self._fems:
            fem.update()

    def _check_param_and_raise(self, prm_name) -> None:
        # TODO:
        #   - チェックする前に、与えられた prm_name が
        #     どの FEM に属するかを特定し、
        #     その FEM に対してのみチェックを行うようにする。
        #   - そのために 与えられた prm_name がどの FEM に属するかを
        #     管理する仕組みが必要。
        #   - 変数は ctx が管理しているから、その仕組みは ctx にしか持てない。
        #   - なので _check_param_and_raise は
        #     optimizer が直接呼び出してはならず、
        #     ctx のほうで prm_name をフィルタして呼び出す必要がある。
        #   - まずはこのケースを通るはずのテストを作成してから実装する。
        pass

    def _get_additional_data(self) -> dict:
        data = {}
        for i, fem in enumerate(self._fems):
            data.update(fem._get_additional_data())
        return data
