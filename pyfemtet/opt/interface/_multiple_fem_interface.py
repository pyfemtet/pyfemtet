from ._base_interface import AbstractFEMInterface
from pyfemtet.opt.problem.problem import TrialInput


class MultipleFEMInterface(AbstractFEMInterface):

    # START = 'START'
    # END = 'END'

    def __init__(self):
        # TODO:
        #   list ではなく dict のほうがいいか？
        #   そもそも AbstractFEMInterface が
        #   name を持ったほうがいいか？
        self._fems: list[AbstractFEMInterface] = []

    def add(self, fem: AbstractFEMInterface):
        self._fems.append(fem)

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
