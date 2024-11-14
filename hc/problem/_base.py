from typing import Sequence, SupportsFloat
from abc import ABC, abstractmethod, abstractproperty
import plotly.graph_objects as go


Floats = Sequence[SupportsFloat]


class InfeasibleException(Exception):
    pass


class AbstractProblem(ABC):
    def __init__(self, n_prm, n_obj):
        self.n_prm = n_prm
        self.n_obj = n_obj

    @property
    @abstractmethod
    def bounds(self) -> list[list[float, float]] or Floats:
        pass

    @property
    @abstractmethod
    def initial_points(self) -> list[Floats]:
        pass

    def objective(self, x: Floats) -> Floats:
        if self._hidden_constraint(x):
            return self._raw_objective(x)
        else:
            raise InfeasibleException

    @abstractmethod
    def _raw_objective(self, x: Floats) -> Floats:
        pass

    @abstractmethod
    def _hidden_constraint(self, x: Floats) -> bool:
        pass

    def create_base_figure(self) -> go.Figure:
        return go.Figure()
