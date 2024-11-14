import numpy as np
from numpy import sin, cos, pi
from plotly import graph_objects as go

from hc.problem._base import AbstractProblem, Floats

_default_spots = [
    dict(center=[0.25, 0.75], radii=0.05),
    dict(center=[0.75, 0.25], radii=0.05),
]


def inside_spot(x, spot: dict) -> bool:
    x = np.array(x)
    c = np.array(spot['center'])
    r = float(spot['radii'])
    return ((x - c) ** 2).sum() < r ** 2


class HyperSpotsInSquare(AbstractProblem):

    def __init__(self, spots, dim):
        self.spots = spots
        self.dim = dim
        super().__init__(dim, dim)

    @property
    def bounds(self) -> list[list[float]]:
        return [[0., 1.] for _ in range(self.dim)]

    @property
    def initial_points(self) -> list[Floats]:
        return [d['center'] for d in self.spots]

    def _raw_objective(self, x: Floats) -> Floats:
        return x

    def _hidden_constraint(self, x: Floats) -> bool:
        return any([inside_spot(x, spot) for spot in self.spots])

    def create_base_figure(self) -> go.Figure:

        fig = go.Figure()

        # 定義域
        x = [0, 0, 1, 1, 0]
        y = [0, 1, 1, 0, 0]

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='none',
            fill='toself',
            name='定義域',
            fillcolor='rgb(200, 200, 200)'
        ))


        # feasible 領域
        for i, spot in enumerate(self.spots):
            center = spot['center']
            radii = spot['radii']

            theta = np.linspace(0, 2*pi, 60)

            x = center[0] + radii * cos(theta)
            y = center[1] + radii * sin(theta)

            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='none',
                fill='toself',
                name='実行可能',
                fillcolor='rgb(150, 200, 200)',
                showlegend=(i == 0),
            ))

        # fig.show()

        return fig



class SpotsInSquare(AbstractProblem):

    def __init__(self, spots=None):
        if spots is not None:
            self.spots = spots
        else:
            self.spots = _default_spots
        super().__init__(2, 2)

    @property
    def bounds(self) -> list[list[float, float]]:
        return [[0., 1.], [0., 1]]

    @property
    def initial_points(self) -> list[Floats]:
        return [d['center'] for d in self.spots]

    def _raw_objective(self, x: Floats) -> Floats:
        return x

    def _hidden_constraint(self, x: Floats) -> bool:
        return any([inside_spot(x, spot) for spot in self.spots])

    def create_base_figure(self) -> go.Figure:

        fig = go.Figure()

        # 定義域
        x = [0, 0, 1, 1, 0]
        y = [0, 1, 1, 0, 0]

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='none',
            fill='toself',
            name='定義域',
            fillcolor='rgb(200, 200, 200)'
        ))


        # feasible 領域
        for i, spot in enumerate(self.spots):
            center = spot['center']
            radii = spot['radii']

            theta = np.linspace(0, 2*pi, 60)

            x = center[0] + radii * cos(theta)
            y = center[1] + radii * sin(theta)

            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='none',
                fill='toself',
                name='実行可能',
                fillcolor='rgb(150, 200, 200)',
                showlegend=(i == 0),
            ))

        # fig.show()

        return fig
