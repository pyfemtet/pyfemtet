import numpy as np
from numpy import sin, cos, pi
from plotly import graph_objects as go

from hc.problem._base import AbstractProblem, Floats

COEF = 5 / 12 * pi


class Spiral(AbstractProblem):

    def __init__(self):
        super().__init__(2, 2)

    @property
    def bounds(self) -> list[list[float, float]]:
        return [[0., 1.], [0., 2 * pi]]

    @property
    def initial_points(self) -> list[Floats]:
        return [[0.5, 0.5 * COEF * 1.5]]

    def _raw_objective(self, x: Floats) -> Floats:
        r, theta = x
        return r * cos(theta), r * sin(theta)

    def _hidden_constraint(self, x: Floats) -> bool:
        r, theta = x
        c1 = COEF * r - theta  # <= 0 is feasible
        c2 = theta - 2 * COEF * r  # <= 0 is feasible
        return (c1 <= 0) and (c2 <= 0)

    def create_base_figure(self) -> go.Figure:

        fig = go.Figure()

        # 定義域
        r = np.ones(60)
        theta = np.linspace(0, 2*pi, 60)

        x = r * cos(theta)
        y = r * sin(theta)

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='none',
            fill='toself',
            name='定義域',
            fillcolor='rgb(200, 200, 200)'
        ))


        # feasible 領域
        r = np.array([
            np.linspace(0, 1, 20),
            np.ones(20),
            np.linspace(1, 0, 20),
        ]).flatten()
        theta = np.array([
            r[:20] * COEF,
            np.linspace(COEF, 2*COEF, 20),
            r[40:] * 2*COEF,
        ]).flatten()

        x = r * cos(theta)
        y = r * sin(theta)

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='none',
            fill='toself',
            name='実行可能',
            fillcolor='rgb(150, 200, 200)'
        ))

        # fig.show()

        return fig
