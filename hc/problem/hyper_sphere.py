import numpy as np
from numpy import sin, cos, pi
from plotly import graph_objects as go

from hc.problem._base import AbstractProblem, Floats


class HyperSphere(AbstractProblem):

    def __init__(self, dim: int):
        self.r_upper = 0.5
        super().__init__(dim, dim)

    @property
    def bounds(self) -> list[list[float, float]]:
        r_bound = [0., 1.]
        fai_bounds = [[0., pi] for i in range(self.n_prm - 2)]
        theta_bound = [0., 2*pi]

        bound = list()
        bound.append(r_bound)
        bound.extend(fai_bounds)
        bound.append(theta_bound)

        return bound

    @property
    def initial_points(self) -> list[Floats]:
        r = 0.25
        fai = [pi/2 for i in range(self.n_prm - 2)]
        theta = pi
        p = list()
        p.append(r)
        p.extend(fai)
        p.append(theta)
        return [p]

    def _raw_objective(self, x: Floats) -> Floats:
        r, *angles = x

        _x = []
        for i in range(self.n_obj - 1):
            __x = r * np.prod([sin(angles[j]) for j in range(i)])
            __x = __x * cos(angles[i])
            _x.append(__x)
        _x.append(r * np.prod([sin(angles[j]) for j in range(self.n_obj - 1)]))

        return _x

    def _hidden_constraint(self, x: Floats) -> bool:
        r, *angles = x
        return r < self.r_upper

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
        r = min(1., self.r_upper) * np.ones(60)
        theta = np.linspace(0, 2*pi, 60)

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
