from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from hc.problem._base import AbstractProblem, Floats, InfeasibleException


class AbstractSampler(ABC):
    def __init__(self, problem: AbstractProblem):
        self.fig: go.Figure = go.Figure()
        self.problem: AbstractProblem = problem
        d = {'feasibility': []}
        d.update({'method': []})
        d.update({f'prm_{k}': [] for k in range(self.problem.n_prm)})
        d.update({f'obj_{k}': [] for k in range(self.problem.n_obj)})
        self.df: pd.DataFrame = pd.DataFrame(d)
        self.setup()

    @abstractmethod
    def setup(self):
        pass

    def _get_col(self, kind) -> list[str]:
        return [col for col in self.df.columns if kind in col]

    def get_prm_columns(self) -> list[str]:
        return self._get_col('prm')

    def get_obj_columns(self) -> list[str]:
        return self._get_col('obj')

    @abstractmethod
    def candidate_x(self) -> Floats:
        pass

    def sampling(self, x: Floats = None) -> None:
        """
        Args:
            x: 入力です。指定しない場合、candidate_x() を呼びます。
        """

        if x is None:
            x = self.candidate_x()

        # update df
        y: Floats = self.problem._raw_objective(x)
        feasibility: bool = self.problem._hidden_constraint(x)

        # update property
        if isinstance(x, np.ndarray):
            _x = x.tolist()
        else:
            _x = x
        method = 'initial points' if _x in self.problem.initial_points else type(self).__name__

        row = pd.DataFrame(
            [[
                feasibility,
                method,
                *x,
                *y
            ]],
            columns=self.df.columns,
            dtype=object,
        )
        self.df = pd.concat([self.df, row], axis=0)

    def update_figure(self):
        # update figure
        self.fig = self.problem.create_base_figure()

        scat = px.scatter(
            self.df,
            x='obj_0',
            y='obj_1',
            symbol='feasibility',
            symbol_map={
                True: 'circle',
                False: 'circle-open'
            },
            color='method',
            color_discrete_sequence=px.colors.qualitative.Alphabet
            # color_discrete_map={'initial points': 'rgb(20, 100, 20)'},
        )
        for trace in scat.data:
            self.fig.add_trace(trace)

    def show_figure(self):
        self.update_figure()
        self.fig.show()

    def save_figure(self, path):
        self.update_figure()
        self.fig.write_image(path, engine="orca")

    def save_csv(self, path):
        self.df.to_csv(path, encoding='cp932', index=None)

    def load_csv(self, path):
        self.df = pd.read_csv(path, encoding='cp932', header=0)

    @property
    def x(self) -> list[Floats] or np.ndarray:
        return self.df[self.get_prm_columns()].values.astype(float)

    @property
    def x_feasible(self) -> list[Floats] or np.ndarray:
        idx = self.df['feasibility']
        return self.df[idx][self.get_prm_columns()].values.astype(float)

    @property
    def y(self) -> list[Floats] or np.ndarray:
        return self.df[self.get_obj_columns()].values.astype(float)

    @property
    def y_feasible(self) -> list[Floats] or np.ndarray:
        idx = self.df['feasibility']
        return self.df[idx][self.get_obj_columns()].values.astype(float)
