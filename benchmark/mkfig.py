import os
from glob import glob
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from common import *


class Graph:

    def __init__(self):
        self.figure = go.Figure()
        self.data = {
            'algorithm': [],
            'seed': [],
            'x': [],
            'y': [],
        }
        self.problem_name = None

    @staticmethod
    def get_min_sequence(y):
        out = [y[0]]
        for y_ in y[1:]:
            out.append(
                min(
                    out[-1],
                    y_
                )
            )
        return np.array(out)

    def add_result(self, result_path, single_objective=False):

        problem_name, algorithm_name, algorithm_config, seed = parse_history_path(result_path)

        y = np.loadtxt(result_path, ndmin=1)

        if single_objective:
            y = np.array(self.get_min_sequence(y))

        n = len(y)
        if n == 0:
            y = np.array([0])
            n = 1

        x = 1 + np.arange(n)

        if self.problem_name is None:
            self.problem_name = problem_name
        else:
            assert self.problem_name == problem_name

        self.data['algorithm'].extend([algorithm_name] * n)  # str
        self.data['seed'].extend([seed] * n)  # str
        self.data['x'].extend(x.tolist())  # float
        self.data['y'].extend(y.tolist())  # float

    def add_results_from_directory(self, dir_path, single_objective=False):
        for path in glob(os.path.join(dir_path, '*.txt')):
            self.add_result(path, single_objective)

    def show(self):
        # raw data
        df = pd.DataFrame(self.data)
        group = df.groupby(['x', 'algorithm'])
        mean = group.mean(numeric_only=True)['y']
        std = group.std(numeric_only=True)['y']

        # statistical data
        figure = go.Figure()
        color_seq = px.colors.qualitative.G10
        for algorithm, color in zip(np.unique(df['algorithm'].values), color_seq):
            mi_loc = mean.index.get_locs([slice(None), [algorithm]])
            mean_per_algorithm = mean.iloc[mi_loc].values
            std_per_algorithm: np.ndarray = std.iloc[mi_loc].values
            std_per_algorithm[np.where(np.isnan(std_per_algorithm))] = 0.
            n = len(std_per_algorithm)
            x = (1 + np.arange(n)).tolist()

            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=mean_per_algorithm,
                    mode='lines',
                    name=algorithm,
                    line=dict(color=color),
                    legendgroup=algorithm,
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=x + x[::-1],
                    y=(
                        (mean_per_algorithm + std_per_algorithm).tolist()
                        + (mean_per_algorithm - std_per_algorithm).tolist()[::-1]
                    ),
                    mode='none',
                    fill='toself',
                    opacity=0.25,
                    legendgroup=algorithm,
                    fillcolor=color,
                    showlegend=False,
                )
            )

            pdf = df[df['algorithm'] == algorithm]

            x_last = []
            y_last = []
            for seed_ in np.unique(pdf['seed'].values):
                ppdf = pdf[pdf['seed'] == seed_]
                x_last.append(ppdf['x'].values[-1])
                y_last.append(ppdf['y'].values[-1])

            figure.add_trace(
                go.Scatter(
                    x=x_last, y=y_last,
                    mode='markers',
                    marker=dict(color=color, symbol='circle-open'),
                    legendgroup=algorithm,
                    showlegend=False,
                )
            )
        figure.show()


if __name__ == '__main__':

    graph = Graph()
    # graph.add_results_from_directory('results/DistanceOnHyperSphere', single_objective=True)
    # graph.add_results_from_directory('results/DistancesOnHyperCube')
    # graph.add_results_from_directory('results/CEC2021_1_PressureVessel')
    graph.add_results_from_directory('results/CEC2021_19_MultiProductBatchPlant')
    graph.show()

