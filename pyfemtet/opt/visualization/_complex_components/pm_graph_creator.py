from typing import Callable, List

import torch
import numpy as np
import pandas as pd

import plotly.graph_objs as go
import plotly.express as px

from pyfemtet.opt._femopt_core import History

from pyfemtet.opt.prediction._base import PyFemtetPredictionModel
from pyfemtet.opt.prediction.single_task_gp import SingleTaskGPModel
from pyfemtet._message import Msg


class PredictionModelCreator:

    # noinspection PyAttributeOutsideInit
    def fit(self, history, df):
        # common
        self.history = history  # used to check fit or not
        self.df = df
        # method specific
        self.pmm = PyFemtetPredictionModel(history, df, SingleTaskGPModel)
        self.pmm.fit()

    def create_figure(
            self,
            prm_name_1: str,
            obj_name: str,
            remaining_x: list[float],
            prm_name_2: str = '',
    ):

        history = self.history
        df = self.df

        # determine axis1, axis2 range
        lb1 = df[f'{prm_name_1}_lower_bound'].min()
        ub1 = df[f'{prm_name_1}_upper_bound'].max()
        lb1 = df[prm_name_1].min() if np.isnan(lb1) else lb1
        ub1 = df[prm_name_1].max() if np.isnan(ub1) else ub1
        if prm_name_2:
            lb2 = df[f'{prm_name_2}_lower_bound'].min()
            ub2 = df[f'{prm_name_2}_upper_bound'].max()
            lb2 = df[prm_name_2].min() if np.isnan(lb2) else lb2
            ub2 = df[prm_name_2].max() if np.isnan(ub2) else ub2

        # create grid data
        if prm_name_2:
            N = 20
            x_grid = np.linspace(lb1, ub1, N)  # shape: (N,)
            y_grid = np.linspace(lb2, ub2, N)
            xx, yy = np.meshgrid(x_grid, y_grid)  # shape: (N, N)
        else:
            N = 100
            xx = np.linspace(lb1, ub1, N)  # shape: (N,)

        # create raveled grid data
        tmp_df = pd.DataFrame(
            columns=history.prm_names,
            index=range(len(xx.ravel())),  # index で df の長さを指定しないと remaining_x の代入で工夫が必要になる
        )

        # 変数名ごとに見ていって、
        # 横軸に使う変数ならグリッド値を入れ、
        # そうでなければ remaining_x の値を入れる（シリーズになる）
        idx = 0
        for prm_name in history.prm_names:
            if prm_name == prm_name_1:
                tmp_df[prm_name_1] = xx.ravel()  # shape: (N**2,) or (N,)
            elif prm_name == prm_name_2:  # no chance to hit if prm_name_2 == ''
                tmp_df[prm_name_2] = yy.ravel()
            else:
                tmp_df[prm_name] = remaining_x[idx]
                idx += 1
        x = tmp_df.values  # shape: (len(prm), N**2)

        # predict by model
        mean, std = self.pmm.predict(x)  # shape: (len(obj), N**2)

        # create mesh data for graph
        obj_index = history.obj_names.index(obj_name)
        zz_mean = mean[:, obj_index].reshape(xx.shape)  # shape: (N**2,) -> (N, N) or (N,) -> (N,)
        zz_std = std[:, obj_index].reshape(xx.shape)
        zz_upper = zz_mean + zz_std
        zz_lower = zz_mean - zz_std

        # ===== create figure =====
        # description: データ点は多次元空間上の点なので、余剰次元を無視して三次元空間上に投影しても何の意味もない。むしろ混乱するのでやってはいけない。実験的機能として、透明度を設定してみる。
        # calculate distance from super-plane to prm point for opacity of scatter plot
        remaining_prm = list(history.prm_names)  # weak copy
        remaining_prm.remove(prm_name_1)
        remaining_prm.remove(prm_name_2) if prm_name_2 in remaining_prm else None
        if len(remaining_prm) > 0:
            super_plane = np.array(remaining_x)  # shape: (len(prm_names)-2) or (len(prm_names)-1)
            target_point = df[remaining_prm].values  # shape: (len(df), len(prm_names)-2) or (len(df), len(prm_names)-1)
            distance = np.linalg.norm(target_point-super_plane, axis=1, keepdims=False)  # shape: (len(df))
            opacity = 1 - (distance / distance.max())  # smaller distance, larger opacity
        else:
            opacity = np.ones(len(df))

        # scatter plot
        if prm_name_2:
            fig = go.Figure(data=go.Scatter3d(
                x=df[prm_name_1], y=df[prm_name_2], z=df[obj_name],
                mode='markers',
                marker=dict(
                    size=3,
                    color='black',
                ),
                name='trial',
            ))
        else:
            fig = go.Figure(data=go.Scatter(
                x=df[prm_name_1], y=df[obj_name],
                mode='markers',
                marker=dict(
                    color='black',
                ),
                name='trial',
            ))

        # set opacity by its distance
        def set_opacity(trace):
            trace.marker.color = [f'rgba(0, 0, 0, {o: .2f})' for o in opacity]
        fig.for_each_trace(set_opacity)

        # main RSM
        if prm_name_2:
            contours = dict(
                x=dict(
                    highlight=True, show=True, color='blue',
                    start=lb1, end=ub1, size=(ub1-lb1)/N,
                ),
                y=dict(
                    highlight=True, show=True, color='blue',
                    start=lb2, end=ub2, size=(ub2-lb2)/N
                ),
                z=dict(highlight=False, show=False),
            )
            fig.add_trace(
                go.Surface(
                    z=zz_mean, x=xx, y=yy,
                    contours=contours,
                    showlegend=True,
                    name=Msg.LEGEND_LABEL_PREDICTION_MODEL,
                    colorbar=dict(
                        x=0.2,
                        xref="container",
                        # orientation='h',
                    )
                )
            )
            # std
            fig.add_trace(go.Surface(z=zz_upper, x=xx, y=yy, showscale=False, opacity=0.3, showlegend=True, name=Msg.LEGEND_LABEL_PREDICTION_MODEL_STDDEV))
            fig.add_trace(go.Surface(z=zz_lower, x=xx, y=yy, showscale=False, opacity=0.3, showlegend=True, name=Msg.LEGEND_LABEL_PREDICTION_MODEL_STDDEV))

            # layout
            fig.update_layout(
                title=Msg.GRAPH_TITLE_PREDICTION_MODEL,
                scene=dict(
                    xaxis_title=prm_name_1,
                    yaxis_title=prm_name_2,
                    zaxis_title=obj_name
                ),
                margin=dict(l=0, r=0, b=0, t=30),
            )

        else:
            fig.add_trace(
                go.Scatter(x=xx, y=zz_mean, name=Msg.LEGEND_LABEL_PREDICTION_MODEL)
            )
            # std
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([xx, xx[::-1]]),
                    y=np.concatenate([zz_upper, zz_lower[::-1]]),
                    opacity=0.3,
                    fill='toself',
                    name=Msg.LEGEND_LABEL_PREDICTION_MODEL_STDDEV,
                )
            )

            # layout
            fig.update_layout(
                title=Msg.GRAPH_TITLE_PREDICTION_MODEL,
                xaxis_title=prm_name_1,
                yaxis_title=obj_name,
            )

        return fig
