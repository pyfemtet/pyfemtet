import numpy as np
import plotly.graph_objects as go

from v1.history import *
from v1.prediction.model import *
from v1.prediction.helper import *


def get_grid_values(df, prm_name_, N) -> np.ndarray:
    if Record.is_numerical_parameter(prm_name_):
        lb_, ub_ = get_bounds_containing_entire_bounds(df, prm_name_)
        out = np.linspace(lb_, ub_, N)
    elif Record.is_categorical_parameter(prm_name_):
        choices = get_choices_containing_entire_bounds(df, prm_name_)
        out = np.array(list(choices))
    else:
        raise NotImplementedError
    return out


def plot3d(
        history: History,
        prm_name1,
        prm_name2,
        remaining_values: dict[str, float],
        obj_name: str,
        df,
        pyfemtet_model: PyFemtetModel,
        N=20,
) -> go.Figure | None:

    # prm_name1, prm_name2 であれば Sequence を作成する
    prm_values1 = get_grid_values(df, prm_name1, N)
    prm_values2 = get_grid_values(df, prm_name2, N)

    # plot 用の格子点を作成
    xx1, xx2 = np.meshgrid(prm_values1, prm_values2)

    # predict 用の入力を作成
    x = np.empty((len(prm_values1) * len(prm_values2), len(history.prm_names))).astype(object)
    x1, x2 = xx1.ravel(), xx2.ravel()
    for i, prm_name in enumerate(history.prm_names):
        if prm_name == prm_name1:
            x[:, i] = x1
        elif prm_name == prm_name2:
            x[:, i] = x2
        else:
            x[:, i] = remaining_values[prm_name]

    # predict
    z_mean_, z_std_ = pyfemtet_model.predict(x)

    # target objective を抽出
    obj_idx = history.obj_names.index(obj_name)
    z_mean, z_std = z_mean_[:, obj_idx], z_std_[:, obj_idx]

    # 3d 用 grid に変換
    zz_mean, zz_std = z_mean.reshape(xx1.shape), z_std.reshape(xx1.shape)

    # plot
    fig = go.Figure()

    # mean surface
    contours = {}
    for key, prm_name in zip(('x', 'y'), (prm_name1, prm_name2)):
        if Record.is_numerical_parameter(prm_name):
            lb, ub = prm_values1.min(), prm_values1.max()
            contours.update({key: dict(
                highlight=True, show=True, color='blue',
                start=lb, end=ub, size=(ub - lb) / N,
            )})
        elif Record.is_categorical_parameter(prm_name):
            contours.update({key: dict(
                highlight=True, show=True, color='blue',
            )})
    fig.add_trace(
        go.Surface(
            x=xx1, y=xx2, z=zz_mean,
            contours=contours,
            showlegend=True,
            colorbar=dict(
                x=0.2,
                xref="container",
            )
        )
    )

    return fig
