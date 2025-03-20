import numpy as np
import plotly.graph_objects as go

from v1.history import *
from v1.prediction.model import *
from v1.prediction.helper import *
from v1.problem import MAIN_FIDELITY_NAME


__all__ = [
    'plot2d',
    'plot3d',
]


def get_grid_values(history: History, df, prm_name_, n) -> np.ndarray:
    if history._records.column_manager.is_numerical_parameter(prm_name_):
        lb_, ub_ = get_bounds_containing_entire_bounds(df, prm_name_)
        out = np.linspace(lb_, ub_, n)
    elif history._records.column_manager.is_categorical_parameter(prm_name_):
        choices = get_choices_containing_entire_bounds(df, prm_name_)
        out = np.array(list(choices))
    else:
        raise NotImplementedError
    return out


def plot2d(
        history: History,
        prm_name1,
        params: dict[str, float],
        obj_name: str,
        df,
        pyfemtet_model: PyFemtetModel,
        n=200,
) -> go.Figure:
    # prm_name1, prm_name2 であれば Sequence を作成する
    x1 = get_grid_values(history, df, prm_name1, n)

    # predict 用の入力を作成
    x = np.empty((len(x1), len(history.prm_names))).astype(object)
    for i, prm_name in enumerate(history.prm_names):
        if prm_name == prm_name1:
            x[:, i] = x1
        else:
            x[:, i] = params[prm_name]

    # predict
    z_mean_, z_std_ = pyfemtet_model.predict(x)

    # target objective を抽出
    obj_idx = history.obj_names.index(obj_name)
    z_mean, z_std = z_mean_[:, obj_idx], z_std_[:, obj_idx]

    # plot
    fig = go.Figure()

    # std
    fig.add_trace(
        go.Scatter(
            x=list(x1) + list(x1)[::-1],
            y=list(z_mean + z_std) + list(z_mean - z_std)[::-1],
            fill='toself',
            opacity=0.3,
            name='予測の信頼性（標準偏差）',
        )
    )

    # mean
    fig.add_trace(
        go.Scatter(
            x=x1,
            y=z_mean,
            name='予測',
        )
    )

    # scatter
    fig.add_trace(
        go.Scatter(
            x=df[prm_name1], y=df[obj_name],
            mode='markers',
            marker=dict(
                size=3,
                color='black',
            ),
            name='trial',
        )
    )

    # set opacity by its distance
    params_ = params.copy()
    params_.pop(prm_name1)
    if len(params_) == 0:
        opacity = np.ones(len(df))
    else:
        print(df.columns)
        print(df.columns)
        target_points = df[list(params_.keys())]
        hyper_plane = np.array(tuple(params_.values()))
        distances_to_hyper_plane = np.linalg.norm(target_points - hyper_plane, axis=1, keepdims=False)
        opacity = 1 - (distances_to_hyper_plane / distances_to_hyper_plane.max())

    def set_opacity(trace):
        if isinstance(trace, go.Scatter):
            trace.marker.color = [f'rgba(0, 0, 0, {o: .2f})' for o in opacity]

    fig.for_each_trace(set_opacity)

    return fig


def plot3d(
        history: History,
        prm_name1,
        prm_name2,
        params: dict[str, float],
        obj_name: str,
        df,
        pyfemtet_model: PyFemtetModel,
        n=20,
) -> go.Figure:

    # prm_name1, prm_name2 であれば Sequence を作成する
    prm_values1 = get_grid_values(history, df, prm_name1, n)
    prm_values2 = get_grid_values(history, df, prm_name2, n)

    # plot 用の格子点を作成
    xx1, xx2 = np.meshgrid(prm_values1, prm_values2)

    # predict 用の入力を作成
    x = np.empty((int(np.prod(xx1.shape)), len(history.prm_names))).astype(object)
    x1, x2 = xx1.ravel(), xx2.ravel()
    for i, prm_name in enumerate(history.prm_names):
        if prm_name == prm_name1:
            x[:, i] = x1
        elif prm_name == prm_name2:
            x[:, i] = x2
        else:
            x[:, i] = params[prm_name]

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
        if history._records.column_manager.is_numerical_parameter(prm_name):
            lb, ub = prm_values1.min(), prm_values1.max()
            contours.update({key: dict(
                highlight=True, show=True, color='blue',
                start=lb, end=ub, size=(ub - lb) / n,
            )})
        elif history._records.column_manager.is_categorical_parameter(prm_name):
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

    # scatter
    fig.add_trace(
        go.Scatter3d(
            x=df[prm_name1], y=df[prm_name2], z=df[obj_name],
            mode='markers',
            marker=dict(
                size=3,
                color='black',
            ),
            name='trial',
        )
    )

    # set opacity by its distance
    params_ = params.copy()
    params_.pop(prm_name1)
    params_.pop(prm_name2)
    if len(params_) == 0:
        opacity = np.ones(len(df))
    else:
        target_points = df[tuple(params_.keys())]
        hyper_plane = np.array(tuple(params_.values()))
        distances_to_hyper_plane = np.linalg.norm(target_points - hyper_plane, axis=1, keepdims=False)
        opacity = 1 - (distances_to_hyper_plane / distances_to_hyper_plane.max())

    def set_opacity(trace):
        if isinstance(trace, go.Scatter3d):
            trace.marker.color = [f'rgba(0, 0, 0, {o: .2f})' for o in opacity]

    fig.for_each_trace(set_opacity)

    return fig
