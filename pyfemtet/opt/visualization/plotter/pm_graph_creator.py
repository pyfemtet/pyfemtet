import numpy as np
import plotly.graph_objects as go

from pyfemtet._i18n import _
from pyfemtet._util.df_util import *
from pyfemtet.opt.history import *
from pyfemtet.opt.prediction._model import *
from pyfemtet.opt.prediction._helper import *
from pyfemtet.opt.problem.problem import MAIN_FIDELITY_NAME


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
    return _plot(
        history=history,
        prm_name1=prm_name1,
        prm_name2=None,
        params=params,
        obj_name=obj_name,
        df=df,
        pyfemtet_model=pyfemtet_model,
        n=n,
    )


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
    return _plot(
        history=history,
        prm_name1=prm_name1,
        prm_name2=prm_name2,
        params=params,
        obj_name=obj_name,
        df=df,
        pyfemtet_model=pyfemtet_model,
        n=n,
    )


def _plot(
        history: History,
        prm_name1: str,
        prm_name2: str | None,
        params: dict[str, float],
        obj_name: str,
        df,
        pyfemtet_model: PyFemtetModel,
        n,
) -> go.Figure:

    is_3d = prm_name2 is not None

    # prepare input
    if is_3d:
        prm_values1 = get_grid_values(history, df, prm_name1, n)
        prm_values2 = get_grid_values(history, df, prm_name2, n)

        # plot 用の格子点を作成
        xx1, xx2 = np.meshgrid(prm_values1, prm_values2)

        # predict 用のデータを作成
        x1 = xx1.ravel()
        x2 = xx2.ravel()
    else:
        prm_values1 = get_grid_values(history, df, prm_name1, n)
        xx1 = prm_values1
        x1 = xx1
        prm_values2 = None
        xx2 = None
        x2 = None

    # predict 用の入力を作成
    x = np.empty((int(np.prod(xx1.shape)), len(history.prm_names))).astype(object)
    for i, prm_name in enumerate(history.prm_names):
        if prm_name == prm_name1:
            x[:, i] = x1
        elif prm_name == prm_name2:
            assert x2 is not None, 'prm_name2 must be None.'
            x[:, i] = x2
        else:
            x[:, i] = params[prm_name]

    # predict
    z_mean_, z_std_ = pyfemtet_model.predict(x)

    # target objective を抽出
    obj_idx = history.obj_names.index(obj_name)
    z_mean, z_std = z_mean_[:, obj_idx], z_std_[:, obj_idx]

    # 3d 用 grid に変換
    if is_3d:
        zz_mean, zz_std = z_mean.reshape(xx1.shape), z_std.reshape(xx1.shape)
    else:
        zz_mean, zz_std = None, None

    # plot
    fig = go.Figure()

    # rsm
    if is_3d:
        assert prm_name2 is not None
        assert prm_values2 is not None
        assert xx2 is not None
        assert zz_mean is not None
        assert zz_std is not None

        zz_upper = zz_mean + zz_std
        zz_lower = zz_mean - zz_std

        # std
        fig.add_trace(go.Surface(
            z=zz_upper, x=xx1, y=xx2,
            showscale=False, opacity=0.3, showlegend=True,
            name=_('Upper of pred. std-dev')))
        fig.add_trace(go.Surface(
            z=zz_lower, x=xx1, y=xx2,
            showscale=False, opacity=0.3, showlegend=True,
            name=_('Lower of pred. std-dev')))

        # mean
        contours = {}
        for key, prm_name, prm_values in zip(('x', 'y'), (prm_name1, prm_name2), (prm_values1, prm_values2)):
            if history._records.column_manager.is_numerical_parameter(prm_name):
                lb, ub = prm_values.min(), prm_values.max()
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
                name=_('Prediction'),
                showlegend=True,
                contours=contours,
                colorbar=dict(
                    x=0.2,
                    xref="container",
                ),
            )
        )
    else:
        # std
        fig.add_trace(
            go.Scatter(
                x=list(x1) + list(x1)[::-1],
                y=list(z_mean + z_std) + list(z_mean - z_std)[::-1],
                name=_('Std-dev of pred.'),
                fill='toself',
                opacity=0.3,
            )
        )

        # mean
        fig.add_trace(
            go.Scatter(
                x=x1, y=z_mean,
                name=_('Prediction'),
            )
        )

    # scatter
    sub_fidelity_names = history.sub_fidelity_names
    obj_names_per_sub_fidelity_list: list[list[str]] = [
        history.obj_names for __ in sub_fidelity_names
    ]

    if is_3d:

        # main
        df_main = get_partial_df(df, MAIN_FILTER)
        fig.add_trace(go.Scatter3d(
            x=df_main[prm_name1], y=df_main[prm_name2], z=df_main[obj_name],
            mode='markers',
            marker=dict(
                size=3,
                line=dict(
                    width=1,
                    color='white',
                ),
            ),
            name='trial',
        ))

        # sub sampling
        buff = get_partial_df(
            df,
            {'sub_fidelity_name': MAIN_FIDELITY_NAME},
        )
        df_main_sub_sampling = get_partial_df(
            buff,
            {'sub_sampling': float('nan')},
            method='all-exclude',
        )
        fig.add_trace(go.Scatter3d(
            x=df_main_sub_sampling[prm_name1],
            y=df_main_sub_sampling[prm_name2],
            z=df_main_sub_sampling[obj_name],
            mode='markers',
            marker=dict(
                size=3,
                line=dict(
                    width=1,
                    color='gray',
                ),
                symbol='square-open'
            ),
            name='trial (sub sampling)',
        ))

        # sub fidelity
        for (
                sub_fidelity_name,
                obj_names_per_sub_fidelity,
                # color
        ) in zip(
            sub_fidelity_names,
            obj_names_per_sub_fidelity_list,
            # px.colors.qualitative.G10[::-1]
        ):

            if sub_fidelity_name == MAIN_FIDELITY_NAME:
                continue
            df_sub = get_partial_df(df, equality_filters=dict(sub_fidelity_name=sub_fidelity_name))

            fig.add_trace(go.Scatter3d(
                x=df_sub[prm_name1], y=df_sub[prm_name2], z=df_sub[obj_name],
                mode='markers',
                marker=dict(
                    size=3,
                    line=dict(
                        width=1,
                        color='green',
                    ),
                    symbol='square-open',  # TODO: sequence 化
                ),
                name=f'trial of {sub_fidelity_name}',
            ))

    else:
        # main
        df_main = get_partial_df(df, MAIN_FILTER)
        fig.add_trace(go.Scatter(
            x=df_main[prm_name1], y=df_main[obj_name],
            mode='markers',
            marker=dict(
                line=dict(
                    width=1,
                    color='white',
                ),
            ),
            name='trial',
        ))

        # sub fidelity
        for (
                sub_fidelity_name,
                obj_names_per_sub_fidelity,
                # color
        ) in zip(
            sub_fidelity_names,
            obj_names_per_sub_fidelity_list,
            # px.colors.qualitative.G10[::-1]
        ):

            if sub_fidelity_name == MAIN_FILTER:
                continue
            df_sub = get_partial_df(df, equality_filters=dict(sub_fidelity_name=sub_fidelity_name))

            fig.add_trace(go.Scatter(
                x=df_sub[prm_name1], y=df_sub[obj_name],
                mode='markers',
                marker=dict(
                    line=dict(
                        width=1,
                        color='white',
                    ),
                    symbol='square-open',  # TODO: sequence 化
                ),
                name=f'trial of {sub_fidelity_name}',
            ))

    # set opacity by its distance
    params_ = params.copy()
    params_.pop(prm_name1)
    if is_3d:
        params_.pop(prm_name2)
    if len(params_) == 0:
        opacity = np.ones(len(df))
    else:
        # distance を計算する用のデータを分割
        prm_names_for_distances = []
        prm_values_for_distances = []
        prm_names_for_categorical = []
        prm_values_for_categorical = []
        for prm_name in params_.keys():
            if history.is_numerical_parameter(prm_name):
                prm_names_for_distances.append(prm_name)
                prm_values_for_distances.append(params_[prm_name])
            elif history.is_categorical_parameter(prm_name):
                prm_names_for_categorical.append(prm_name)
                prm_values_for_categorical.append(params_[prm_name])
            else:
                raise NotImplementedError

        # distance が大きい程 opacity を小さくする
        if len(prm_names_for_distances) > 0:
            target_points = df[prm_names_for_distances]
            hyper_plane = np.array(prm_values_for_distances)
            distances_to_hyper_plane = np.linalg.norm(target_points - hyper_plane, axis=1, keepdims=False)
            if distances_to_hyper_plane.max() == 0:
                opacity = np.ones_like(distances_to_hyper_plane)
            else:
                opacity = 1 - (distances_to_hyper_plane / distances_to_hyper_plane.max())
        else:
            opacity = np.ones(len(df))

        # categorical データが一致しないぶんだけ opacity を 1/N する
        target_points = df[prm_names_for_categorical].values
        hyper_plane = np.array(prm_values_for_categorical)
        # noinspection PyUnresolvedReferences
        count = (target_points == hyper_plane).astype(float).sum(axis=1)
        count = count + 1  # 0 になると困る
        # noinspection PyUnusedLocal
        opacity = opacity * (count / count.max())

    def set_opacity(trace):
        if isinstance(trace, go.Scatter3d) or isinstance(trace, go.Scatter):
            trace.marker.color = [f'rgba(0, 0, 0, {o: .2f})' for o in opacity]

    fig.for_each_trace(set_opacity)

    # axis name
    if is_3d:
        # layout
        fig.update_layout(
            title=_('Prediction Model'),
            scene=dict(
                xaxis=dict(title=dict(text=prm_name1)),
                yaxis=dict(title=dict(text=prm_name2)),
                zaxis=dict(title=dict(text=obj_name)),
            ),
        )

    else:
        fig.update_layout(
            title=_('Prediction Model'),
            xaxis=dict(title=dict(text=prm_name1)),
            yaxis=dict(title=dict(text=obj_name)),
            margin=dict(l=0, r=0, b=0, t=30),
        )

    return fig
