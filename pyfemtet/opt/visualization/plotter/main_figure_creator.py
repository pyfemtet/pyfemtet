from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

from pyfemtet._util.df_util import *
from pyfemtet.opt.history import *
from pyfemtet.opt.problem.problem import MAIN_FIDELITY_NAME
from pyfemtet._i18n import Msg, _


__all__ = [
    'get_hypervolume_plot',
    'get_default_figure',
    'get_objective_plot',
]

# ===== i18n =====
def _i18n_not_defined_default():
    return _('Not included in the evaluation.',
             '評価対象外')


# optimality
_LOCALIZED_OPTIMALITY = _('Optimality', '最適かどうか')


def _i18n_optimality(value, is_single_objective) -> str:
    if value is True:
        if is_single_objective:
            return _('optimal', '最適解')
        else:
            return _('non dominated', '非劣解')
    elif value is False:
        if is_single_objective:
            return _('not optimal', '最適でない解')
        else:
            return _('dominated', '劣解')
    else:
        return _i18n_not_defined_default()


(_color_mapping_single_objective := defaultdict(lambda: '#888888')).update({
    _i18n_optimality(True, True): "#007bff",
    _i18n_optimality(False, True): "#6c757d",
})
(_color_mapping_multi_objective := defaultdict(lambda: '#888888')).update({
    _i18n_optimality(True, False): "#007bff",
    _i18n_optimality(False, False): "#6c757d",
})


# feasibility
_LOCALIZED_FEASIBILITY = _('Feasibility', '実行可能性')


def _i18n_feasibility(value) -> str:
    if value is True:
        return _('feasible', '実行可能')
    elif value is False:
        return _('infeasible', '実行不可能')
    else:
        return _i18n_not_defined_default()


(_symbol_mapping := defaultdict(lambda: 'square-open')).update({
    _i18n_feasibility(True): 'circle',
    _i18n_feasibility(False): 'circle-open',
})


# common
def _i18n_df(df, is_single_objective):
    # 元の df を変更しないようにコピー
    df = df.copy()

    # feasibility 列のデータに対し、
    # <翻訳された列名>: <グラフに表示するデータ> に
    # 変換した df を作る。
    df[_LOCALIZED_FEASIBILITY] = [_i18n_feasibility(v)
                                 for v in df['feasibility']]

    # optimality も同様
    df[_LOCALIZED_OPTIMALITY] = [_i18n_optimality(v, is_single_objective)
                                 for v in df['optimality']]

    return df


# ===== main =====

def get_hypervolume_plot(_: History, df: pd.DataFrame) -> go.Figure:

    df = _i18n_df(df, is_single_objective=False)

    # メインデータを抽出
    df = get_partial_df(df, equality_filters=MAIN_FILTER)

    # create figure
    fig = px.line(
        df,
        x="trial",
        y="hypervolume",
        markers=True,
        custom_data=['trial'],
    )

    fig.update_layout(
        dict(
            title_text=Msg.GRAPH_TITLE_HYPERVOLUME,
            # transition_duration=1000,  # Causes graph freeze on tab change
            xaxis_title=Msg.GRAPH_AXIS_LABEL_TRIAL,
            yaxis_title='hypervolume',
            xaxis=dict(
                tick0=1,
                type='category',
            )
        )
    )

    return fig


def get_default_figure(history, df) -> go.Figure:

    # data setting
    obj_names = history.obj_names

    fig = go.Figure()

    if len(obj_names) == 1:
        fig = _get_single_objective_plot(history, df)

    elif len(obj_names) >= 2:
        fig = _get_multi_objective_pairplot(history, df)

    fig.update_traces(hoverinfo="none", hovertemplate=None)
    fig.update_layout(
        clickmode='event+select',
        transition_duration=900,  # 画面更新の頻度よりも短くする
    )

    return fig


def _get_single_objective_plot(history: History, df: pd.DataFrame):

    df: pd.DataFrame = _i18n_df(df, is_single_objective=True)
    obj_name = history.obj_names[0]

    # 「成功した試行数」を表示するために
    # obj が na ではない row の連番を作成
    df = (df[~df[obj_name].isna()]).copy()
    SUCCEEDED_TRIAL_COLUMN = 'succeeded_trial'
    assert SUCCEEDED_TRIAL_COLUMN not in df.columns
    df[SUCCEEDED_TRIAL_COLUMN] = range(len(df))
    df[SUCCEEDED_TRIAL_COLUMN] += 1

    df_main = get_partial_df(df, equality_filters=MAIN_FILTER)

    df.columns = [c.replace(' / ', '<BR>/ ') for c in df.columns]
    df_main.columns = [c.replace(' / ', '<BR>/ ') for c in df.columns]
    obj_name = obj_name.replace(' / ', '<BR>/ ')

    sub_fidelity_names = history.sub_fidelity_names
    obj_names_per_sub_fidelity_list: list[list[str]] = [
        history.obj_names for _ in sub_fidelity_names
    ]

    # ===== base figure =====
    fig = go.Figure()

    # ===== all results of sub-fidelity =====
    for (
            sub_fidelity_name,
            obj_names_per_sub_fidelity,
            color
    ) in zip(
            sub_fidelity_names,
            obj_names_per_sub_fidelity_list,
            px.colors.qualitative.G10[::-1]
    ):

        if sub_fidelity_name == MAIN_FIDELITY_NAME:
            continue

        assert len(obj_names_per_sub_fidelity) == 1
        obj_name_per_sub_fidelity = obj_names_per_sub_fidelity[0]
        df_sub = get_partial_df(df, equality_filters=dict(sub_fidelity_name=sub_fidelity_name))
        trace = go.Scatter(
            x=df_sub[SUCCEEDED_TRIAL_COLUMN],
            y=df_sub[obj_name_per_sub_fidelity],
            mode='markers',
            marker=dict(color=color, symbol='square-open'),
            name=sub_fidelity_name,
        )
        fig.add_trace(trace)

    # ===== i 番目が、その時点までで最適かどうか =====
    # NOTE: 最後の direction をその最適化全体の direction と見做す
    # その時点までの最適な点 index
    indices = []
    anti_indices = []
    objectives = df_main[obj_name].values
    direction = df_main[f'{obj_name}_direction'].values[-1]
    for i, obj in enumerate(objectives):

        obj_halfway = objectives[:i+1]
        feasible_obj_halfway = obj_halfway[np.where(~np.isnan(obj_halfway))]

        if len(feasible_obj_halfway) == 0:
            indices.append(i)
            continue

        if direction == 'maximize':
            if obj == max(feasible_obj_halfway):
                indices.append(i)
            else:
                anti_indices.append(i)

        elif direction == 'minimize':
            if obj == min(feasible_obj_halfway):
                indices.append(i)
            else:
                anti_indices.append(i)

        else:
            residuals_halfway = (feasible_obj_halfway - direction) ** 2
            if ((obj - direction) ** 2) == min(residuals_halfway):
                indices.append(i)
            else:
                anti_indices.append(i)

    # ===== すべての点を灰色で打つ =====
    fig.add_trace(
        go.Scatter(
            x=df_main[SUCCEEDED_TRIAL_COLUMN],
            y=df_main[obj_name],
            customdata=df_main[SUCCEEDED_TRIAL_COLUMN].values.reshape((-1, 1)),
            mode="markers",
            marker=dict(color='#6c757d', size=6),
            name=Msg.LEGEND_LABEL_ALL_SOLUTIONS,
        )
    )

    # ===== その時点までの最小の点を青で打つ（上から描く） =====
    fig.add_trace(
        go.Scatter(
            x=df_main[SUCCEEDED_TRIAL_COLUMN].iloc[indices],
            y=df_main[obj_name].iloc[indices],
            mode="markers+lines",
            marker=dict(color='#007bff', size=9),
            name=Msg.LEGEND_LABEL_OPTIMAL_SOLUTIONS,
            line=dict(width=1, color='#6c757d',),
            customdata=df_main['trial'].iloc[indices].values.reshape((-1, 1)),
            legendgroup='optimality',
        )
    )

    # ===== その時点までの最小の点から現在までの平行点線を引く =====
    if len(indices) > 1:
        x = [df_main[SUCCEEDED_TRIAL_COLUMN].iloc[indices].iloc[-1],
             df_main[SUCCEEDED_TRIAL_COLUMN].iloc[-1]]
        y = [df_main[obj_name].iloc[indices].iloc[-1]] * 2
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(width=0.5, color='#6c757d', dash='dash'),
                showlegend=False,
                legendgroup='optimality',
            )
        )

    # ===== direction が float の場合、目標値を描く =====
    if len(df_main) > 1:
        if isinstance(direction, float):
            x = [df_main[SUCCEEDED_TRIAL_COLUMN].iloc[0],
                 df_main[SUCCEEDED_TRIAL_COLUMN].iloc[-1]]
            y = [direction] * 2
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(width=0.5, color='#FF2400', dash='dash'),
                    name=Msg.LEGEND_LABEL_OBJECTIVE_TARGET,
                )
            )

    # ===== layout =====
    fig.update_layout(
        dict(
            title_text=Msg.GRAPH_TITLE_SINGLE_OBJECTIVE,
            xaxis_title=Msg.GRAPH_AXIS_LABEL_TRIAL,
            yaxis_title=obj_name,
            xaxis=dict(
                tick0=1,
                type='category',  # 負の数や小数点を表示しない。ただし hover の callback があるのでオフセットしたり文字を入れたりしないこと
            )
        )
    )

    return fig


def _get_multi_objective_pairplot(history: History, df: pd.DataFrame):

    df = _i18n_df(df, is_single_objective=False)
    df_main = get_partial_df(df, equality_filters=MAIN_FILTER)

    obj_names = history.obj_names

    df.columns = [c.replace(' / ', '<BR>/ ') for c in df.columns]
    df_main.columns = [c.replace(' / ', '<BR>/ ') for c in df.columns]
    obj_names = [o.replace(' / ', '<BR>/ ') for o in obj_names]

    sub_fidelity_names = history.sub_fidelity_names
    obj_names_per_sub_fidelity_list: list[list[str]] = [
        history.obj_names for _ in sub_fidelity_names
    ]

    common_kwargs = dict(
        color=_LOCALIZED_OPTIMALITY,
        color_discrete_map=_color_mapping_multi_objective,
        symbol=_LOCALIZED_FEASIBILITY,
        symbol_map=_symbol_mapping,
        custom_data=['trial'],
        category_orders={
            _LOCALIZED_FEASIBILITY: (
                _i18n_feasibility(None),  # True, False でなければなんでもよい
                _i18n_feasibility(False),
                _i18n_feasibility(True),
            ),
            _LOCALIZED_OPTIMALITY: (
                _i18n_optimality(None, False),
                _i18n_optimality(False, False),
                _i18n_optimality(True, False),
            ),
        },
    )

    if len(obj_names) == 2:

        # plot main data
        fig = px.scatter(
            data_frame=df_main,
            x=obj_names[0],
            y=obj_names[1],
            **common_kwargs,
        )

        # ===== sub-fidelity =====
        for (
                sub_fidelity_name,
                obj_names_per_sub_fidelity,
                color
        ) in zip(
            sub_fidelity_names,
            obj_names_per_sub_fidelity_list,
            px.colors.qualitative.G10[::-1]
        ):

            if sub_fidelity_name == MAIN_FIDELITY_NAME:
                continue

            assert len(obj_names_per_sub_fidelity) == 2
            name1, name2 = obj_names_per_sub_fidelity
            df_sub = get_partial_df(df, equality_filters=dict(sub_fidelity_name=sub_fidelity_name))
            trace = go.Scatter(
                x=df_sub[name1],
                y=df_sub[name2],
                mode='markers',
                marker=dict(color=color, symbol='square-open'),
                name=sub_fidelity_name,
            )
            fig.add_trace(trace)

        fig.update_layout(
            dict(
                xaxis_title=obj_names[0],
                yaxis_title=obj_names[1],
            )
        )

    else:
        # plot main data
        fig = px.scatter_matrix(
            data_frame=df_main,
            dimensions=obj_names,
            **common_kwargs,
        )

        # ===== sub-fidelity =====
        for (
                sub_fidelity_name,
                obj_names_per_sub_fidelity,
                color
        ) in zip(
            sub_fidelity_names,
            obj_names_per_sub_fidelity_list,
            px.colors.qualitative.G10[::-1]
        ):

            if sub_fidelity_name == MAIN_FIDELITY_NAME:
                continue
            df_sub = get_partial_df(df, equality_filters=dict(sub_fidelity_name=sub_fidelity_name))

            fig.add_trace(
                go.Splom(
                    dimensions=[
                        {
                            'label': obj_name,
                            'values': df_sub[sub_obj_name]
                        } for obj_name, sub_obj_name in zip(
                            obj_names, obj_names_per_sub_fidelity
                        )
                    ],
                    marker=dict(color=color, symbol='square-open'),
                    name=sub_fidelity_name,
                )
            )

        fig.update_traces(
            patch={'diagonal.visible': False},
            showupperhalf=False,
        )

    fig.update_layout(
        dict(
            title_text=Msg.GRAPH_TITLE_MULTI_OBJECTIVE,
        )
    )

    return fig


def get_objective_plot(history: History, df: pd.DataFrame, obj_names: list[str]) -> go.Figure:

    df = _i18n_df(df, is_single_objective=False)

    df_main = get_partial_df(df, equality_filters=MAIN_FILTER)

    df.columns = [c.replace(' / ', '<BR>/ ') for c in df.columns]
    df_main.columns = [c.replace(' / ', '<BR>/ ') for c in df.columns]
    obj_names = [o.replace(' / ', '<BR>/ ') for o in obj_names]

    sub_fidelity_names = history.sub_fidelity_names
    obj_names_per_sub_fidelity_list: list[list[str]] = [
        history.obj_names for _ in sub_fidelity_names
    ]

    common_kwargs = dict(
        color=_LOCALIZED_OPTIMALITY,
        color_discrete_map=_color_mapping_multi_objective,
        symbol=_LOCALIZED_FEASIBILITY,
        symbol_map=_symbol_mapping,
        custom_data=['trial'],
        category_orders={
            _LOCALIZED_FEASIBILITY: (
                _i18n_feasibility(None),  # True, False でなければなんでもよい
                _i18n_feasibility(False),
                _i18n_feasibility(True),
            ),
            _LOCALIZED_OPTIMALITY: (
                _i18n_optimality(None, False),
                _i18n_optimality(False, False),
                _i18n_optimality(True, False),
            ),
        },
    )

    if len(obj_names) == 2:

        # main plot
        fig = px.scatter(
            data_frame=df_main,
            x=obj_names[0],
            y=obj_names[1],
            **common_kwargs,
        )

        # ===== sub-fidelity =====
        for sub_fidelity_name, obj_names_per_sub_fidelity, color \
            in zip(sub_fidelity_names,
                   obj_names_per_sub_fidelity_list,
                   px.colors.qualitative.G10[::-1]
                   ):

            if sub_fidelity_name == MAIN_FIDELITY_NAME:
                continue
            df_sub = get_partial_df(df, equality_filters=dict(sub_fidelity_name=sub_fidelity_name))

            sub_name0 = obj_names_per_sub_fidelity[history.obj_names.index(obj_names[0])]
            sub_name1 = obj_names_per_sub_fidelity[history.obj_names.index(obj_names[1])]

            trace = go.Scatter(
                x=df_sub[sub_name0],
                y=df_sub[sub_name1],
                mode='markers',
                marker=dict(color=color, symbol='square-open'),
                name=sub_fidelity_name,
            )
            fig.add_trace(trace)

        fig.update_layout(
            dict(
                xaxis_title=obj_names[0],
                yaxis_title=obj_names[1],
            )
        )

    elif len(obj_names) == 3:

        # main plot
        fig = px.scatter_3d(
            df_main,
            x=obj_names[0],
            y=obj_names[1],
            z=obj_names[2],
            **common_kwargs,
        )

        # ===== sub-fidelity =====
        for sub_fidelity_name, obj_names_per_sub_fidelity, color \
            in zip(sub_fidelity_names,
                   obj_names_per_sub_fidelity_list,
                   px.colors.qualitative.G10[::-1]
                   ):

            if sub_fidelity_name == MAIN_FIDELITY_NAME:
                continue
            df_sub = get_partial_df(df, equality_filters=dict(sub_fidelity_name=sub_fidelity_name))

            sub_name0 = obj_names_per_sub_fidelity[history.obj_names.index(obj_names[0])]
            sub_name1 = obj_names_per_sub_fidelity[history.obj_names.index(obj_names[1])]
            sub_name2 = obj_names_per_sub_fidelity[history.obj_names.index(obj_names[2])]

            fig.add_trace(
                go.Scatter3d(
                    x=df_sub[sub_name0],
                    y=df_sub[sub_name1],
                    z=df_sub[sub_name2],
                    mode='markers',
                    marker=dict(color=color, symbol='square-open'),
                    name=sub_fidelity_name,
                )
            )

        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=30),
            scene=dict(aspectmode="cube")
        )
        fig.update_traces(
            marker=dict(
                size=3,
            ),
        )

    else:
        raise Exception

    fig.update_layout(
        dict(
            title_text="Objective plot",
        )
    )

    fig.update_traces(hoverinfo="none", hovertemplate=None)

    return fig
