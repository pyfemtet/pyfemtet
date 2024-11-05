import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

from pyfemtet.opt._femopt_core import History
from pyfemtet._message import Msg


class _ColorSet:
    non_domi = {True: '#007bff', False: '#6c757d'}  # color


class _SymbolSet:
    feasible = {True: 'circle', False: 'circle-open'}  # style


class _LanguageSet:

    feasible = {'label': 'feasible', True: True, False: False}
    non_domi = {'label': 'non_domi', True: True, False: False}

    def __init__(self):
        self.feasible = {
            'label': Msg.LEGEND_LABEL_CONSTRAINT,
            True: Msg.LEGEND_LABEL_FEASIBLE,
            False: Msg.LEGEND_LABEL_INFEASIBLE,
        }
        self.non_domi = {
            'label': Msg.LEGEND_LABEL_OPTIMAL,
            True: Msg.LEGEND_LABEL_NON_DOMI,
            False: Msg.LEGEND_LABEL_DOMI,
        }

    def localize(self, df):
        # 元のオブジェクトを変更しないようにコピー
        cdf = df.copy()

        # feasible, non_domi の localize
        cdf[self.feasible['label']] = [self.feasible[v] for v in cdf['feasible']]
        cdf[self.non_domi['label']] = [self.non_domi[v] for v in cdf['non_domi']]

        return cdf


_ls = _LanguageSet()
_cs = _ColorSet()
_ss = _SymbolSet()


def get_hypervolume_plot(_: History, df):
    df = _ls.localize(df)

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
        )
    )

    return fig


def get_default_figure(history, df):
    # df = history.local_data  # monitor process and history process is different workers, so history.local_data is not updated in monitor process.
    # df = history.actor_data.copy()  # access to actor from flask callback makes termination unstable.

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
        transition_duration=1000,
    )

    return fig


def _get_single_objective_plot(history, df):

    df = _ls.localize(df)
    obj_name = history.obj_names[0]

    df.columns = [c.replace(' / ', '<BR>/ ') for c in df.columns]
    obj_name = obj_name.replace(' / ', '<BR>/ ')

    fig = px.scatter(
        df,
        x='trial',
        y=obj_name,
        symbol=_ls.feasible['label'],
        symbol_map={
            _ls.feasible[True]: _ss.feasible[True],
            _ls.feasible[False]: _ss.feasible[False],
        },
        hover_data={
            _ls.feasible['label']: False,
            'trial': True,
        },
        custom_data=['trial'],
    )

    fig.add_trace(
        go.Scatter(
            x=df['trial'],
            y=df[obj_name],
            mode="lines",
            line=go.scatter.Line(
                width=0.5,
                color='#6c757d',
            ),
            showlegend=False
        )
    )

    fig.update_layout(
        dict(
            title_text=Msg.GRAPH_TITLE_SINGLE_OBJECTIVE,
            xaxis_title=Msg.GRAPH_AXIS_LABEL_TRIAL,
            yaxis_title=obj_name,
        )
    )

    return fig


def _get_multi_objective_pairplot(history, df):
    df = _ls.localize(df)

    obj_names = history.obj_names

    df.columns = [c.replace(' / ', '<BR>/ ') for c in df.columns]
    obj_names = [o.replace(' / ', '<BR>/ ') for o in obj_names]

    common_kwargs = dict(
        color=_ls.non_domi['label'],
        color_discrete_map={
            _ls.non_domi[True]: _cs.non_domi[True],
            _ls.non_domi[False]: _cs.non_domi[False],
        },
        symbol=_ls.feasible['label'],
        symbol_map={
            _ls.feasible[True]: _ss.feasible[True],
            _ls.feasible[False]: _ss.feasible[False],
        },
        custom_data=['trial'],
        category_orders={
            _ls.feasible['label']: (_ls.feasible[False], _ls.feasible[True]),
            _ls.non_domi['label']: (_ls.non_domi[False], _ls.non_domi[True]),
        },
    )

    if len(obj_names) == 2:
        fig = px.scatter(
            data_frame=df,
            x=obj_names[0],
            y=obj_names[1],
            **common_kwargs,
        )
        fig.update_layout(
            dict(
                xaxis_title=obj_names[0],
                yaxis_title=obj_names[1],
            )
        )

    else:
        fig = px.scatter_matrix(
            data_frame=df,
            dimensions=obj_names,
            **common_kwargs,
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
    df = _ls.localize(df)
    df.columns = [c.replace(' / ', '<BR>/ ') for c in df.columns]
    obj_names = [o.replace(' / ', '<BR>/ ') for o in obj_names]

    common_kwargs = dict(
        color=_ls.non_domi['label'],
        color_discrete_map={
            _ls.non_domi[True]: _cs.non_domi[True],
            _ls.non_domi[False]: _cs.non_domi[False],
        },
        symbol=_ls.feasible['label'],
        symbol_map={
            _ls.feasible[True]: _ss.feasible[True],
            _ls.feasible[False]: _ss.feasible[False],
        },
        custom_data=['trial'],
        category_orders={
            _ls.feasible['label']: (_ls.feasible[False], _ls.feasible[True]),
            _ls.non_domi['label']: (_ls.non_domi[False], _ls.non_domi[True]),
        },
    )

    if len(obj_names) == 2:
        fig = px.scatter(
            data_frame=df,
            x=obj_names[0],
            y=obj_names[1],
            **common_kwargs,
        )
        fig.update_layout(
            dict(
                xaxis_title=obj_names[0],
                yaxis_title=obj_names[1],
            )
        )

    elif len(obj_names) == 3:
        fig = px.scatter_3d(
            df,
            x=obj_names[0],
            y=obj_names[1],
            z=obj_names[2],
            **common_kwargs,
        )
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=30),
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
