import plotly.graph_objs as go
import plotly.express as px


class ColorSet:
    feasible = {True: '#007bff', False: '#6c757d'}


class LanguageSet:

    feasible = {'feasible': 'feasible', True: True, False: False}
    non_domi = {'non_domi': 'non_domi', True: True, False: False}

    def __init__(self, language: str = 'ja'):
        self.lang = language
        if self.lang.lower() == 'ja':
            self.feasible = {'feasible': '拘束条件', True: '満足', False: '違反'}
            self.non_domi = {'non_domi': '最適性', True: '非劣解', False: '劣解'}

    def localize(self, df):
        cdf = df.copy()
        cdf[self.feasible['feasible']] = [self.feasible[v] for v in cdf['feasible']]
        cdf[self.non_domi['non_domi']] = [self.non_domi[v] for v in cdf['non_domi']]
        return cdf


ls = LanguageSet('ja')
cs = ColorSet()


def update_hypervolume_plot(_, df):
    df = ls.localize(df)

    # create figure
    fig = px.line(
        df,
        x="trial",
        y="hypervolume",
        markers=True
    )

    fig.update_layout(
        dict(
            title_text="ハイパーボリュームプロット",
        )
    )

    return fig


def update_default_figure(history, df):

    # data setting
    obj_names = history.obj_names

    if len(obj_names) == 0:
        return go.Figure()

    elif len(obj_names) == 1:
        return update_single_objective_plot(history, df)

    elif len(obj_names) >= 2:
        return update_multi_objective_pairplot(history, df)


def update_single_objective_plot(history, df):
    df = ls.localize(df)

    obj_name = history.obj_names[0]

    fig = px.scatter(
        df,
        x='trial',
        y=obj_name,
        color=ls.feasible['feasible'],
        color_discrete_map={
            ls.feasible[True]: cs.feasible[True],
            ls.feasible[False]: cs.feasible[False],
        },
        hover_data={ls.feasible['feasible']: False},
    )

    fig.add_trace(
        go.Scatter(
            x=df['trial'],
            y=df[obj_name],
            mode="lines",
            line=go.scatter.Line(
                color=cs.feasible[False],
                width=0.5,
            ),
            showlegend=False
        )
    )

    fig.update_layout(
        dict(
            title_text="目的プロット",
            xaxis_title="解析実行回数(回)",
            yaxis_title=obj_name,
        )
    )

    return fig


def update_multi_objective_pairplot(history, df):
    df = ls.localize(df)

    obj_names = history.obj_names

    if len(obj_names) == 2:
        fig = px.scatter(
            data_frame=df,
            x=obj_names[0],
            y=obj_names[1],
            color=ls.feasible['feasible'],
            color_discrete_map={
                ls.feasible[True]: cs.feasible[True],
                ls.feasible[False]: cs.feasible[False],
            },
            hover_data={ls.feasible['feasible']: False},
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
            color=ls.feasible['feasible'],
            color_discrete_map={
                ls.feasible[True]: cs.feasible[True],
                ls.feasible[False]: cs.feasible[False],
            },
            hover_data={ls.feasible['feasible']: False},
        )
        fig.update_traces(
            patch={'diagonal.visible': False},
            showupperhalf=False,
        )

    fig.update_layout(
        dict(
            title_text="多目的ペアプロット",
        )
    )

    return fig


if __name__ == '__main__':
    import os
    import pandas as pd


    class History:
        def __init__(self, _df):
            suffix = '_direction'
            self.obj_names = [c[:-len(suffix)] for c in _df.columns if c.endswith(suffix)]


    os.chdir(os.path.dirname(__file__))
    # csv_path = '_sample_history_include_infeasible_3obj.csv'
    # csv_path = '_sample_history_include_infeasible_2obj.csv'
    csv_path = '_sample_history_include_infeasible_1obj.csv'
    _df = pd.read_csv(csv_path, encoding='shift-jis')
    _h = History(_df)

    # _f = update_hypervolume_plot(_h, _df)
    _f = update_default_figure(_h, _df)
    _f.show()
