import plotly.graph_objs as go
import plotly.express as px


def update_hypervolume_plot(history, df):
    # create figure
    fig = px.line(df, x="trial", y="hypervolume", markers=True)

    return fig


def update_scatter_matrix(history, data):
    # data setting
    obj_names = history.obj_names

    # create figure
    fig = go.Figure()

    # graphs setting dependent on n_objectives
    if len(obj_names) == 0:
        return fig

    elif len(obj_names) == 1:
        fig.add_trace(
            go.Scatter(
                x=data['trial'],
                y=data[obj_names[0]].values,
                mode='markers+lines',
            )
        )
        fig.update_layout(
            dict(
                title_text="単目的プロット",
                xaxis_title="解析実行回数(回)",
                yaxis_title=obj_names[0],
            )
        )

    elif len(obj_names) == 2:
        fig.add_trace(
            go.Scatter(
                x=data[obj_names[0]],
                y=data[obj_names[1]],
                mode='markers',
            )
        )
        fig.update_layout(
            dict(
                title_text="多目的ペアプロット",
                xaxis_title=obj_names[0],
                yaxis_title=obj_names[1],
            )
        )

    elif len(obj_names) >= 3:
        fig.add_trace(
            go.Splom(
                dimensions=[dict(label=c, values=data[c]) for c in obj_names],
                diagonal_visible=False,
                showupperhalf=False,
            )
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
    csv_path = '_sample_history.csv'
    df = pd.read_csv(csv_path, encoding='shift-jis')
    h = History(df)

    # fig = update_hypervolume_plot(h, df)
    fig = update_scatter_matrix(h, df)
    fig.show()
