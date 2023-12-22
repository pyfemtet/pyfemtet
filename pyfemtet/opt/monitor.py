import webbrowser
import logging
from dash import Dash, html, dcc
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objs as go


def update_scatter_matrix(femopt):
    # data setting
    data = femopt.history.data
    obj_names = femopt.history.obj_names

    # create figure
    fig = go.Figure()
    fig.update_layout(
        dict(
            width=800,
            height=600,
        )
    )

    # graphs setting dependent on n_objectives
    if len(obj_names) == 0:
        return fig

    elif len(obj_names) == 1:
        fig.add_trace(
            go.Scatter(
                x=tuple(range(len(obj_names[0]))),
                y=data[obj_names[0]],
                mode='markers+lines',
            )
        )
        fig.update_layout(
            dict(
                title_text="単目的ペアプロット",
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


class Monitor(object):

    def __init__(self, femopt):

        self.femopt = femopt

        log_path = self.femopt.history.path.replace('.csv', '.uilog')
        l = logging.getLogger()
        l.addHandler(logging.FileHandler(log_path))

        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


        #### simple sidebar app
        # https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar/

        # the style arguments for the sidebar. We use position:fixed and a fixed width
        SIDEBAR_STYLE = {
            "position": "fixed",
            "top": 0,
            "left": 0,
            "bottom": 0,
            "width": "16rem",
            "padding": "2rem 1rem",
            "background-color": "#f8f9fa",
        }

        # the styles for the main content position it to the right of the sidebar and
        # add some padding.
        CONTENT_STYLE = {
            "margin-left": "18rem",
            "margin-right": "2rem",
            "padding": "2rem 1rem",
        }

        # setup sidebar
        sidebar = html.Div(
            [
                html.H2("Sidebar", className="display-4"),
                html.Hr(),
                html.P(
                    "最適化の進捗の可視化手法を選択できます.", className="lead"
                ),
                dbc.Nav(
                    [
                        dbc.NavLink("Home", href="/", active="exact"),
                        dbc.NavLink("ペアプロット", href="/page-1", active="exact"),
                        # dbc.NavLink("Page 2", href="/page-2", active="exact"),
                    ],
                    vertical=True,
                    pills=True,
                ),
            ],
            style=SIDEBAR_STYLE,
        )

        content = html.Div(id="page-content", style=CONTENT_STYLE)
        self.app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

        #### settings for multiobjective pairplot
        self.home = self.setup_home()
        self.multi_pairplot_layout = self.setup_page1()


        # sidebar によるページ遷移のための callback
        @self.app.callback(Output("page-content", "children"), [Input("url", "pathname")])
        def render_page_content(pathname):
            if pathname == "/":
                return self.home
            elif pathname == "/page-1":
                return self.multi_pairplot_layout
            # elif pathname == "/page-2":
            #     return html.P("Oh cool, this is page 2!")
            # If the user tries to reach a different page, return a 404 message
            return html.Div(
                [
                    html.H1("404: Not found", className="text-danger"),
                    html.Hr(),
                    html.P(f"The pathname {pathname} was not recognised..."),
                ],
                className="p-3 bg-light rounded-3",
            )


        # 中断の設定
        @self.app.callback(
            Output('dummy', 'value'),
            Input('interrupt-button', 'n_clicks'))
        def interrupt(_):
            if _ is not None:
                self.femopt.ipv.set_state('interrupted')
            return ''

        # scatter matrix
        @self.app.callback(
            Output('scatter-matrix-graph', 'figure'),
            Input('interval-component', 'n_intervals'))
        def update_sm(_):
            return update_scatter_matrix(self.femopt)

        # 終了していたら更新をやめる、中断ボタンを disable にする
        @self.app.callback(
            [Output('interval-component', 'max_intervals'),
             Output('interrupt-button', 'disabled'),],
            [Input('interval-component', 'n_intervals'),])
        def stop_interval(_):
            try:
                state = self.femopt.ipv.get_state()
                should_stop = (state == 'interrupted') or (state == 'terminated')
            except AttributeError:
                should_stop = True
            if should_stop:
                max_intervals = 0
                button_disable = True
            else:
                max_intervals = -1
                button_disable = False
            return max_intervals, button_disable

    def setup_home(self):
        # components の設定
        text = dcc.Markdown('''
# 最適化の進行状況モニター
---
#### 左のサイドバーから、可視化方法を選択してください
- このページでは、最適化の進捗状況を見ることができます。
- ブラウザによる進捗状況確認機能ですが、インターネット通信は行いません。
- このページを閉じても最適化は進行します。再びこのページを開くには、ブラウザのアドレスバーに __localhost:8080__ と入力してください。
        ''')

        return text


    def setup_page1(self):
        # components の設定
        # https://dash-bootstrap-components.opensource.faculty.ai/docs/components/accordion/
        dummy = html.Div('', id='dummy')
        interval = dcc.Interval(
            id='interval-component',
            interval=1*1000,  # in milliseconds
            n_intervals=0,
        )
        header = html.H1("最適化の進行状況"),
        graph = dcc.Graph(id='scatter-matrix-graph')
        interrupt_button = dbc.Button('最適化を中断', id='interrupt-button', color='danger')

        # layout の設定
        layout = dbc.Container([
            dbc.Row([dbc.Col(dummy), dbc.Col(interval)]),
            dbc.Row([dbc.Col(header)]),
            dbc.Row([dbc.Col(graph)]),
            dbc.Row([dbc.Col(interrupt_button)], justify="center",),
        ])
        return layout


    def start_server(self, host='localhost', port=8080):

        if host is None:
            host = 'localhost'
        if port is None:
            port = 8080

        if host == '0.0.0.0':
            webbrowser.open(f'http://localhost:{str(port)}')
        else:
            webbrowser.open(f'http://{host}:{str(port)}')
        self.app.run(debug=False, host=host, port=port)


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    class IPV:
        def __init__(self):
            self.state = 'running'
        def get_state(self):
            return self.state
        def set_state(self, state):
            self.state = state
    class History:
        def __init__(self):
            self.obj_names = 'A B C D E'.split()
            self.data = pd.DataFrame(
                np.random.rand(5, len(self.obj_names)),
                columns=self.obj_names,
            )
            self.path = 'tmp.csv'
    class FEMOPT:
        def __init__(self, history, ipv):
            self.history = history
            self.ipv = ipv

    history = History()
    ipv = IPV()
    femopt = FEMOPT(history, ipv)
    monitor = Monitor(femopt)
    monitor.start_server()
