import webbrowser
import logging
from dash import Dash, html, dcc, ctx, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px


def update_hypervolume_plot(femopt):
    # data setting
    df = femopt.history.data

    # create figure
    fig = px.line(df, x="trial", y="hypervolume", markers=True)

    return fig



def update_scatter_matrix(femopt):
    # data setting
    data = femopt.history.data
    obj_names = femopt.history.obj_names

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


def setup_home():
    # components の設定
    # https://dash-bootstrap-components.opensource.faculty.ai/docs/components/accordion/
    dummy = html.Div('', id='dummy')
    interval = dcc.Interval(
        id='interval-component',
        interval=1*1000,  # in milliseconds
        n_intervals=0,
    )
    header = html.H1("最適化の進行状況"),
    graphs = dbc.Card(
        [
            dbc.CardHeader(
                dbc.Tabs(
                    [
                        dbc.Tab(label="目的プロット", tab_id="tab-1"),
                        dbc.Tab(label="Hypervolume", tab_id="tab-2"),
                    ],
                    id="card-tabs",
                    active_tab="tab-1",
                )
            ),
            dbc.CardBody(html.P(id="card-content", className="card-text")),
        ]
    )
    toggle_update_button = dbc.Button('グラフの自動更新の一時停止', id='toggle-update-button')
    interrupt_button = dbc.Button('最適化を中断', id='interrupt-button', color='danger')
    status_text = dcc.Markdown(f'''
---
- このページでは、最適化の進捗状況を見ることができます。
- このページを閉じても最適化は進行します。
- この機能はブラウザによる状況確認機能ですが、インターネット通信は行いません。
- 再びこのページを開くには、ブラウザのアドレスバーに __localhost:8080__ と入力してください。
- ※ 特定のホスト名及びポートを指定するには、OptimizerBase.main() の実行前に
OptimizerBase.set_monitor_server() を実行してください。
    ''')

    # layout の設定
    layout = dbc.Container([
        dbc.Row([dbc.Col(dummy), dbc.Col(interval)]),
        dbc.Row([dbc.Col(header)]),
        dbc.Row([dbc.Col(graphs)]),
        dbc.Row([dbc.Col(toggle_update_button), dbc.Col(interrupt_button)]),
        dbc.Row([dbc.Col(status_text)]),
    ], fluid=True)

    return layout


class Monitor(object):

    def __init__(self, femopt):

        # 引数の処理
        self.femopt = femopt

        # ログファイルの保存場所
        log_path = self.femopt.history.path.replace('.csv', '.uilog')
        l = logging.getLogger()
        l.addHandler(logging.FileHandler(log_path))

        # app の立上げ
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # ページの components と layout の設定
        self.home = setup_home()

        # setup sidebar
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
        sidebar = html.Div(
            [
                html.H2("PyFemtet Monitor", className="display-4"),
                html.Hr(),
                html.P(
                    "最適化の進捗を可視化します.", className="lead"
                ),
                dbc.Nav(
                    [
                        dbc.NavLink("Home", href="/", active="exact"),
                        # dbc.NavLink("ペアプロット", href="/page-1", active="exact"),
                    ],
                    vertical=True,
                    pills=True,
                ),
            ],
            style=SIDEBAR_STYLE,
        )
        content = html.Div(id="page-content", style=CONTENT_STYLE)
        self.app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

        # sidebar によるページ遷移のための callback
        @self.app.callback(Output("page-content", "children"), [Input("url", "pathname")])
        def render_page_content(pathname):
            if pathname == "/":  # p0
                return self.home
            # elif pathname == "/page-1":
            #     return self.multi_pairplot_layout
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

        # 1. 一定時間ごとに ==> 自動更新が有効なら figure を更新する
        # 2. 中断ボタンを押したら ==> 更新を無効にする and 中断を無効にする
        # 3. メイン処理が中断 or 終了していたら ==> 更新を無効にする and 中断を無効にする
        # 4. toggle_button が押されたら ==> 更新を有効にする or 更新を無効にする
        # 5. タブを押したら ==> グラフの種類を切り替える
        @self.app.callback(
            [
                Output('interval-component', 'max_intervals'),  # 2 3 4
                Output('interrupt-button', 'disabled'),  # 2 3
                Output('toggle-update-button', 'disabled'),  # 2 3 4
                Output('toggle-update-button', 'children'),  # 2 3 4
                Output('card-content', 'children'),  # 1 5
            ],
            [
                Input('interval-component', 'n_intervals'),  # 1 3
                Input('toggle-update-button', 'n_clicks'),  # 4
                Input('interrupt-button', 'n_clicks'),  # 2
                Input("card-tabs", "active_tab"),  # 5
            ]
        )
        def control(
                _,  # n_intervals
                toggle_n_clicks,
                interrupt_n_clicks,
                active_tab_id,
        ):
            # 引数の処理
            toggle_n_clicks = 0 if toggle_n_clicks is None else toggle_n_clicks
            interrupt_n_clicks = 0 if interrupt_n_clicks is None else interrupt_n_clicks

            # 下記を基本に戻り値を上書きしていく（優先のものほど下に来る）
            max_intervals = -1  # enable
            button_disable = False
            toggle_text = 'グラフの自動更新を一時停止する'
            graph = None

            # toggle_button が奇数なら interval を disable にする
            if toggle_n_clicks % 2 == 1:
                max_intervals = 0  # disable
                button_disable = False
                toggle_text = 'グラフの自動更新を再開する'

            # 中断又は終了なら interval とボタンを disable にする
            should_stop = False
            try:
                state = self.femopt.ipv.get_state()
                should_stop = (state == 'interrupted') or (state == 'terminated')
            except AttributeError:
                should_stop = True
            finally:
                if should_stop:
                    max_intervals = 0  # disable
                    button_disable = True
                    toggle_text = 'グラフの更新は行われません'

            # 中断ボタンが押されたなら interval とボタンを disable にして femopt の状態を set する
            button_id = ctx.triggered_id if not None else 'No clicks yet'
            if button_id == 'interrupt-button':
                max_intervals = 0  # disable
                button_disable = True
                toggle_text = 'グラフの更新は行われません'
                self.femopt.ipv.set_state('interrupted')

            # グラフを更新する
            if active_tab_id is not None:
                if active_tab_id == "tab-1":
                    graph = dcc.Graph(figure=update_scatter_matrix(self.femopt))
                elif active_tab_id == "tab-2":
                    graph = dcc.Graph(figure=update_hypervolume_plot(self.femopt))

            return max_intervals, button_disable, button_disable, toggle_text, graph

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
    import datetime
    from time import sleep
    from threading import Thread
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
            self.path = 'tmp.csv'
            self.data = None
            t = Thread(target=self.update)
            t.start()

        def update(self):

            d = dict(
                trial=range(5),
                hypervolume=np.random.rand(5),
                time=[datetime.datetime(year=2000, month=1, day=1, second=s) for s in range(5)]
            )
            for obj_name in self.obj_names:
                d[obj_name] = np.random.rand(5)

            while True:
                self.data = pd.DataFrame(d)
                sleep(1)


    class FEMOPT:
        def __init__(self, history, ipv):
            self.history = history
            self.ipv = ipv


    _ipv = IPV()
    _history = History()
    _femopt = FEMOPT(_history, _ipv)
    monitor = Monitor(_femopt)
    monitor.start_server()
