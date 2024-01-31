import webbrowser
import logging
from time import sleep
from threading import Thread

import plotly.graph_objs as go
import plotly.express as px
from dash import Dash, html, dcc, ctx, Output, Input
import dash_bootstrap_components as dbc


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


def create_home_layout():
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
    status = dbc.Alert(
        children=[html.H4("optimization status here", className="alert-heading"),],
        color="primary",
        id='status-alert'
    )
    toggle_update_button = dbc.Button('グラフの自動更新の一時停止', id='toggle-update-button')
    interrupt_button = dbc.Button('最適化を中断', id='interrupt-button', color='danger')
    note_text = dcc.Markdown(f'''
---
- このページでは、最適化の進捗状況を見ることができます。
- このページを閉じても最適化は進行します。
- この機能はブラウザによる状況確認機能ですが、インターネット通信は行いません。
- 再びこのページを開くには、ブラウザのアドレスバーに __localhost:8080__ と入力してください。
- ※ 特定のホスト名及びポートを指定するには、OptimizerBase.main() の実行前に
OptimizerBase.set_monitor_host() を実行してください。
    ''')

    # layout の設定
    layout = dbc.Container([
        dbc.Row([dbc.Col(dummy), dbc.Col(interval)]),
        dbc.Row([dbc.Col(header)]),
        dbc.Row([dbc.Col(graphs)]),
        dbc.Row([dbc.Col(status)]),
        dbc.Row([dbc.Col(toggle_update_button), dbc.Col(interrupt_button)]),
        dbc.Row([dbc.Col(note_text)]),
    ], fluid=True)

    return layout


def create_worker_monitor_layout(
        worker_addresses,
        worker_status_int_list
):
    from .base import OptimizationStatus

    interval = dcc.Interval(
        id='worker-status-update-interval',
        interval=1*1000,  # in milliseconds
        n_intervals=0,
    )

    rows = [interval]
    for i, (worker_address, status_int) in enumerate(zip(worker_addresses, worker_status_int_list)):
        status_msg = OptimizationStatus.const_to_str(status_int)
        a = dbc.Alert(
            children=[
                f'({worker_address}) ',
                html.P(
                    status_msg,
                    id=f'worker-status-msg-{i}'
                ),
            ],
            id=f'worker-status-color-{i}',
            color="primary",
        )
        rows.append(dbc.Row([dbc.Col(a)]))

    layout = dbc.Container(
        rows,
        fluid=True
    )

    return layout



class Monitor(object):

    def __init__(self, history, status, worker_addresses, worker_status_list):

        from .base import OptimizationStatus

        # 引数の処理
        self.history = history
        self.status = status

        # メインスレッドで更新してもらうメンバー
        self.current_status_int = self.status.get()
        self.current_status = self.status.get_text()
        self.current_worker_status_list = [s.get() for s in worker_status_list]
        self.df = self.history.actor_data.copy()

        # ログファイルの保存場所
        log_path = self.history.path.replace('.csv', '.uilog')
        l = logging.getLogger()
        l.addHandler(logging.FileHandler(log_path))

        # app の立上げ
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # ページの components と layout の設定
        self.home = create_home_layout()
        self.worker_monitor = create_worker_monitor_layout(worker_addresses, self.current_worker_status_list)

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
                        dbc.NavLink("Workers", href="/page-1", active="exact"),
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
            elif pathname == "/page-1":
                return self.worker_monitor
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
        # 2. 中断ボタンを押したら ==> interrupt をセットする
        # 3. メイン処理が終了していたら ==> 更新を無効にする and 中断を無効にする
        # 4. toggle_button が押されたら ==> 更新を有効にする or 更新を無効にする
        # 5. タブを押したら ==> グラフの種類を切り替える
        # 6. state に応じて  を切り替える
        @self.app.callback(
            [
                Output('interval-component', 'max_intervals'),  # 3 4
                Output('interrupt-button', 'disabled'),  # 3
                Output('toggle-update-button', 'disabled'),  # 3 4
                Output('toggle-update-button', 'children'),  # 3 4
                Output('card-content', 'children'),  # 1 5
                Output('status-alert', 'children'),  # 6
                Output('status-alert', 'color'),  # 6
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
            status_color = 'primary'

            # toggle_button が奇数なら interval を disable にする
            if toggle_n_clicks % 2 == 1:
                max_intervals = 0  # disable
                button_disable = False
                toggle_text = 'グラフの自動更新を再開する'

            # 終了なら interval とボタンを disable にする
            if self.current_status_int == OptimizationStatus.TERMINATED:
                max_intervals = 0  # disable
                button_disable = True
                toggle_text = 'グラフの更新は行われません'

            # 中断ボタンが押されたなら中断状態にする
            button_id = ctx.triggered_id if not None else 'No clicks yet'
            if button_id == 'interrupt-button':
                self.current_status_int = OptimizationStatus.INTERRUPTING
                self.current_status = OptimizationStatus.const_to_str(OptimizationStatus.INTERRUPTING)

            # グラフを更新する
            if active_tab_id is not None:
                if active_tab_id == "tab-1":
                    graph = dcc.Graph(figure=update_scatter_matrix(self.history, self.df))
                elif active_tab_id == "tab-2":
                    graph = dcc.Graph(figure=update_hypervolume_plot(self.history, self.df))

            # status を更新する
            status_children = [
                html.H4(
                    'optimization status: ' + self.current_status,
                    className="alert-heading"
                ),
            ]
            if self.current_status_int == OptimizationStatus.INTERRUPTING:
                status_color = 'warning'
            if self.current_status_int == OptimizationStatus.TERMINATED:
                status_color = 'dark'
            if self.current_status_int == OptimizationStatus.TERMINATE_ALL:
                status_color = 'dark'

            return max_intervals, button_disable, button_disable, toggle_text, graph, status_children, status_color

        # worker_monitor のための callback
        @self.app.callback(
            [Output(f'worker-status-msg-{i}', 'children') for i in range(len(worker_addresses))],
            [Output(f'worker-status-color-{i}', 'color') for i in range(len(worker_addresses))],
            [Input('worker-status-update-interval', 'n_intervals'),]
        )
        def update_worker_state(_):
            msgs = [OptimizationStatus.const_to_str(i) for i in self.current_worker_status_list]

            colors = []
            for status_int in self.current_worker_status_list:
                if status_int == OptimizationStatus.INTERRUPTING:
                    colors.append('warning')
                elif status_int == OptimizationStatus.TERMINATED:
                    colors.append('dark')
                elif status_int == OptimizationStatus.TERMINATE_ALL:
                    colors.append('dark')
                else:
                    colors.append('primary')

            ret = msgs
            ret.extend(colors)

            return tuple(ret)

    def _run_server_forever(self, app, host, port):
        app.run(debug=False, host=host, port=port)

    def start_server(
            self,
            worker_addresses,
            worker_status_list,  # [actor]
            host='localhost',
            port=8080,
    ):

        from .base import OptimizationStatus

        # 引数の処理
        if host is None:
            host = 'localhost'
        if port is None:
            port = 8080

        # ブラウザを起動
        if host == '0.0.0.0':
            webbrowser.open(f'http://localhost:{str(port)}')
        else:
            webbrowser.open(f'http://{host}:{str(port)}')

        # dash app server を daemon thread で起動
        server_thread = Thread(
            target=self._run_server_forever,
            args=(self.app, host, port,),
            daemon=True,
        )
        server_thread.start()

        # dash app (=flask server) の callback で dask の actor にアクセスすると
        # おかしくなることがあるので、ここで必要な情報のみやり取りする
        while True:
            # running 以前に monitor が current status を interrupting にしていれば actor に反映
            if (
                    (self.status.get() <= OptimizationStatus.RUNNING)  # メインプロセスが RUNNING 以前である
                    and
                    (self.current_status_int == OptimizationStatus.INTERRUPTING)  # monitor の status が INTERRUPT である
            ):
                self.status.set(OptimizationStatus.INTERRUPTING)

            # current status と df を actor から monitor に反映する
            self.current_status_int = self.status.get()
            self.current_status = self.status.get_text()
            self.df = self.history.actor_data.copy()
            self.current_worker_status_list = [s.get() for s in worker_status_list]

            # terminate_all 指令があれば monitor server をホストするプロセスごと終了する
            if self.status.get() == OptimizationStatus.TERMINATE_ALL:
                return 0  # take server down with me

            # interval
            sleep(1)

#
# if __name__ == '__main__':
#     import datetime
#     import numpy as np
#     import pandas as pd
#
#
#     class IPV:
#         def __init__(self):
#             self.state = 'running'
#
#         def get_state(self):
#             return self.state
#
#         def set_state(self, state):
#             self.state = state
#
#
#     class History:
#         def __init__(self):
#             self.obj_names = 'A B C D E'.split()
#             self.path = 'tmp.csv'
#             self.data = None
#             t = Thread(target=self.update)
#             t.start()
#
#         def update(self):
#
#             d = dict(
#                 trial=range(5),
#                 hypervolume=np.random.rand(5),
#                 time=[datetime.datetime(year=2000, month=1, day=1, second=s) for s in range(5)]
#             )
#             for obj_name in self.obj_names:
#                 d[obj_name] = np.random.rand(5)
#
#             while True:
#                 self.data = pd.DataFrame(d)
#                 sleep(1)
#
#
#     class FEMOPT:
#         def __init__(self, history, ipv):
#             self.history = history
#             self.ipv = ipv
#
#
#     _ipv = IPV()
#     _history = History()
#     _femopt = FEMOPT(_history, _ipv)
#     monitor = Monitor(_femopt)
#     monitor.start_server()
