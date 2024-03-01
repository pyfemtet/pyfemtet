from typing import Optional, List

import webbrowser
import logging
from time import sleep
from threading import Thread

import psutil
from plotly.graph_objects import Figure
from dash import Dash, html, dcc, Output, Input, State, callback_context, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from .visualization import update_default_figure, update_hypervolume_plot
import pyfemtet
from pyfemtet.logger import get_logger
logger = get_logger('viz')
logger.setLevel(logging.INFO)


def _unused_port_number(start=49152):
    # "LISTEN" 状態のポート番号をリスト化
    used_ports = [conn.laddr.port for conn in psutil.net_connections() if conn.status == 'LISTEN']
    port = start
    for port in range(start, 65535 + 1):
        # 未使用のポート番号ならreturn
        if port not in set(used_ports):
            break
    if port != start:
        logger.warn(f'Specified port "{start}" seems to be used. Port "{port}" is used instead.')
    return port


class Home:

    layout = None

    # component id
    ID_DUMMY = 'home-dummy'
    ID_INTERVAL = 'home-interval'
    ID_GRAPH_TABS = 'home-graph-tabs'
    ID_GRAPH_CARD_BODY = 'home-graph-card-body'
    ID_GRAPH = 'home-graph'

    # component id for Monitor
    ID_ENTIRE_PROCESS_STATUS_ALERT = 'home-entire-process-status-alert'
    ID_ENTIRE_PROCESS_STATUS_ALERT_CHILDREN = 'home-entire-process-status-alert-children'
    ID_TOGGLE_INTERVAL_BUTTON = 'home-toggle-interval-button'
    ID_INTERRUPT_PROCESS_BUTTON = 'home-interrupt-process-button'

    # invisible components
    dummy = html.Div(id=ID_DUMMY)
    interval = dcc.Interval(id=ID_INTERVAL, interval=1000, n_intervals=0)

    # visible components
    header = html.H1('最適化の結果分析')
    graph_card = dbc.Card()
    contents = dbc.Container(fluid=True)

    # card + tabs + tab + loading + graph 用
    graphs = {}  # {tab_id: {label, fig_func}}

    def __init__(self, monitor):
        self.monitor = monitor
        self.app: Dash = monitor.app
        self.history = monitor.history
        self.df = monitor.local_df
        self.setup_graph_card()
        self.setup_contents(is_processing=monitor.is_processing)
        self.setup_layout()

    def setup_graph_card(self):

        # graph の追加
        default_tab = 'tab-id-1'
        self.graphs[default_tab] = dict(
            label='目的プロット',
            fig_func=update_default_figure,
        )

        self.graphs['tab-id-2'] = dict(
            label='ハイパーボリューム',
            fig_func=update_hypervolume_plot,
        )

        # graphs から tab 作成
        tabs = []
        for tab_id, graph in self.graphs.items():
            tabs.append(dbc.Tab(label=graph['label'], tab_id=tab_id))

        # tab から tabs 作成
        tabs_component = dbc.Tabs(
            tabs,
            id=self.ID_GRAPH_TABS,
            active_tab=default_tab,
        )

        # タブ + グラフ本体のコンポーネント作成
        self.graph_card = dbc.Card(
            [
                dbc.CardHeader(tabs_component),
                dbc.CardBody(
                    children=[
                        # Loading : child が Output である callback について、
                        # それが発火してから return するまでの間 Spinner が出てくる
                        dcc.Loading(
                            dcc.Graph(id=self.ID_GRAPH),
                        ),
                    ],
                    id=self.ID_GRAPH_CARD_BODY,
                ),
            ],
        )

        # Loading 表示のためページロード時のみ発火させる callback
        @self.app.callback(
            [
                Output(self.ID_GRAPH, 'figure'),
            ],
            [
                Input(self.ID_GRAPH_CARD_BODY, 'children'),
            ],
            [
                State(self.ID_GRAPH_TABS, 'active_tab'),
            ]
        )
        def on_load(_, active_tab):
            # 1. initial_call または 2. card-body (即ちその中の graph) が変化した時に call される
            # 2 で call された場合は ctx に card_body が格納されるのでそれで判定する
            initial_call = callback_context.triggered_id is None
            if initial_call:
                return [self.get_fig_by_tab_id(active_tab)]
            else:
                raise PreventUpdate

        # tab によるグラフ切替 callback
        @self.app.callback(
            [Output(self.ID_GRAPH, 'figure', allow_duplicate=True),],
            [Input(self.ID_GRAPH_TABS, 'active_tab'),],
            prevent_initial_call=True,
        )
        def switch_fig_by_tab(active_tab_id):
            logger.debug(f'switch_fig_by_tab: {active_tab_id}')
            if active_tab_id in self.graphs.keys():
                fig_func = self.graphs[active_tab_id]['fig_func']
                fig = fig_func(self.history, self.monitor.local_df)
            else:
                from plotly.graph_objects import Figure
                fig = Figure()

            return [fig]

    def get_fig_by_tab_id(self, tab_id):
        if tab_id in self.graphs.keys():
            fig_func = self.graphs[tab_id]['fig_func']
            fig = fig_func(self.history, self.monitor.local_df)
        else:
            fig = Figure()
        return fig

    def setup_contents(self, is_processing):
        # contents
        if is_processing:
            self.setup_contents_as_process_monitor()
        else:
            note = dcc.Markdown(
                '---\n'
                '- 最適化の結果分析画面です。\n'
                '- ブラウザを使用しますが、ネットワーク通信は行いません。\n'
                '- ブラウザを閉じてもプログラムは終了しません。'
                '  - コマンドプロンプトを閉じるかコマンドプロンプトに `CTRL+C` を入力してプログラムを終了してください。\n'
            )
            self.contents.children = dbc.Row([dbc.Col(note)])

    def setup_contents_as_process_monitor(self):

        # header
        self.header.children = '最適化の進捗状況'

        # whole status
        status_alert = dbc.Alert(
            children=html.H4(
                'Optimization status will be shown here.',
                className='alert-heading',
                id=self.ID_ENTIRE_PROCESS_STATUS_ALERT_CHILDREN,
            ),
            id=self.ID_ENTIRE_PROCESS_STATUS_ALERT,
            color='secondary',
        )

        # buttons
        toggle_interval_button = dbc.Button('グラフ自動更新を一時停止', id=self.ID_TOGGLE_INTERVAL_BUTTON, color='primary')
        interrupt_button = dbc.Button('最適化を中断', id=self.ID_INTERRUPT_PROCESS_BUTTON, color='danger')

        note = dcc.Markdown(
            '---\n'
            '- 最適化の結果分析画面です。\n'
            '- ブラウザを使用しますが、ネットワーク通信は行いません。\n'
            '- この画面を閉じても最適化は中断されません。\n'
            f'- この画面を再び開くにはブラウザのアドレスバーに「localhost:{self.monitor.DEFAULT_PORT}」と入力して下さい。\n'
            '  - ネットワーク経由で他の PC からこの画面を開く方法は '
            'API reference (pyfemtet.opt.monitor.StaticMonitor.run) を参照してください。\n'
        )

        self.contents.children = [
            dbc.Row([dbc.Col(status_alert)]),
            dbc.Row([dbc.Col(toggle_interval_button), dbc.Col(interrupt_button)]),
            dbc.Row([dbc.Col(note)]),
        ]

        self.add_callback_as_process_monitor()

    def add_callback_as_process_monitor(self):
        # 1. interval           =>  figure を更新する
        # 2. btn interrupt      =>  x(status を interrupt にする) and (interrupt を無効)
        # 3. btn toggle         =>  (toggle の children を切替) and (interval を切替)
        # a. status             =>  status-alert を更新
        # b. status terminated  =>  (toggle を無効) and (interrupt を無効) and (interval を無効)
        @self.monitor.app.callback(
            [
                Output(self.ID_GRAPH_CARD_BODY, 'children'),  # 1
                Output(self.ID_INTERRUPT_PROCESS_BUTTON, 'disabled'),  # 2, b
                Output(self.ID_TOGGLE_INTERVAL_BUTTON, 'children'),  # 3
                Output(self.ID_INTERVAL, 'max_intervals'),  # 3, b
                Output(self.ID_TOGGLE_INTERVAL_BUTTON, 'disabled'),  # b
                Output(self.ID_ENTIRE_PROCESS_STATUS_ALERT_CHILDREN, 'children'),  # a
                Output(self.ID_ENTIRE_PROCESS_STATUS_ALERT, 'color'),  # a
            ],
            [
                Input(self.ID_INTERVAL, 'n_intervals'),  # 1
                Input(self.ID_INTERRUPT_PROCESS_BUTTON, 'n_clicks'),  # 2
                Input(self.ID_TOGGLE_INTERVAL_BUTTON, 'n_clicks'),  # 3
            ],
            [
                State(self.ID_GRAPH_TABS, "active_tab"),
            ],
            prevent_initial_call=True,
        )
        def monitor_feature(
                _1,
                _2,
                toggle_n_clicks,
                active_tab_id,
        ):
            # 発火確認
            logger.debug(f'monitor_feature: {active_tab_id}')

            # 引数の処理
            toggle_n_clicks = toggle_n_clicks or 0
            trigger = callback_context.triggered_id or 'implicit trigger'

            # cls
            OptimizationStatus = pyfemtet.opt.base.OptimizationStatus

            # return 値の default 値(Python >= 3.7 で順番を保持する仕様を利用)
            ret = {
                (card_body := 'card_body'): no_update,
                (disable_interrupt := 'disable_interrupt'): no_update,
                (toggle_btn_msg := 'toggle_btn_msg'): no_update,
                (max_intervals := 'max_intervals'): no_update,  # 0:disable, -1:enable
                (disable_toggle := 'disable_toggle'): no_update,
                (status_msg := 'status_children'): no_update,
                (status_color := 'status_color'): no_update,
            }

            # 2. btn interrupt => x(status を interrupt にする) and (interrupt を無効)
            if trigger == self.ID_INTERRUPT_PROCESS_BUTTON:
                # status を更新
                # component は下流の処理で更新する
                self.monitor.local_entire_status_int = OptimizationStatus.INTERRUPTING
                self.monitor.local_entire_status = OptimizationStatus.const_to_str(OptimizationStatus.INTERRUPTING)

            # 1. interval => figure を更新する
            if (len(self.monitor.local_df) > 0) and (active_tab_id is not None):
                fig = self.get_fig_by_tab_id(active_tab_id)
                ret[card_body] = dcc.Graph(figure=fig, id=self.ID_GRAPH)  # list にせんとダメかも

            # 3. btn toggle => (toggle の children を切替) and (interval を切替)
            if toggle_n_clicks % 2 == 1:
                ret[max_intervals] = 0  # disable
                ret[toggle_btn_msg] = '自動更新を再開'
            else:
                ret[max_intervals] = -1  # enable
                ret[toggle_btn_msg] = '自動更新を一時停止'

            # a. status => status-alert を更新
            ret[status_msg] = 'Optimization status: ' + self.monitor.local_entire_status
            if self.monitor.local_entire_status_int == OptimizationStatus.INTERRUPTING:
                ret[status_color] = 'warning'
            elif self.monitor.local_entire_status_int == OptimizationStatus.TERMINATED:
                ret[status_color] = 'dark'
            elif self.monitor.local_entire_status_int == OptimizationStatus.TERMINATE_ALL:
                ret[status_color] = 'dark'
            else:
                ret[status_color] = 'primary'

            # b. status terminated => (interrupt を無効) and (interval を無効)
            # 中断以降なら中断ボタンを disable にする
            if self.monitor.local_entire_status_int >= OptimizationStatus.INTERRUPTING:
                ret[disable_interrupt] = True
            # 終了以降ならさらに toggle, interval を disable にする
            if self.monitor.local_entire_status_int >= OptimizationStatus.TERMINATED:
                ret[max_intervals] = 0  # disable
                ret[disable_toggle] = True
                ret[toggle_btn_msg] = '更新されません'

            return tuple(ret.values())

    def setup_layout(self):
        # https://dash-bootstrap-components.opensource.faculty.ai/docs/components/accordion/
        self.layout = dbc.Container([
            dbc.Row([dbc.Col(self.dummy), dbc.Col(self.interval)]),
            dbc.Row([dbc.Col(self.header)]),
            dbc.Row([dbc.Col(self.graph_card)]),
            dbc.Row([dbc.Col(self.contents)]),
        ], fluid=True)


class WorkerMonitor:

    # layout
    layout = dbc.Container()

    # id
    ID_INTERVAL = 'worker-monitor-interval'
    id_worker_alert_list = []

    def __init__(self, monitor):
        self.monitor = monitor
        self.setup_layout()
        self.add_callback()

    def setup_layout(self):
        # common
        dummy = html.Div()
        interval = dcc.Interval(id=self.ID_INTERVAL, interval=1000)

        rows = [dbc.Row([dbc.Col(dummy), dbc.Col(interval)])]

        # contents
        worker_status_alerts = []
        for i in range(len(self.monitor.worker_addresses)):
            id_worker_alert = f'worker-status-alert-{i}'
            alert = dbc.Alert('worker status here', id=id_worker_alert, color='dark')
            worker_status_alerts.append(dbc.Row([dbc.Col(alert)]))
            self.id_worker_alert_list.append(id_worker_alert)

        rows.extend(worker_status_alerts)

        self.layout = dbc.Container(rows, fluid=True)

    def add_callback(self):
        # worker_monitor のための callback
        @self.monitor.app.callback(
            [Output(f'{id_worker_alert}', 'children') for id_worker_alert in self.id_worker_alert_list],
            [Output(f'{id_worker_alert}', 'color') for id_worker_alert in self.id_worker_alert_list],
            [Input(self.ID_INTERVAL, 'n_intervals'),]
        )
        def update_worker_state(_):

            OptimizationStatus = pyfemtet.opt.base.OptimizationStatus

            ret = []

            for worker_address, worker_status_int in zip(self.monitor.worker_addresses, self.monitor.local_worker_status_int_list):
                worker_status_message = OptimizationStatus.const_to_str(worker_status_int)
                ret.append(f'{worker_address} is {worker_status_message}')

            colors = []
            for status_int in self.monitor.local_worker_status_int_list:
                if status_int == OptimizationStatus.INTERRUPTING:
                    colors.append('warning')
                elif status_int == OptimizationStatus.TERMINATED:
                    colors.append('dark')
                elif status_int == OptimizationStatus.TERMINATE_ALL:
                    colors.append('dark')
                else:
                    colors.append('primary')

            ret.extend(colors)

            return tuple(ret)


class StaticMonitor(object):

    # process_monitor or not
    is_processing = False

    # port
    DEFAULT_PORT = 49152

    # members for sidebar application
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    }
    CONTENT_STYLE = {
        "margin-left": "18rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    }
    pages = {}  # {href: layout}
    nav_links = {}  # {order(positive float): NavLink}

    # members updated by main threads
    local_df = None

    def __init__(
            self,
            history,
    ):
        # 引数の処理
        self.history = history

        # df の初期化
        self.local_df = self.history.local_data

        # app の立上げ
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        self.setup_home()

    def setup_home(self):
        href = "/"
        home = Home(self)
        self.pages[href] = home.layout
        order = 1
        self.nav_links[order] = dbc.NavLink("Home", href=href, active="exact")

    def setup_layout(self):
        # setup sidebar
        # https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar/

        # sidebar に表示される順に並び替え
        ordered_items = sorted(self.nav_links.items(), key=lambda x: x[0])
        ordered_links = [value for key, value in ordered_items]

        # sidebar と contents から app 全体の layout を作成
        sidebar = html.Div(
            [
                html.H2('PyFemtet Monitor', className='display-4'),
                html.Hr(),
                html.P('最適化結果の可視化', className='lead'),
                dbc.Nav(ordered_links, vertical=True, pills=True),
            ],
            style=self.SIDEBAR_STYLE,
        )
        content = html.Div(id="page-content", style=self.CONTENT_STYLE)
        self.app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

        # sidebar によるページ遷移のための callback
        @self.app.callback(Output("page-content", "children"), [Input("url", "pathname")])
        def switch_page_content(pathname):
            if pathname in list(self.pages.keys()):
                return self.pages[pathname]

            else:
                return html.Div(
                    [
                        html.H1("404: Not found", className="text-danger"),
                        html.Hr(),
                        html.P(f"The pathname {pathname} was not recognised..."),
                    ],
                    className="p-3 bg-light rounded-3",
                )

    def run(self, host='localhost', port=None):
        self.setup_layout()
        port = port or self.DEFAULT_PORT
        # port を検証
        port = _unused_port_number(port)
        # ブラウザを起動
        if host == '0.0.0.0':
            webbrowser.open(f'http://localhost:{str(port)}')
        else:
            webbrowser.open(f'http://{host}:{str(port)}')
        self.app.run(debug=False, host=host, port=port)


class Monitor(StaticMonitor):

    DEFAULT_PORT = 8080

    def __init__(
            self,
            history,
            status: 'pyfemtet.opt.base.OptimizationStatus',
            worker_addresses: List[str],
            worker_status_list: List['pyfemtet.opt.base.OptimizationStatus'],
    ):

        self.status = status
        self.worker_addresses = worker_addresses
        self.worker_status_list = worker_status_list
        self.is_processing = True

        # ログファイルの保存場所
        if logger.level != logging.DEBUG:
            log_path = history.path.replace('.csv', '.uilog')
            monitor_logger = logging.getLogger('werkzeug')
            monitor_logger.addHandler(logging.FileHandler(log_path))

        # メインスレッドで更新してもらうメンバーを一旦初期化
        self.local_entire_status = self.status.get_text()
        self.local_entire_status_int = self.status.get()
        self.local_worker_status_int_list = [s.get() for s in self.worker_status_list]

        # dash app 設定
        super().__init__(history)

        # page 設定
        self.setup_worker_monitor()

    def setup_worker_monitor(self):
        href = "/worker-monitor"
        page = WorkerMonitor(self)
        self.pages[href] = page.layout
        order = 2
        self.nav_links[order] = dbc.NavLink("Workers", href=href, active="exact")

    def start_server(
            self,
            host=None,
            port=None,
    ):
        host = host or 'localhost'
        port = port or self.DEFAULT_PORT

        from .base import OptimizationStatus

        # dash app server を daemon thread で起動
        server_thread = Thread(
            target=self.run,
            args=(host, port,),
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
                    (self.local_entire_status_int == OptimizationStatus.INTERRUPTING)  # monitor の status が INTERRUPT である
            ):
                self.status.set(OptimizationStatus.INTERRUPTING)

            # current status と df を actor から monitor に反映する
            self.local_entire_status_int = self.status.get()
            self.local_entire_status = self.status.get_text()
            self.local_df = self.history.actor_data.copy()
            self.local_worker_status_int_list = [s.get() for s in self.worker_status_list]

            # terminate_all 指令があれば monitor server をホストするプロセスごと終了する
            if self.status.get() == OptimizationStatus.TERMINATE_ALL:
                return 0  # take server down with me

            # interval
            sleep(1)
