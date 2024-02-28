from typing import Optional, List

import webbrowser
import logging
from time import sleep
from threading import Thread

from dash import Dash, html, dcc, ctx, Output, Input
import dash_bootstrap_components as dbc

from .visualization import update_default_figure, update_hypervolume_plot

import pyfemtet


class Home:

    layout = None

    # system components
    dummy = html.Div('', id='dummy-component')
    interval = dcc.Interval(id='interval-component', interval=1000, n_intervals=0)

    # visible components
    header = html.H1('最適化の結果分析')
    graph_card = dbc.Card()
    contents = dbc.Container(fluid=True)

    # 自動更新
    auto_update = False
    graphs = {}  # {tab_id: {label, fig_func}}
    active_tab = None

    def __init__(self, monitor):
        self.monitor = monitor
        self.app: Dash = monitor.app
        self.history = monitor.history
        self.df = monitor.local_df
        self.setup_graph_card()
        self.setup_contents(is_processing=monitor.is_processing)
        self.setup_layout()

    def setup_graph_card(self):

        # タブと graph の追加
        self.graphs['tab-id-1'] = dict(
            label='目的プロット',
            fig_func=update_default_figure,
        )
        self.active_tab = 'tab-id-1'

        self.graphs['tab-id-2'] = dict(
            label='ハイパーボリューム',
            fig_func=update_hypervolume_plot,
        )

        # tab 作成
        tabs = []
        for tab_id, graph in self.graphs.items():
            tabs.append(dbc.Tab(label=graph['label'], tab_id=tab_id))

        # タブ
        tabs_component = dbc.Tabs(
            tabs,
            id='card-tabs',
            active_tab=self.active_tab,
        )

        # タブ + グラフ本体のコンポーネント作成
        self.graph_card = dbc.Card(
            [
                dbc.CardHeader(tabs_component),
                dbc.CardBody(html.P(dcc.Graph(id='graph-component'), className="card-text")),
            ]
        )

        # tab によるグラフ切替 callback
        @self.app.callback(
            [Output('graph-component', 'figure'),],
            [Input("card-tabs", "active_tab"),]
        )
        def switch_fig_by_tab(active_tab_id):

            self.active_tab = active_tab_id

            graph_figure = dcc.Graph()

            if len(self.df) == 0:
                return (graph_figure,)

            if self.active_tab in self.graphs.keys():
                fig_func = self.graphs[self.active_tab]['fig_func']
                graph_figure = fig_func(self.history, self.monitor.local_df)

            return (graph_figure,)

    def setup_contents(self, is_processing):
        # contents
        if is_processing:
            self.setup_contents_as_process_monitor()
        else:
            note = dcc.Markdown(
                '---\n'
                '- 最適化の結果分析画面です。\n'
                '- ブラウザを使用しますが、ネットワーク通信は行いません。\n'
            )
            self.contents.children = dbc.Row([dbc.Col(note)])

    def setup_contents_as_process_monitor(self):

        # 自動更新
        self.auto_update = True

        # header
        self.header.children = '最適化の進捗状況'

        # whole status
        status_alert = dbc.Alert('Optimization status will be shown here.', id='status-alert', color='primary')

        # buttons
        toggle_interval_button = dbc.Button('グラフ自動更新を一時停止', id='toggle-interval-button', color='primary')
        interrupt_button = dbc.Button('最適化を中断', id='interrupt-button', color='danger')

        note = dcc.Markdown(
            '---\n'
            '- 最適化の結果分析画面です。\n'
            '- ブラウザを使用しますが、ネットワーク通信は行いません。\n'
            '- この画面を閉じても最適化は中断されません。\n'
            '- この画面を再び開くにはブラウザのアドレスバーに「localhost:8050」と入力して下さい。\n'
            '  - ネットワーク経由で他の PC からこの画面を開く方法は '
            'API reference (pyfemtet.opt.monitor.Monitor.run) を参照してください。\n'
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
                Output('graph-component', 'figure', allow_duplicate=True),  # 1 tab も fig を切り替える
                Output('interrupt-button', 'disabled'),  # 2, b
                Output('toggle-interval-button', 'children'),  # 3
                Output('interval-component', 'max_intervals'),  # 3, b
                Output('toggle-interval-button', 'disabled'),  # b
                Output('status-alert', 'children'),  # a
                Output('status-alert', 'color'),  # a
                # Output('dummy-component', 'children'),  # debug
            ],
            [
                Input('interval-component', 'n_intervals'),  # 1
                Input('interrupt-button', 'n_clicks'),  # 2
                Input('toggle-interval-button', 'n_clicks'),  # 3
            ],
            prevent_initial_call=True,
        )
        def monitor_work(
                _,
                interrupt_n_clicks,
                toggle_n_clicks,
        ):

            OptimizationStatus = pyfemtet.opt.base.OptimizationStatus

            # 引数の処理
            toggle_n_clicks = 0 if toggle_n_clicks is None else toggle_n_clicks
            pressed_button_id = ctx.triggered_id if not None else 'No clicks yet'

            # default の戻り値
            fig = dcc.Graph()
            interrupt_disabled = False
            toggle_children = 'グラフ自動更新を一時停止'
            max_intervals = -1  # enable
            toggle_disabled = False
            status_children = html.H4('optimization status: ' + self.monitor.local_entire_status, className="alert-heading")
            status_color = 'primary'

            # 1. interval => figure を更新する
            if (len(self.monitor.local_df) > 0) and (self.active_tab is not None):
                fig_func = self.graphs[self.active_tab]['fig_func']
                fig = fig_func(self.history, self.monitor.local_df)

            # 2. btn interrupt => x(status を interrupt にする) and (interrupt を無効)
            if pressed_button_id == 'interrupt-button':
                interrupt_disabled = True
                self.monitor.local_entire_status_int = OptimizationStatus.INTERRUPTING
                self.monitor.local_entire_status = OptimizationStatus.const_to_str(OptimizationStatus.INTERRUPTING)

            # 3. btn toggle => (toggle の children を切替) and (interval を切替)
            if toggle_n_clicks % 2 == 1:
                max_intervals = 0  # disable
                toggle_children = 'グラフ自動更新を再開'

            # a. status => status-alert を更新
            if self.monitor.local_entire_status_int == OptimizationStatus.INTERRUPTING:
                status_color = 'warning'
            if self.monitor.local_entire_status_int == OptimizationStatus.TERMINATED:
                status_color = 'dark'
            if self.monitor.local_entire_status_int == OptimizationStatus.TERMINATE_ALL:
                status_color = 'dark'

            # b. status terminated => (interrupt を無効) and (interval を無効)
            if self.monitor.local_entire_status_int >= OptimizationStatus.INTERRUPTING:
                interrupt_disabled = True
                max_intervals = 0  # disable

            ret = (
                fig,
                interrupt_disabled,
                toggle_children,
                max_intervals,
                toggle_disabled,
                status_children,
                status_color,
            )

            return ret

    def setup_layout(self):
        # https://dash-bootstrap-components.opensource.faculty.ai/docs/components/accordion/
        self.layout = dbc.Container([
            dbc.Row([dbc.Col(self.dummy), dbc.Col(self.interval)]),
            dbc.Row([dbc.Col(self.header)]),
            dbc.Row([dbc.Col(self.graph_card)]),
            dbc.Row([dbc.Col(self.contents)]),
        ], fluid=True)


class StaticMonitor(object):

    # process_monitor or not
    is_processing = False

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
        self.setup_sidebar()

    def setup_home(self):
        href = "/"
        home = Home(self)
        self.pages[href] = home.layout
        self.nav_links[1.] = dbc.NavLink("Home", href=href, active="exact")

    def setup_sidebar(self):
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


class Monitor(StaticMonitor):

    def __init__(
            self,
            history,
            status: 'pyfemtet.opt.base.OptimizationStatus',
            worker_addresses: List[str],
            worker_status_list: List['pyfemtet.opt.base.OptimizationStatus'],
    ):

        # 引数の処理
        self.history = history

        # df の初期化
        self.local_df = self.history.local_data


        self.status = status
        self.worker_addresses = worker_addresses
        self.worker_status_list = worker_status_list
        self.is_processing = True

        # # ログファイルの保存場所
        # log_path = history.path.replace('.csv', '.uilog')
        # logger = logging.getLogger()
        # logger.addHandler(logging.FileHandler(log_path))

        # メインスレッドで更新してもらうメンバーを一旦初期化
        self.local_df = history.local_data.copy()
        self.local_entire_status = self.status.get_text()
        self.local_entire_status_int = self.status.get()
        # self.local_worker_status_list = [s.get() for s in self.worker_status_list]

        super().__init__(history)


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
                    (self.local_entire_status_int == OptimizationStatus.INTERRUPTING)  # monitor の status が INTERRUPT である
            ):
                self.status.set(OptimizationStatus.INTERRUPTING)

            # current status と df を actor から monitor に反映する
            self.local_entire_status_int = self.status.get()
            self.local_entire_status = self.status.get_text()
            self.local_df = self.history.actor_data.copy()
            # self.local_worker_status_list = [s.get() for s in worker_status_list]

            # terminate_all 指令があれば monitor server をホストするプロセスごと終了する
            if self.status.get() == OptimizationStatus.TERMINATE_ALL:
                return 0  # take server down with me

            # interval
            sleep(1)

