import os
from typing import Optional, List

import json
import webbrowser
from time import sleep
from threading import Thread

import pandas as pd
import psutil
from plotly.graph_objects import Figure
from dash import Dash, html, dcc, Output, Input, State, callback_context, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import pyfemtet
from pyfemtet.opt.interface import FemtetInterface
from pyfemtet.opt.visualization._graphs import (
    update_default_figure,
    update_hypervolume_plot,
    _CUSTOM_DATA_DICT
)


import logging
from pyfemtet.logger import get_logger
logger = get_logger('viz')
logger.setLevel(logging.INFO)


DBC_COLUMN_STYLE_CENTER = {
    'display': 'flex',
    'justify-content': 'center',
    'align-items': 'center',
}

DBC_COLUMN_STYLE_RIGHT = {
    'display': 'flex',
    'justify-content': 'right',
    'align-items': 'right',
}


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


class FemtetControl:

    # component ID
    ID_LAUNCH_FEMTET_BUTTON = 'femtet-launch-button'
    ID_CONNECT_FEMTET_BUTTON = 'femtet-connect-button'
    ID_CONNECT_FEMTET_DROPDOWN = 'femtet-connect-dropdown'
    ID_FEMTET_STATE_DATA = 'femtet-state-data'
    ID_TRANSFER_PARAMETER_BUTTON = 'femtet-transfer-parameter-button'
    ID_INTERVAL = 'femtet-interval'
    ID_ALERTS = 'femtet-control-alerts'
    ID_CLEAR_ALERTS = 'femtet-control-clear-alerts'

    # arbitrary attribute name
    ATTR_FEMTET_DATA = 'data-femtet'  # attribute of html.Data should start with data-*.

    # interface
    fem: Optional['pyfemtet.opt.interface.FemtetInterface'] = None

    @classmethod
    def create_components(cls):

        interval = dcc.Interval(id=cls.ID_INTERVAL, interval=1000)
        femtet_state = {'pid': 0}
        femtet_state_data = html.Data(id=cls.ID_FEMTET_STATE_DATA, **{cls.ATTR_FEMTET_DATA: femtet_state})

        launch_new_femtet = dbc.DropdownMenuItem(
            '新しい Femtet を起動して接続',
            id=cls.ID_LAUNCH_FEMTET_BUTTON,
        )

        connect_existing_femtet = dbc.DropdownMenuItem(
            '既存の Femtet と接続',
            id=cls.ID_CONNECT_FEMTET_BUTTON,
        )

        launch_femtet_dropdown = dbc.DropdownMenu(
            [
                launch_new_femtet,
                connect_existing_femtet
            ],
            label='Femtet と接続',
            id=cls.ID_CONNECT_FEMTET_DROPDOWN,
        )

        transfer_parameter_button = dbc.Button('変数を Femtet に転送', id=cls.ID_TRANSFER_PARAMETER_BUTTON, color='primary', disabled=True)
        clear_alert = dbc.Button('警告のクリア', id=cls.ID_CLEAR_ALERTS, color='secondary')

        buttons = dbc.Row([
            dbc.Col(
                dbc.ButtonGroup([
                    launch_femtet_dropdown,
                    transfer_parameter_button,
                ])
            ),
            dbc.Col(clear_alert, style=DBC_COLUMN_STYLE_RIGHT),
        ])

        alerts = dbc.CardBody(children=[], id=cls.ID_ALERTS)

        control = dbc.Card([
            dbc.CardHeader(buttons),
            alerts,
        ])

        component = dbc.Container(
            [
                control,
            ]
        )

        return [interval, femtet_state_data, component]

    @classmethod
    def add_callback(cls, home: 'HomePageBase'):

        def launch_femtet_rule():
            # default は手動で開く
            femprj_path = None
            model_name = None
            msg = None
            color = None

            # metadata の有無を確認
            additional_metadata = home.monitor.history.metadata[0]
            is_invalid = False

            # json である && femprj_path と model_name が key にある
            prj_data = None
            try:
                prj_data = json.loads(additional_metadata)
                _ = prj_data['femprj_path']
                _ = prj_data['model_name']
            except (TypeError, json.decoder.JSONDecodeError, KeyError):
                is_invalid = True

            if is_invalid:
                color = 'warning'
                msg = (
                    f'{home.monitor.history.path} に'
                    '解析プロジェクトファイルを示すデータが存在しないので'
                    '解析ファイルを自動で開けません。'
                    'Femtet 起動後、最適化を行ったファイルを'
                    '手動で開いてください。'
                )
                logger.warn(msg)

            else:

                # femprj が存在しなければ警告
                if not os.path.isfile(prj_data['femprj_path']):
                    color = 'warning'
                    msg = (
                        f'{femprj_path} が見つかりません。'
                        'Femtet 起動後、最適化を行ったファイルを'
                        '手動で開いてください。'
                    )

                # femprj はあるがその中に model があるかないかは開かないとわからない
                # そうでなければ自動で開く
                else:
                    femprj_path = prj_data['femprj_path']
                    model_name = prj_data['model_name']

            return femprj_path, model_name, msg, color

        def add_alert(current_alerts, msg, color):
            new_alert = dbc.Alert(
                msg,
                id=f'alert-{len(current_alerts) + 1}',
                dismissable=True,
                color=color,
            )
            current_alerts.append(new_alert)
            return current_alerts

        #
        # DESCRIPTION:
        #   Femtet を起動する際の挙動に従って warning を起こす callback
        @home.monitor.app.callback(
            [
                Output(cls.ID_ALERTS, 'children', allow_duplicate=True),
            ],
            [
                Input(cls.ID_LAUNCH_FEMTET_BUTTON, 'n_clicks'),
                Input(cls.ID_CONNECT_FEMTET_BUTTON, 'n_clicks'),
            ],
            [
                State(cls.ID_ALERTS, 'children'),
            ],
            prevent_initial_call=True,
        )
        def update_alert(_1, _2, current_alerts):
            logger.debug('update-femtet-control-alert')

            # ボタン経由でないなら無視
            if callback_context.triggered_id is None:
                raise PreventUpdate

            # 戻り値
            ret = {
                (new_alerts_key := 1): no_update,
            }

            # 引数の処理
            current_alerts = current_alerts or []

            _, _, msg, color = launch_femtet_rule()

            if msg is not None:
                new_alerts = add_alert(current_alerts, msg, color)
                ret[new_alerts_key] = new_alerts

            return tuple(ret.values())

        #
        # DESCRIPTION:
        #  Femtet を起動し、pid を記憶する callback
        @home.monitor.app.callback(
            [
                Output(cls.ID_FEMTET_STATE_DATA, cls.ATTR_FEMTET_DATA, allow_duplicate=True),
                Output(cls.ID_ALERTS, 'children'),
            ],
            [
                Input(cls.ID_LAUNCH_FEMTET_BUTTON, 'n_clicks'),
                Input(cls.ID_CONNECT_FEMTET_BUTTON, 'n_clicks'),
            ],
            [
                State(cls.ID_ALERTS, 'children'),
            ],
            prevent_initial_call=True,
        )
        def launch_femtet(launch_n_clicks, connect_n_clicks, current_alerts):
            logger.debug(f'launch_femtet')

            # 戻り値の設定
            ret = {
                (femtet_state_data := 1): no_update,
                (key_new_alerts := 2): no_update,
            }

            # 引数の処理
            current_alerts = current_alerts or []

            # ボタン経由でなければ無視
            if callback_context.triggered_id is None:
                raise PreventUpdate

            # Femtet を起動するための引数をルールに沿って取得
            femprj_path, model_name, _, _ = launch_femtet_rule()

            # Femtet の起動方法を取得
            method = None
            if callback_context.triggered_id == cls.ID_LAUNCH_FEMTET_BUTTON:
                method = 'new'
            elif callback_context.triggered_id == cls.ID_CONNECT_FEMTET_BUTTON:
                method = 'existing'

            # Femtet を起動
            try:
                cls.fem = FemtetInterface(
                    femprj_path=femprj_path,
                    model_name=model_name,
                    connect_method=method,
                    allow_without_project=True
                )
                cls.fem.quit_when_destruct = False
                ret[femtet_state_data] = {'pid': cls.fem.femtet_pid}
            except Exception as e:  # 広めに取る
                msg = e.args[0]
                color = 'danger'
                new_alerts = add_alert(current_alerts, msg, color)
                ret[femtet_state_data] = {'pid': 0}  # button の disable を再制御するため
                ret[key_new_alerts] = new_alerts

            return tuple(ret.values())

        #
        # DESCRIPTION
        #  Femtet の情報をアップデートするなどの監視 callback
        @home.monitor.app.callback(
            [
                Output(cls.ID_FEMTET_STATE_DATA, cls.ATTR_FEMTET_DATA),
            ],
            [
                Input(cls.ID_INTERVAL, 'n_intervals'),
            ],
            [
                State(cls.ID_FEMTET_STATE_DATA, cls.ATTR_FEMTET_DATA)
            ],
        )
        def interval_callback(n_intervals, current_femtet_data):
            logger.debug('interval_callback')

            # 戻り値
            ret = {(femtet_data := 1): no_update}

            # 引数の処理
            n_intervals = n_intervals or 0
            current_femtet_data = current_femtet_data or {}

            # check pid
            if 'pid' in current_femtet_data.keys():
                pid = current_femtet_data['pid']
                if pid > 0:
                    # 生きていると主張するなら、死んでいれば更新
                    if not psutil.pid_exists(pid):
                        ret[femtet_data] = current_femtet_data.update({'pid': 0})

            return tuple(ret.values())

        #
        # DESCRIPTION
        #  Femtet の生死でボタンを disable にする callback
        @home.monitor.app.callback(
            [
                Output(cls.ID_CONNECT_FEMTET_DROPDOWN, 'disabled', allow_duplicate=True),
                Output(cls.ID_TRANSFER_PARAMETER_BUTTON, 'disabled', allow_duplicate=True),
            ],
            [
                Input(cls.ID_FEMTET_STATE_DATA, cls.ATTR_FEMTET_DATA),
            ],
            prevent_initial_call=True,
        )
        def disable_femtet_button_on_state(femtet_data):
            logger.debug(f'disable_femtet_button')
            ret = {
                (femtet_button_disabled := '1'): no_update,
                (transfer_button_disabled := '3'): no_update,
            }

            if callback_context.triggered_id is not None:
                femtet_data = femtet_data or {}
                if femtet_data['pid'] > 0:
                    ret[femtet_button_disabled] = True
                    ret[transfer_button_disabled] = False
                else:
                    ret[femtet_button_disabled] = False
                    ret[transfer_button_disabled] = True

            return tuple(ret.values())

        #
        # DESCRIPTION: Femtet 起動ボタンを押したとき、起動完了を待たずにボタンを disable にする callback
        @home.monitor.app.callback(
            [
                Output(cls.ID_CONNECT_FEMTET_DROPDOWN, 'disabled', allow_duplicate=True),
                Output(cls.ID_TRANSFER_PARAMETER_BUTTON, 'disabled', allow_duplicate=True),
            ],
            [
                Input(cls.ID_LAUNCH_FEMTET_BUTTON, 'n_clicks'),
                Input(cls.ID_CONNECT_FEMTET_BUTTON, 'n_clicks'),
            ],
            prevent_initial_call=True,
        )
        def disable_femtet_button_immediately(launch_n_clicks, connect_n_clicks):
            logger.debug(f'disable_button_on_state')
            ret = {
                (disable_femtet_button := '1'): no_update,
                (disable_transfer_button := '2'): no_update,
            }

            if callback_context.triggered_id is not None:
                ret[disable_femtet_button] = True
                ret[disable_transfer_button] = True

            return tuple(ret.values())

        #
        # DESCRIPTION: 転送ボタンを押すと Femtet にデータを転送する callback
        @home.monitor.app.callback(
            [
                Output(home.ID_DUMMY, 'children', allow_duplicate=True),
                Output(cls.ID_ALERTS, 'children', allow_duplicate=True),
            ],
            [
                Input(cls.ID_TRANSFER_PARAMETER_BUTTON, 'n_clicks'),
            ],
            [
                State(cls.ID_FEMTET_STATE_DATA, cls.ATTR_FEMTET_DATA),
                State(home.ID_SELECTION_DATA, home.ATTR_SELECTION_DATA),
                State(cls.ID_ALERTS, 'children'),
            ],
            prevent_initial_call=True,
        )
        def transfer_parameter_to_femtet(
                n_clicks,
                femtet_data,
                selection_data,
                current_alerts,
        ):
            logger.debug('transfer_parameter_to_femtet')

            # 戻り値
            ret = {
                (dummy := 1): no_update,
                (key_new_alerts := 2): no_update,
            }

            # 引数の処理
            femtet_data = femtet_data or {}
            selection_data = selection_data or {}
            alerts = current_alerts or []

            # Femtet が生きているか
            femtet_alive = False
            logger.debug(femtet_data.keys())
            if 'pid' in femtet_data.keys():
                pid = femtet_data['pid']
                logger.debug(pid)
                if (pid > 0) and (psutil.pid_exists(pid)):  # pid_exists(0) == True
                    femtet_alive = True

            additional_alerts = []

            if not femtet_alive:
                msg = '接続された Femtet が見つかりません。'
                color = 'danger'
                logger.warning(msg)
                additional_alerts = add_alert(additional_alerts, msg, color)

            if 'points' not in selection_data.keys():
                msg = '何も選択されていません。'
                color = 'danger'
                logger.warning(msg)
                additional_alerts = add_alert(additional_alerts, msg, color)

            logger.debug(additional_alerts)
            logger.debug(len(additional_alerts) > 0)

            # この時点で警告（エラー）があれば処理せず終了
            if len(additional_alerts) > 0:
                alerts.extend(additional_alerts)
                ret[key_new_alerts] = alerts
                logger.debug(alerts)
                return tuple(ret.values())

            # もし手動で開いているなら現在開いているファイルとモデルを記憶させる
            if cls.fem.femprj_path is None:
                cls.fem.femprj_path = cls.fem.Femtet.Project
                cls.fem.model_name = cls.fem.Femtet.AnalysisModelName

            points_dicts = selection_data['points']
            for points_dict in points_dicts:
                logger.debug(points_dict)
                trial = points_dict['customdata'][_CUSTOM_DATA_DICT['trial']]
                logger.debug(trial)
                index = trial - 1
                names = [name for name in home.monitor.local_df.columns if name.startswith('prm_')]
                values = home.monitor.local_df.iloc[index][names]

                df = pd.DataFrame(
                    dict(
                        name=[name[4:] for name in names],
                        value=values,
                    )
                )

                try:
                    # femprj の保存先を設定
                    wo_ext, ext = os.path.splitext(cls.fem.femprj_path)
                    new_femprj_path = wo_ext + f'_trial{trial}' + ext

                    # 保存できないエラー
                    if os.path.exists(new_femprj_path):
                        msg = f'{new_femprj_path} は存在するため、保存はスキップされます。'
                        color = 'danger'
                        alerts = add_alert(alerts, msg, color)
                        ret[key_new_alerts] = alerts

                    else:
                        # Femtet に値を転送
                        warnings = cls.fem.update_model(df, with_warning=True)  # exception の可能性
                        for msg in warnings:
                            color = 'warning'
                            logger.warning(msg)
                            alerts = add_alert(alerts, msg, color)
                            ret[key_new_alerts] = alerts

                        # 存在する femprj に対して bForce=False で SaveProject すると
                        # Exception が発生して except 節に飛んでしまう
                        cls.fem.Femtet.SaveProject(
                            new_femprj_path,  # ProjectFile
                            False  # bForce
                        )

                except Exception as e:  # 広めに取る
                    msg = ' '.join([arg for arg in e.args if type(arg) is str]) + 'このエラーが発生する主な理由は、Femtet でプロジェクトが開かれていないことです。'
                    color = 'danger'
                    alerts = add_alert(alerts, msg, color)
                    ret[key_new_alerts] = alerts

            # 別のファイルを開いているならば元に戻す
            if cls.fem.Femtet.Project != cls.fem.femprj_path:
                try:
                    cls.fem.Femtet.LoadProjectAndAnalysisModel(
                        cls.fem.femprj_path,  # ProjectFile
                        cls.fem.model_name,  # AnalysisModelName
                        True  # bForce
                    )
                except Exception as e:
                    msg = ' '.join([arg for arg in e.args if type(arg) is str]) + '元のファイルを開けません。'
                    color = 'danger'
                    alerts = add_alert(alerts, msg, color)
                    ret[key_new_alerts] = alerts

            return tuple(ret.values())

        #
        # DESCRIPTION
        #  Alerts を全部消す callback
        @home.monitor.app.callback(
            [
                Output(cls.ID_ALERTS, 'children', allow_duplicate=True),
            ],
            [
                Input(cls.ID_CLEAR_ALERTS, 'n_clicks'),
            ],
            prevent_initial_call=True
        )
        def clear_all_alerts(
                clear_n_clicks
        ):
            ret = {
                (alerts_key := 1): no_update
            }

            if callback_context.triggered_id is not None:
                ret[alerts_key] = []

            return tuple(ret.values())


class HomePageBase:

    layout = None

    # component id
    ID_DUMMY = 'home-dummy'
    ID_INTERVAL = 'home-interval'
    ID_GRAPH_TABS = 'home-graph-tabs'
    ID_GRAPH_CARD_BODY = 'home-graph-card-body'
    ID_GRAPH = 'home-graph'
    ID_SELECTION_DATA = 'home-selection-data'

    # selection data attribute
    ATTR_SELECTION_DATA = 'data-selection'  # should start with data-*

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
        self.setup_contents()
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
                        dcc.Loading(dcc.Graph(id=self.ID_GRAPH)),
                    ],
                    id=self.ID_GRAPH_CARD_BODY,
                ),
                html.Data(
                    id=self.ID_SELECTION_DATA,
                    **{self.ATTR_SELECTION_DATA: {}}
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

        # 選択したデータを記憶する callback
        @self.monitor.app.callback(
            [
                Output(self.ID_SELECTION_DATA, self.ATTR_SELECTION_DATA),
            ],
            [
                Input(self.ID_GRAPH, 'selectedData')
            ],
        )
        def on_select(selected_data):
            logger.debug(f'on_select: {selected_data}')
            return [selected_data]

    def get_fig_by_tab_id(self, tab_id):
        if tab_id in self.graphs.keys():
            fig_func = self.graphs[tab_id]['fig_func']
            fig = fig_func(self.history, self.monitor.local_df)
        else:
            fig = Figure()
        return fig

    def setup_contents(self):
        pass

    def setup_layout(self):
        # https://dash-bootstrap-components.opensource.faculty.ai/docs/components/accordion/
        self.layout = dbc.Container([
            dbc.Row([dbc.Col(self.dummy), dbc.Col(self.interval)]),
            dbc.Row([dbc.Col(self.header)]),
            dbc.Row([dbc.Col(self.graph_card)]),
            dbc.Row([dbc.Col(self.contents)]),
        ], fluid=True)



class ResultViewerAppHomePage(HomePageBase):

    def setup_contents(self):
        # Femtet control
        div = FemtetControl.create_components()
        FemtetControl.add_callback(self)

        # note
        note = dcc.Markdown(
            '---\n'
            '- 最適化の結果分析画面です。\n'
            '- 凡例をクリックすると、対応する要素の表示/非表示を切り替えます。\n'
            '- ブラウザを使用しますが、ネットワーク通信は行いません。\n'
            '- ブラウザを閉じてもプログラムは終了しません。'
            '  - コマンドプロンプトを閉じるかコマンドプロンプトに `CTRL+C` を入力してプログラムを終了してください。\n'
        )
        self.contents.children = [
            dbc.Row(div),
            dbc.Row(note)
        ]


class ProcessMonitorAppHomePage(HomePageBase):

    # component id for Monitor
    ID_ENTIRE_PROCESS_STATUS_ALERT = 'home-entire-process-status-alert'
    ID_ENTIRE_PROCESS_STATUS_ALERT_CHILDREN = 'home-entire-process-status-alert-children'
    ID_TOGGLE_INTERVAL_BUTTON = 'home-toggle-interval-button'
    ID_INTERRUPT_PROCESS_BUTTON = 'home-interrupt-process-button'

    def setup_contents(self):

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
        )

        self.contents.children = [
            dbc.Row([dbc.Col(status_alert)]),
            dbc.Row([
                dbc.Col(toggle_interval_button, style=DBC_COLUMN_STYLE_CENTER),
                dbc.Col(interrupt_button, style=DBC_COLUMN_STYLE_CENTER)
            ]),
            dbc.Row([dbc.Col(note)]),
        ]

        self.add_callback()

    def add_callback(self):
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
            from pyfemtet.opt._femopt_core import OptimizationStatus

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


class ProcessMonitorAppWorkerPage:

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

            from pyfemtet.opt._femopt_core import OptimizationStatus

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
                elif status_int == OptimizationStatus.CRASHED:
                    colors.append('danger')
                else:
                    colors.append('primary')

            ret.extend(colors)

            return tuple(ret)


class AppBase(object):

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
        self.app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title='PyFemtet Monitor',
            update_title=None,
        )

    # def setup_some_page(self):
    #     href = "/link-url"
    #     page = SomePageClass(self)
    #     self.pages[href] = page.layout
    #     order: int = 1
    #     self.nav_links[order] = dbc.NavLink('Some Page', href=href, active="exact")

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


class ResultViewerApp(AppBase):

    def __init__(
            self,
            history,
    ):

        # 引数の設定、app の設定
        super().__init__(history)

        # page 設定
        self.setup_home()


    def setup_home(self):
        href = "/"
        page = ResultViewerAppHomePage(self)
        self.pages[href] = page.layout
        order = 1
        self.nav_links[order] = dbc.NavLink("Home", href=href, active="exact")


class ProcessMonitorApp(AppBase):

    DEFAULT_PORT = 8080

    def __init__(
            self,
            history,
            status,
            worker_addresses: List[str],
            worker_status_list: List['pyfemtet.opt._core.OptimizationStatus'],
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
        self.setup_home()
        self.setup_worker_monitor()

    def setup_home(self):
        href = "/"
        page = ProcessMonitorAppHomePage(self)
        self.pages[href] = page.layout
        order = 1
        self.nav_links[order] = dbc.NavLink("Home", href=href, active="exact")

    def setup_worker_monitor(self):
        href = "/worker-monitor/"
        page = ProcessMonitorAppWorkerPage(self)
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

        # dash app server を daemon thread で起動
        server_thread = Thread(
            target=self.run,
            args=(host, port,),
            daemon=True,
        )
        server_thread.start()

        # dash app (=flask server) の callback で dask の actor にアクセスすると
        # おかしくなることがあるので、ここで必要な情報のみやり取りする
        from pyfemtet.opt._femopt_core import OptimizationStatus
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
