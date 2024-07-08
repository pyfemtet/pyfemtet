# type hint
from dash.development.base_component import Component

# callback
from dash import Output, Input, State, no_update, callback_context
from dash.exceptions import PreventUpdate

# components
from dash import dash_table
from pyfemtet.opt.visualization.wrapped_components import html, dcc, dbc

# graph
import pandas as pd
# import plotly.express as px

# the others
import logging
import os
from enum import Enum
import base64
import json
import numpy as np
# noinspection PyUnresolvedReferences
from pythoncom import com_error

from pyfemtet.opt.visualization.complex_components import main_figure_creator
from pyfemtet.opt.visualization.base import PyFemtetApplicationBase, AbstractPage, logger
from pyfemtet.opt.interface._femtet import FemtetInterface
from pyfemtet.opt.visualization.complex_components.main_graph import MainGraph



class FemtetState(Enum):
    unconnected = -1
    missing = 0
    connected_but_empty = 1
    connected = 2


class FemtetControl(AbstractPage):

    def __init__(self):
        super().__init__()
        self.fem: FemtetInterface = None

    def setup_component(self):

        # noinspection PyAttributeOutsideInit
        self.dummy = html.Div(
            id='control-femtet-dummy',
            hidden=True,
        )

        # button
        # noinspection PyAttributeOutsideInit
        self.connect_femtet_button = dbc.Button(
            children='Connect to Femtet',
            id='connect-femtet-button',
        )

        # noinspection PyAttributeOutsideInit
        self.connect_femtet_button_spinner = dbc.Spinner(
            self.connect_femtet_button,
            color='primary',
        )

        # store femtet connection state
        # noinspection PyAttributeOutsideInit
        self.femtet_state_prop = 'data-femtet-state'  # must start with "data-"
        # noinspection PyAttributeOutsideInit
        self.femtet_state = html.Data(
            id='femtet-state',
            **{self.femtet_state_prop: {
                    'pid': 0,
                    'alert': '',
                }
            }
        )

    def setup_layout(self):
        self.layout = [self.dummy, self.femtet_state]

    def setup_callback(self):
        # setup callback of subpages
        super().setup_callback()

        app = self.application.app

        # ===== launch femtet and show spinner =====
        @app.callback(
            Output(self.femtet_state.id, self.femtet_state_prop),
            Output(self.connect_femtet_button.id, self.connect_femtet_button.Prop.children),  # for spinner
            Input(self.connect_femtet_button.id, self.connect_femtet_button.Prop.n_clicks),)
        def connect_femtet(_):

            # ignore when loaded
            if callback_context.triggered_id is None:
                raise PreventUpdate

            # create FemtetInterface arguments
            kwargs, warning_msg = self.create_femtet_interface_args()

            # try to launch femtet
            try:
                self.fem = FemtetInterface(**kwargs)

            except com_error:
                # model_name is not included in femprj
                kwargs.update(dict(model_name=None))
                self.fem = FemtetInterface(**kwargs)
                warning_msg = 'Analysis model name described in csv does not exist in project.'

            # if 'new' femtet, Interface try to terminate Femtet when the self.fem is reconstructed.
            self.fem.quit_when_destruct = False

            state_data = {
                'pid': self.fem.femtet_pid,
                'alert': warning_msg,
            }

            return state_data, no_update


    def create_femtet_interface_args(self) -> [dict, str]:
        """Returns the argument of FemtetInterface and warning message

        # femprj_path=None,
        # model_name=None,
        # connect_method='auto',  # dask worker では __init__ の中で 'new' にするので super() の引数にしない。（しても意味がない）
        # save_pdt='all',  # 'all' or None
        # strictly_pid_specify=True,  # dask worker では True にしたいので super() の引数にしない。
        # allow_without_project=False,  # main でのみ True を許容したいので super() の引数にしない。
        # open_result_with_gui=True,
        # parametric_output_indexes_use_as_objective=None,

        """

        kwargs = dict(
            connect_method='auto',
            femprj_path=None,
            model_name=None,
            allow_without_project=True,
        )

        # check holding history
        if self.application.history is None:
            return kwargs, 'History csv is not read yet. Open your project manually.'

        # get metadata
        additional_metadata = self.application.history.metadata[0]

        # check metadata exists
        if additional_metadata == '':
            return kwargs, 'Cannot read project data from csv. Open your project manually.'

        # check the metadata is valid json
        try:
            d = json.loads(additional_metadata)
            femprj_path = os.path.abspath(d['femprj_path'])
        except (TypeError, json.decoder.JSONDecodeError, KeyError):
            return kwargs, ('Cannot read project data from csv because of the invalid format. '
                            'Valid format is like: '
                            '"{""femprj_path"": ""c:\\path\\to\\sample.femprj"", ""model_name"": ""<model_name>""}". ' 
                            'Open your project manually.')

        # check containing femprj
        if femprj_path is None:
            return kwargs, 'Cannot read project data from csv. Open your project manually.'

        # check femprj exists
        if not os.path.exists(femprj_path):
            return kwargs, '.femprj file described in csv is not found. Open your project manually.'

        # at this point, femprj is valid at least.
        kwargs.update({'femprj_path': femprj_path})

        # check model name
        model_name = d['model_name'] if 'model_name' in d.keys() else None
        msg = '' if model_name is not None else 'Analysis model name is not specified. Open your model in the project manually.'
        kwargs.update({'model_name': model_name})
        return kwargs, msg

    def check_femtet_state(self) -> FemtetState:

        # check fem is initialized
        if self.fem is None:
            return FemtetState.missing

        Femtet = self.fem.Femtet

        # check Femtet is constructed
        if Femtet is None:
            return FemtetState.missing

        # check Femtet is alive
        if not self.fem.femtet_is_alive():
            return FemtetState.missing

        # check a project is opened
        if Femtet.Project == '':
            return FemtetState.connected_but_empty

        return FemtetState.connected

#
# None
#
#
#         #
#         # DESCRIPTION
#         #  Femtet の生死でボタンを disable にする callback
#         @home.monitor.app.callback(
#             [
#                 Output(cls.ID_CONNECT_FEMTET_DROPDOWN, 'disabled', allow_duplicate=True),
#                 Output(cls.ID_TRANSFER_PARAMETER_BUTTON, 'disabled', allow_duplicate=True),
#             ],
#             [
#                 Input(cls.ID_FEMTET_STATE_DATA, cls.ATTR_FEMTET_DATA),
#             ],
#             prevent_initial_call=True,
#         )
#         def disable_femtet_button_on_state(femtet_data):
#             logger.debug(f'disable_femtet_button')
#             ret = {
#                 (femtet_button_disabled := '1'): no_update,
#                 (transfer_button_disabled := '3'): no_update,
#             }
#
#             if callback_context.triggered_id is not None:
#                 femtet_data = femtet_data or {}
#                 if femtet_data['pid'] > 0:
#                     ret[femtet_button_disabled] = True
#                     ret[transfer_button_disabled] = False
#                 else:
#                     ret[femtet_button_disabled] = False
#                     ret[transfer_button_disabled] = True
#
#             return tuple(ret.values())
#
#         #
#         # DESCRIPTION: Femtet 起動ボタンを押したとき、起動完了を待たずにボタンを disable にする callback
#         @home.monitor.app.callback(
#             [
#                 Output(cls.ID_CONNECT_FEMTET_DROPDOWN, 'disabled', allow_duplicate=True),
#                 Output(cls.ID_TRANSFER_PARAMETER_BUTTON, 'disabled', allow_duplicate=True),
#             ],
#             [
#                 Input(cls.ID_LAUNCH_FEMTET_BUTTON, 'n_clicks'),
#                 Input(cls.ID_CONNECT_FEMTET_BUTTON, 'n_clicks'),
#             ],
#             prevent_initial_call=True,
#         )
#         def disable_femtet_button_immediately(launch_n_clicks, connect_n_clicks):
#             logger.debug(f'disable_button_on_state')
#             ret = {
#                 (disable_femtet_button := '1'): no_update,
#                 (disable_transfer_button := '2'): no_update,
#             }
#
#             if callback_context.triggered_id is not None:
#                 ret[disable_femtet_button] = True
#                 ret[disable_transfer_button] = True
#
#             return tuple(ret.values())
#
#         #
#         # DESCRIPTION: 転送ボタンを押すと Femtet にデータを転送する callback
#         @home.monitor.app.callback(
#             [
#                 Output(home.ID_DUMMY, 'children', allow_duplicate=True),
#                 Output(cls.ID_ALERTS, 'children', allow_duplicate=True),
#             ],
#             [
#                 Input(cls.ID_TRANSFER_PARAMETER_BUTTON, 'n_clicks'),
#             ],
#             [
#                 State(cls.ID_FEMTET_STATE_DATA, cls.ATTR_FEMTET_DATA),
#                 State(home.ID_SELECTION_DATA, home.ATTR_SELECTION_DATA),
#                 State(cls.ID_ALERTS, 'children'),
#             ],
#             prevent_initial_call=True,
#         )
#         def transfer_parameter_to_femtet(
#                 n_clicks,
#                 femtet_data,
#                 selection_data,
#                 current_alerts,
#         ):
#             logger.debug('transfer_parameter_to_femtet')
#
#             # 戻り値
#             ret = {
#                 (dummy := 1): no_update,
#                 (key_new_alerts := 2): no_update,
#             }
#
#             # 引数の処理
#             femtet_data = femtet_data or {}
#             selection_data = selection_data or {}
#             alerts = current_alerts or []
#
#             # Femtet が生きているか
#             femtet_alive = False
#             logger.debug(femtet_data.keys())
#             if 'pid' in femtet_data.keys():
#                 pid = femtet_data['pid']
#                 logger.debug(pid)
#                 if (pid > 0) and (psutil.pid_exists(pid)):  # pid_exists(0) == True
#                     femtet_alive = True
#
#             additional_alerts = []
#
#             if not femtet_alive:
#                 msg = '接続された Femtet が見つかりません。'
#                 color = 'danger'
#                 logger.warning(msg)
#                 additional_alerts = add_alert(additional_alerts, msg, color)
#
#             if 'points' not in selection_data.keys():
#                 msg = '何も選択されていません。'
#                 color = 'danger'
#                 logger.warning(msg)
#                 additional_alerts = add_alert(additional_alerts, msg, color)
#
#             logger.debug(additional_alerts)
#             logger.debug(len(additional_alerts) > 0)
#
#             # この時点で警告（エラー）があれば処理せず終了
#             if len(additional_alerts) > 0:
#                 alerts.extend(additional_alerts)
#                 ret[key_new_alerts] = alerts
#                 logger.debug(alerts)
#                 return tuple(ret.values())
#
#             # もし手動で開いているなら現在開いているファイルとモデルを記憶させる
#             if cls.fem.femprj_path is None:
#                 cls.fem.femprj_path = cls.fem.Femtet.Project
#                 cls.fem.model_name = cls.fem.Femtet.AnalysisModelName
#
#             points_dicts = selection_data['points']
#             for points_dict in points_dicts:
#                 logger.debug(points_dict)
#                 trial = points_dict['customdata'][0]
#                 logger.debug(trial)
#                 index = trial - 1
#                 names = [name for name in home.monitor.local_df.columns if name.startswith('prm_')]
#                 values = home.monitor.local_df.iloc[index][names]
#
#                 df = pd.DataFrame(
#                     dict(
#                         name=[name[4:] for name in names],
#                         value=values,
#                     )
#                 )
#
#                 try:
#                     # femprj の保存先を設定
#                     wo_ext, ext = os.path.splitext(cls.fem.femprj_path)
#                     new_femprj_path = wo_ext + f'_trial{trial}' + ext
#
#                     # 保存できないエラー
#                     if os.path.exists(new_femprj_path):
#                         msg = f'{new_femprj_path} は存在するため、保存はスキップされます。'
#                         color = 'danger'
#                         alerts = add_alert(alerts, msg, color)
#                         ret[key_new_alerts] = alerts
#
#                     else:
#                         # Femtet に値を転送
#                         warnings = cls.fem.update_model(df, with_warning=True)  # exception の可能性
#                         for msg in warnings:
#                             color = 'warning'
#                             logger.warning(msg)
#                             alerts = add_alert(alerts, msg, color)
#                             ret[key_new_alerts] = alerts
#
#                         # 存在する femprj に対して bForce=False で SaveProject すると
#                         # Exception が発生して except 節に飛んでしまう
#                         cls.fem.Femtet.SaveProject(
#                             new_femprj_path,  # ProjectFile
#                             False  # bForce
#                         )
#
#                 except Exception as e:  # 広めに取る
#                     msg = ' '.join([arg for arg in e.args if type(arg) is str]) + 'このエラーが発生する主な理由は、Femtet でプロジェクトが開かれていないことです。'
#                     color = 'danger'
#                     alerts = add_alert(alerts, msg, color)
#                     ret[key_new_alerts] = alerts
#
#             # 別のファイルを開いているならば元に戻す
#             if cls.fem.Femtet.Project != cls.fem.femprj_path:
#                 try:
#                     cls.fem.Femtet.LoadProjectAndAnalysisModel(
#                         cls.fem.femprj_path,  # ProjectFile
#                         cls.fem.model_name,  # AnalysisModelName
#                         True  # bForce
#                     )
#                 except Exception as e:
#                     msg = ' '.join([arg for arg in e.args if type(arg) is str]) + '元のファイルを開けません。'
#                     color = 'danger'
#                     alerts = add_alert(alerts, msg, color)
#                     ret[key_new_alerts] = alerts
#
#             return tuple(ret.values())
#
#         #
#         # DESCRIPTION
#         #  Alerts を全部消す callback
#         @home.monitor.app.callback(
#             [
#                 Output(cls.ID_ALERTS, 'children', allow_duplicate=True),
#             ],
#             [
#                 Input(cls.ID_CLEAR_ALERTS, 'n_clicks'),
#             ],
#             prevent_initial_call=True
#         )
#         def clear_all_alerts(
#                 clear_n_clicks
#         ):
#             ret = {
#                 (alerts_key := 1): no_update
#             }
#
#             if callback_context.triggered_id is not None:
#                 ret[alerts_key] = []
#
#             return tuple(ret.values())
