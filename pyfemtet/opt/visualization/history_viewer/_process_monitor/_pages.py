from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from dash import Output, Input, State, callback_context, no_update, ALL
from dash.exceptions import PreventUpdate

from pyfemtet.opt.visualization.history_viewer._wrapped_components import html, dcc, dbc
from pyfemtet.opt.visualization.history_viewer._base_application import *
from pyfemtet.opt.visualization.history_viewer._complex_components.main_graph import *

from pyfemtet.opt.worker_status import *
from pyfemtet._i18n import Msg, _

if TYPE_CHECKING:
    from pyfemtet.opt.visualization.history_viewer._process_monitor._application import ProcessMonitorApplication


__all__ = [
    'HomePage',
    'WorkerPage',
]


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


class HomePage(AbstractPage):
    application: ProcessMonitorApplication

    def __init__(self, title, rel_url, application: ProcessMonitorApplication):
        super().__init__(title, rel_url, application)

    def setup_component(self):
        # main graph
        # noinspection PyAttributeOutsideInit
        self.main_graph: MainGraph = MainGraph()
        self.add_subpage(self.main_graph)

        # entire optimization status
        # noinspection PyAttributeOutsideInit
        self.entire_status_message = html.H4(
            Msg.DEFAULT_STATUS_ALERT,
            className='alert-heading',
            id='optimization-entire-status-message',
        )
        # noinspection PyAttributeOutsideInit
        self.entire_status = dbc.Alert(
            children=self.entire_status_message,
            id='optimization-entire-status',
            color='secondary',
        )

        # keep axis range
        button_label = _(
                en_message='Keep {y_or_xy} ranges',
                jp_message='{y_or_xy} 範囲を維持',
                y_or_xy='Y' if len(self.application.history.obj_names) == 1 else 'XY'
            )
        # noinspection PyAttributeOutsideInit
        self.toggle_keep_graph_range_button = dbc.Checkbox(
            label=button_label,
            class_name='form-switch',
            id='toggle-keep-range',
            value=False,
        )

        # interrupt button
        # noinspection PyAttributeOutsideInit
        self.interrupt_button = dbc.Button(
            children=Msg.LABEL_INTERRUPT,
            color='danger',
            id='interrupt-optimization-button',
            disabled=True,
        )

        # sync interval
        # noinspection PyAttributeOutsideInit
        self.interval = dcc.Interval(id='process-monitor-home-interval', interval=3000)

    def setup_layout(self):
        """"""
        """
            =======================
            |  | ---------------- |
            |  | |              | |
            |  | |  Main Graph  | |
            |  | |              | |
            |  | ---------------- |
            |  | [stop][interrupt]<---- Buttons
            |  | ---------------- |
            |  | | Status       | |
            | ^| ---------------- |
            ==|====================
              |
              SideBar
        """
        self.layout = dbc.Container(
            children=[
                self.interval,
                self.main_graph.layout,
                self.entire_status,
                dbc.Row(
                    children=[
                        dbc.Col(self.toggle_keep_graph_range_button),
                        dbc.Col(self.interrupt_button, style=DBC_COLUMN_STYLE_RIGHT),
                    ],
                ),
            ]
        )

    def setup_callback(self):
        # setup callback of subpages
        super().setup_callback()

        app = self.application.app

        # ===== Delete Loading Animation =====
        @app.callback(
            Output(self.main_graph.loading.id, self.main_graph.loading.Prop.target_components),
            Input(self.main_graph.graph.id, 'figure'),
            prevent_initial_call=True,
        )
        def disable_loading_animation(_):
            return {}

        # ===== history data to graph ======
        @app.callback(
            Output(self.main_graph.callback_chain_key.id, self.main_graph.callback_chain_arg_keep_range, allow_duplicate=True),  # fire update graph callback
            Input(self.interval.id, self.interval.Prop.n_intervals),
            State(self.toggle_keep_graph_range_button.id, self.toggle_keep_graph_range_button.Prop.value),
            State(self.main_graph.data_length.id, self.main_graph.data_length_prop),  # check should update or not
            prevent_initial_call=True,)
        def update_graph(_, keep_range, current_graph_data_length):
            current_graph_data_length = 0 if current_graph_data_length is None else current_graph_data_length

            if callback_context.triggered_id is None:
                raise PreventUpdate

            # If new data does not exist, do nothing
            if len(self.application.get_df()) <= current_graph_data_length:
                raise PreventUpdate

            # fire callback
            return keep_range

        # ===== show optimization state =====
        @app.callback(
            Output(self.entire_status.id, self.entire_status.Prop.color),
            Output(self.entire_status_message.id, 'children'),
            Input(self.interval.id, self.interval.Prop.n_intervals),
            prevent_initial_call=False,)
        def update_entire_status(*_):
            # get status message
            msg = str(self.application.entire_status.value)
            color = self.application.get_status_color(self.application.entire_status)
            return color, msg

        # ===== Interrupt Optimization and Control Button Disabled =====
        @app.callback(
            Output(self.interrupt_button.id, self.interrupt_button.Prop.disabled),
            Input(self.interrupt_button.id, self.interrupt_button.Prop.n_clicks),
            prevent_initial_call=False,)
        def interrupt_optimization(*_):
            # If page (re)loaded,
            if callback_context.triggered_id is None:
                # Enable only if the entire_state < INTERRUPTING
                if self.application.entire_status.value < WorkerStatus.interrupting:
                    return False
                # Keep disable(default) if process is interrupting or terminating.
                else:
                    raise PreventUpdate

            # If the entire_state < INTERRUPTING, set INTERRUPTING
            if self.application.entire_status.value < WorkerStatus.interrupting:
                self.application.entire_status.value = WorkerStatus.interrupting
                return True

            # keep disabled
            raise PreventUpdate

        # ===== TEST CODE =====
        if self.application.is_debug:

            # # increment status
            # self.layout.children.append(dbc.Button(children='local_status を進める', id='debug-button-1'))
            #
            # @app.callback(Output(self.interval.id, self.interval.Prop.interval),
            #               Input('debug-button-1', 'n_clicks'),
            #               prevent_initial_call=True)
            # def status_change(*_):
            #     self.application.entire_status += 10
            #     for i in range(len(self.application.worker_status_list)):
            #         self.application.worker_status_list[i] += 10
            #     raise PreventUpdate

            # increment data
            self.layout.children.append(dbc.Button(children='local_data を増やす', id='debug-button-2'))

            @app.callback(Output(self.interval.id, self.interval.Prop.interval, allow_duplicate=True),
                          Input('debug-button-2', 'n_clicks'),
                          prevent_initial_call=True)
            def add_data(*_):
                meta_columns = self.application.history._records.column_manager.meta_columns
                df = self.application.local_data

                new_row = df.iloc[-2:]
                obj_index = np.where(np.array(meta_columns) == 'obj')[0]
                for idx in obj_index:
                    new_row.iloc[:, idx] = np.random.rand(len(new_row))

                df = pd.concat([df, new_row])
                df.trial = np.array(range(len(df)))
                logger.debug(df)

                self.application.local_data = df

                raise PreventUpdate


class WorkerPage(AbstractPage):
    application: ProcessMonitorApplication

    def __init__(self, title, rel_url, application):
        super().__init__(title, rel_url, application)

    def setup_component(self):
        # noinspection PyAttributeOutsideInit
        self.interval = dcc.Interval(id='worker-page-interval', interval=1000)

        # noinspection PyAttributeOutsideInit
        self.worker_status_alerts = []
        for i in range(len(self.application.worker_names)):
            id_worker_alert = f'worker-status-alert-{i}'
            alert = dbc.Alert('worker status here', id=id_worker_alert, color='dark')
            self.worker_status_alerts.append(alert)

    def setup_layout(self):
        rows = [self.interval]
        rows.extend(
            [dbc.Row([dbc.Col(html.Div(dcc.Loading(alert, id={"type": "loading", "index": i})))])
             for i, alert in enumerate(self.worker_status_alerts)])

        self.layout = dbc.Container(
            children=rows,
            fluid=True,
        )

    def setup_callback(self):
        app = self.application.app

        @app.callback(
            [Output(alert.id, 'children') for alert in self.worker_status_alerts],
            [Output(alert.id, 'color') for alert in self.worker_status_alerts],
            Output({"type": "loading", "index": ALL}, "target_components"),
            Input(self.interval.id, 'n_intervals'),
        )
        def update_worker_state(_):

            ret = []

            for worker_name, worker_address, worker_status in (
                    zip(
                        self.application.worker_names,
                        self.application.worker_addresses,
                        self.application.worker_status_list,
                    )):
                worker_status_message = worker_status.value.str()
                ret.append(f'{worker_name} ({worker_address}) is '
                           f'{worker_status_message}')

            ret.extend(
                [self.application.get_status_color(worker_status)
                 for worker_status in self.application.worker_status_list])
            ret.append(
                [({} if callback_context.triggered_id is None else no_update)
                 for _ in range(len(self.worker_status_alerts))])

            return tuple(ret)
