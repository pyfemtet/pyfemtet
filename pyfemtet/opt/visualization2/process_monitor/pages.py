import json
import os
import tempfile
import base64
import logging

import numpy as np
import pandas as pd

from dash import Output, Input, State, callback_context, no_update
from dash.exceptions import PreventUpdate

from pyfemtet.opt.visualization2.wrapped_components import dcc, dbc, html
from pyfemtet.opt.visualization2.base import AbstractPage, logger
from pyfemtet.opt.visualization2.complex_components.main_graph import MainGraph  # , FLEXBOX_STYLE_ALLOW_VERTICAL_FILL
from pyfemtet.opt.visualization2.complex_components.control_femtet import FemtetControl, FemtetState
from pyfemtet.opt.visualization2.complex_components.alert_region import AlertRegion

from pyfemtet.opt._femopt_core import History



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


def is_iterable(component):
    return hasattr(component, '__len__')


class HomePage(AbstractPage):

    def __init__(self, title, rel_url='/'):
        super().__init__(title, rel_url)

    def setup_component(self):
        # main graph
        # noinspection PyAttributeOutsideInit
        self.main_graph: MainGraph = MainGraph()
        self.add_subpage(self.main_graph)

        # entire optimization status
        # noinspection PyAttributeOutsideInit
        self.entire_status_message = html.H4(
            'Optimization status will be shown here.',
            className='alert-heading',
            id='optimization-entire-status-message',
        )
        # noinspection PyAttributeOutsideInit
        self.entire_status = dbc.Alert(
            children=self.entire_status_message,
            id='optimization-entire-status',
            color='secondary',
        )

        # stop update button
        # noinspection PyAttributeOutsideInit
        self.toggle_update_graph_button = dbc.Checkbox(
            label='Auto-update graph',
            class_name='form-switch',
            id='toggle-update-graph',
            value=True,
        )

        # interrupt button
        # noinspection PyAttributeOutsideInit
        self.interrupt_button = dbc.Button(
            children='Interrupt Optimization',
            color='danger',
            id='interrupt-optimization-button',
            disabled=True,
        )

        # sync interval
        # noinspection PyAttributeOutsideInit
        self.interval = dcc.Interval(id='process-monitor-home-interval', interval=1000)

    def setup_layout(self):
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
                        dbc.Col(self.toggle_update_graph_button),
                        dbc.Col(self.interrupt_button, style=DBC_COLUMN_STYLE_RIGHT),
                    ],
                ),
            ]
        )

    def setup_callback(self):
        # setup callback of subpages
        super().setup_callback()

        from pyfemtet.opt.visualization2.process_monitor.application import ProcessMonitorApplication
        from pyfemtet.opt._femopt_core import OptimizationStatus
        self.application: ProcessMonitorApplication = self.application

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
            Output(self.main_graph.dummy.id, 'children', allow_duplicate=True),  # fire update graph callback
            Input(self.interval.id, self.interval.Prop.n_intervals),
            State(self.toggle_update_graph_button.id, self.toggle_update_graph_button.Prop.value),
            State(self.main_graph.data_length.id, self.main_graph.data_length_prop),  # check should update or not
            prevent_initial_call=True,)
        def update_graph(_, update_switch, current_graph_data_length):
            logger.debug('====================')
            logger.debug(f'update_graph called by {callback_context.triggered_id}')

            current_graph_data_length = 0 if current_graph_data_length is None else current_graph_data_length

            if callback_context.triggered_id is None:
                raise PreventUpdate

            # update_switch is unchecked, do nothing
            if not update_switch:
                raise PreventUpdate

            # If new data does not exist, do nothing
            if len(self.application.local_data) <= current_graph_data_length:
                raise PreventUpdate

            # fire callback
            return ''

        # ===== show optimization state =====
        @app.callback(
            Output(self.entire_status.id, self.entire_status.Prop.color),
            Output(self.entire_status_message.id, 'children'),
            Input(self.interval.id, self.interval.Prop.n_intervals),
            prevent_initial_call=False,)
        def update_entire_status(*args):
            # get status message
            status_int = self.application.local_entire_status_int
            msg = OptimizationStatus.const_to_str(status_int)
            color = self.application.get_status_color(status_int)
            return color, msg

        # ===== Interrupt Optimization and Control Button Disabled =====
        @app.callback(
            Output(self.interrupt_button.id, self.interrupt_button.Prop.disabled),
            Input(self.interrupt_button.id, self.interrupt_button.Prop.n_clicks),
            prevent_initial_call=False,)
        def interrupt_optimization(*args):
            # If page (re)loaded,
            if callback_context.triggered_id is None:
                # Enable only if the entire_state < INTERRUPTING
                if self.application.local_entire_status_int < OptimizationStatus.INTERRUPTING:
                    return False
                # Keep disable(default) if process is interrupting or terminating.
                else:
                    raise PreventUpdate

            # If the entire_state < INTERRUPTING, set INTERRUPTING
            if self.application.local_entire_status_int < OptimizationStatus.INTERRUPTING:
                self.application.local_entire_status_int = OptimizationStatus.INTERRUPTING
                return True

            # keep disabled
            raise PreventUpdate

        # ===== TEST CODE =====
        if self.application.is_debug:

            # increment status
            self.layout.children.append(dbc.Button(children='local_status を進める', id='debug-button-1'))

            @app.callback(Output(self.interval.id, self.interval.Prop.interval),
                          Input('debug-button-1', 'n_clicks'),
                          prevent_initial_call=True)
            def status_change(*args):
                self.application.local_entire_status_int += 10
                raise PreventUpdate


            # increment data
            self.layout.children.append(dbc.Button(children='local_data を増やす', id='debug-button-2'))

            @app.callback(Output(self.interval.id, self.interval.Prop.interval, allow_duplicate=True),
                          Input('debug-button-2', 'n_clicks'),
                          prevent_initial_call=True)
            def add_data(*args):
                metadata = self.application.history.metadata
                df = self.application.local_data

                new_row = df.iloc[-2:]
                obj_index = np.where(np.array(metadata) == 'obj')[0]
                for idx in obj_index:
                    new_row.iloc[:, idx] = np.random.rand(len(new_row))

                df = pd.concat([df, new_row])
                df.trial = np.array(range(len(df)))
                logger.debug(df)

                self.application.local_data = df

                raise PreventUpdate


class WorkerPage(AbstractPage):

    def setup_component(self):
        pass

    def setup_layout(self):
        pass

    def setup_callback(self):
        pass

