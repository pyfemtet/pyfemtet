import numpy as np
import pandas as pd

import optuna

from dash import Output, Input, State, callback_context, no_update, ALL
from dash.exceptions import PreventUpdate

from pyfemtet.opt.visualization.wrapped_components import dcc, dbc, html
from pyfemtet.opt.visualization.base import AbstractPage, logger
from pyfemtet.opt.visualization.complex_components.main_graph import MainGraph  # , FLEXBOX_STYLE_ALLOW_VERTICAL_FILL
from pyfemtet.opt.visualization.complex_components.pm_graph import PredictionModelGraph
from pyfemtet.message import Msg


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

        # stop update button
        # noinspection PyAttributeOutsideInit
        self.toggle_update_graph_button = dbc.Checkbox(
            label=Msg.LABEL_AUTO_UPDATE,
            class_name='form-switch',
            id='toggle-update-graph',
            value=True,
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
                        dbc.Col(self.toggle_update_graph_button),
                        dbc.Col(self.interrupt_button, style=DBC_COLUMN_STYLE_RIGHT),
                    ],
                ),
            ]
        )

    def setup_callback(self):
        # setup callback of subpages
        super().setup_callback()

        from pyfemtet.opt.visualization.process_monitor.application import ProcessMonitorApplication
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
        def update_entire_status(*_):
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
        def interrupt_optimization(*_):
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
            def status_change(*_):
                self.application.local_entire_status_int += 10
                for i in range(len(self.application.local_worker_status_int_list)):
                    self.application.local_worker_status_int_list[i] += 10
                raise PreventUpdate

            # increment data
            self.layout.children.append(dbc.Button(children='local_data を増やす', id='debug-button-2'))

            @app.callback(Output(self.interval.id, self.interval.Prop.interval, allow_duplicate=True),
                          Input('debug-button-2', 'n_clicks'),
                          prevent_initial_call=True)
            def add_data(*_):
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

    def __init__(self, title, rel_url, application):
        from pyfemtet.opt.visualization.process_monitor.application import ProcessMonitorApplication
        self.application: ProcessMonitorApplication = None
        super().__init__(title, rel_url, application)

    def setup_component(self):
        # noinspection PyAttributeOutsideInit
        self.interval = dcc.Interval(id='worker-page-interval', interval=1000)

        # noinspection PyAttributeOutsideInit
        self.worker_status_alerts = []
        for i in range(len(self.application.worker_addresses)):
            id_worker_alert = f'worker-status-alert-{i}'
            alert = dbc.Alert('worker status here', id=id_worker_alert, color='dark')
            self.worker_status_alerts.append(alert)

    def setup_layout(self):
        rows = [self.interval]
        rows.extend([dbc.Row([dbc.Col(html.Div(dcc.Loading(alert, id={"type": "loading", "index": i})))]) for i, alert in enumerate(self.worker_status_alerts)])

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
            from pyfemtet.opt._femopt_core import OptimizationStatus

            ret = []

            for worker_address, worker_status_int in zip(self.application.worker_addresses, self.application.local_worker_status_int_list):
                worker_status_message = OptimizationStatus.const_to_str(worker_status_int)
                ret.append(f'{worker_address} is {worker_status_message}')

            ret.extend([self.application.get_status_color(status_int) for status_int in self.application.local_worker_status_int_list])
            ret.append([({} if callback_context.triggered_id is None else no_update) for _ in range(len(self.worker_status_alerts))])

            return tuple(ret)


class PredictionModelPage(AbstractPage):
    """"""

    def __init__(self, title, rel_url, application):
        from pyfemtet.opt.visualization.process_monitor.application import ProcessMonitorApplication
        self.application: ProcessMonitorApplication = None
        super().__init__(title, rel_url, application)

    def setup_component(self):
        self.rsm_graph: PredictionModelGraph = PredictionModelGraph()
        self.add_subpage(self.rsm_graph)

    def setup_layout(self):
        self.layout = self.rsm_graph.layout


class OptunaVisualizerPage(AbstractPage):

    def __init__(self, title, rel_url, application):
        from pyfemtet.opt.visualization.process_monitor.application import ProcessMonitorApplication
        self.application: ProcessMonitorApplication = None
        super().__init__(title, rel_url, application)

    def setup_component(self):
        self.location = dcc.Location(id='optuna-page-location', refresh=True)
        self._layout = html.Div(children=[Msg.DETAIL_PAGE_TEXT_BEFORE_LOADING])
        self.layout = [self.location, self._layout]

    def _setup_layout(self):

        study = self.application.history.create_optuna_study()
        prm_names = self.application.history.prm_names
        obj_names = self.application.history.obj_names

        layout = []

        layout.append(html.H2(Msg.DETAIL_PAGE_HISTORY_HEADER))
        layout.append(html.H4(Msg.DETAIL_PAGE_HISTORY_DESCRIPTION))
        for i, obj_name in enumerate(obj_names):
            fig = optuna.visualization.plot_optimization_history(
                study,
                target=lambda t: t.values[i],
                target_name=obj_name
            )
            layout.append(dcc.Graph(figure=fig, style={'height': '70vh'}))

        layout.append(html.H2(Msg.DETAIL_PAGE_PARALLEL_COOR_HEADER))
        layout.append(html.H4(Msg.DETAIL_PAGE_PARALLEL_COOR_DESCRIPTION))
        for i, obj_name in enumerate(obj_names):
            fig = optuna.visualization.plot_parallel_coordinate(
                study,
                target=lambda t: t.values[i],
                target_name=obj_name
            )
            layout.append(dcc.Graph(figure=fig, style={'height': '70vh'}))

        layout.append(html.H2(Msg.DETAIL_PAGE_CONTOUR_HEADER))
        layout.append(html.H4(Msg.DETAIL_PAGE_CONTOUR_DESCRIPTION))
        for i, obj_name in enumerate(obj_names):
            fig = optuna.visualization.plot_contour(
                study,
                target=lambda t: t.values[i],
                target_name=obj_name
            )
            layout.append(dcc.Graph(figure=fig, style={'height': '90vh'}))

        # import itertools
        # for (i, j) in itertools.combinations(range(len(obj_names)), 2):
        #     fig = optuna.visualization.plot_pareto_front(
        #         study,
        #         targets=lambda t: (t.values[i], t.values[j]),
        #         target_names=[obj_names[i], obj_names[j]],
        #     )
        #     self.graphs.append(dcc.Graph(figure=fig, style={'height': '50vh'}))

        layout.append(html.H2(Msg.DETAIL_PAGE_SLICE_HEADER))
        layout.append(html.H4(Msg.DETAIL_PAGE_SLICE_DESCRIPTION))
        for i, obj_name in enumerate(obj_names):
            fig = optuna.visualization.plot_slice(
                study,
                target=lambda t: t.values[i],
                target_name=obj_name
            )
            layout.append(dcc.Graph(figure=fig, style={'height': '70vh'}))

        return layout

    def setup_callback(self):
        app = self.application.app

        @app.callback(
            Output(self._layout, 'children'),
            Input(self.location, 'pathname'),  # on page load
        )
        def update_page(_):
            if self.application.history is None:
                return Msg.ERR_NO_HISTORY_SELECTED

            if len(self.data_accessor()) == 0:
                return Msg.ERR_NO_FEM_RESULT

            return self._setup_layout()


    def setup_layout(self):
        pass

    def data_accessor(self) -> pd.DataFrame:
        from pyfemtet.opt.visualization.process_monitor.application import ProcessMonitorApplication
        if isinstance(self.application, ProcessMonitorApplication):
            df = self.application.local_data
        else:
            df = self.application.history.get_df()
        return df
