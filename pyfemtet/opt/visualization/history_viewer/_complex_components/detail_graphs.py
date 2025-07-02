from __future__ import annotations

from typing import Literal

# dash components
from pyfemtet.opt.visualization.history_viewer._wrapped_components import dcc, dbc, html

# dash callback
from dash import Output, Input, State, callback_context, no_update, ALL
from dash.exceptions import PreventUpdate

from pyfemtet.logger import get_module_logger

from pyfemtet.opt.history import MAIN_FILTER
from pyfemtet.opt.visualization.history_viewer._base_application import AbstractPage
from pyfemtet.opt.visualization.plotter.parallel_plot_creator import parallel_plot
from pyfemtet.opt.visualization.plotter.contour_creator import contour_creator


class SelectablePlot(AbstractPage):
    location: dcc.Location
    graph: dcc.Graph
    input_items: dcc.Checklist | dcc.RadioItems
    output_items: dcc.Checklist | dcc.RadioItems
    InputItemsClass = dcc.Checklist
    OutputItemsClass = dcc.Checklist
    alerts: html.Div
    input_item_kind: set[Literal['all', 'prm', 'obj', 'cns']] = {'prm'}
    output_item_kind: set[Literal['all', 'prm', 'obj', 'cns']] = {'obj', 'cns'}

    def __init__(self, title='base-page', rel_url='/', application=None,
                 location=None):
        self.location = location
        super().__init__(title, rel_url, application)

    @property
    def plot_title(self) -> str:
        raise NotImplementedError

    def setup_layout(self):

        self.layout = dbc.Container([
            # ----- hidden -----
            dbc.Row([self.location]),
            dbc.Row([html.H2(self.plot_title)]),

            # ----- visible -----
            dbc.Row(
                [
                    dbc.Col(dbc.Spinner(self.graph)),
                    dbc.Col(
                        [dbc.Row(self.input_items), dbc.Row(self.output_items)],
                        md=2
                    )
                ],
            ),
            dbc.Row([self.alerts])
        ])

    def setup_component(self):

        if self.location is None:
            self.location = dcc.Location(id='parallel-plot-location', refresh=True)

        # graph
        self.graph = dcc.Graph()

        # checklist
        self.input_items = self.InputItemsClass(options=[])
        self.output_items = self.OutputItemsClass(options=[])

        # alert
        self.alerts = html.Div()

    def _check_precondition(self, logger):

        if callback_context.triggered_id is None:
            logger.debug('PreventUpdate. No trigger.')
            raise PreventUpdate

        if self.application is None:
            logger.debug('PreventUpdate. No application.')
            raise PreventUpdate

        if self.application.history is None:
            logger.debug('PreventUpdate. No history.')
            raise PreventUpdate

        history = self.application.history

        df = self.application.get_df()
        main_df = self.application.get_df(MAIN_FILTER)
        if len(df) == 0:
            logger.debug('PreventUpdate. No df.')
            raise PreventUpdate

        return history, df, main_df

    @staticmethod
    def _return_checklist_options_and_value(history, types) -> tuple[list[dict], list[str]]:

        keys = []

        if 'all' in types:
            keys.extend(history.prm_names)
            keys.extend(history.obj_names)
            keys.extend(history.cns_names)

        if 'prm' in types:
            keys.extend(history.prm_names)
        if 'obj' in types:
            keys.extend(history.obj_names)
        if 'cns' in types:
            keys.extend(history.cns_names)

        return [dict(label=key, value=key) for key in keys], keys

    def _return_input_checklist_options_and_value(self, history):
        return self._return_checklist_options_and_value(history, self.input_item_kind)

    def _return_output_checklist_options_and_value(self, history):
        return self._return_checklist_options_and_value(history, self.output_item_kind)

    def setup_update_plot_checklist_callback(self):

        @self.application.app.callback(
            Output(self.input_items, 'options'),
            Output(self.input_items, 'value'),
            Input(self.location, 'pathname'),  # on page load
        )
        def update_plot_input_checklist(_):

            logger_name = f'opt.{type(self).__name__}.update_plot_input_checklist()'

            logger = get_module_logger(
                logger_name,
                debug=False,
            )

            logger.debug('callback fired!')

            # ----- preconditions -----

            history, _, _ = self._check_precondition(logger)

            # ----- main -----
            options, value = self._return_input_checklist_options_and_value(history)

            if isinstance(self.input_items, dcc.RadioItems):
                value = value[0]

            return options, value

        @self.application.app.callback(
            Output(self.output_items, 'options'),
            Output(self.output_items, 'value'),
            Input(self.location, 'pathname'),  # on page load
        )
        def update_plot_output_checklist(_):

            logger_name = f'opt.{type(self).__name__}.update_plot_output_checklist()'

            logger = get_module_logger(
                logger_name,
                debug=False,
            )

            logger.debug('callback fired!')

            # ----- preconditions -----

            history, _, _ = self._check_precondition(logger)

            # ----- main -----
            options, value = self._return_output_checklist_options_and_value(history)

            logger.debug(value)
            if isinstance(self.output_items, dcc.RadioItems):
                value = value[0]
            logger.debug(value)

            return options, value

    def setup_update_plot_graph_callback(self):

        @self.application.app.callback(
            # graph output
            Output(self.graph, 'figure'),
            Output(self.alerts, 'children'),
            # checklist input
            inputs=dict(
                selected_input_values=Input(self.input_items, 'value'),
                selected_output_values=Input(self.output_items, 'value'),
            ),
        )
        def update_plot_graph(
                selected_input_values: list[str] | str,
                selected_output_values: list[str] | str,
        ):

            logger_name = f'opt.{type(self).__name__}.update_plot_graph()'

            logger = get_module_logger(
                logger_name,
                debug=False,
            )

            logger.debug('callback fired!')

            # ----- preconditions -----

            history, df, main_df = self._check_precondition(logger)

            # null selected values
            if selected_input_values is None:
                logger.debug('No input items.')
                return no_update, [dbc.Alert('No input items.', color='danger')]

            if selected_output_values is None:
                logger.debug('No output items.')
                return no_update, [dbc.Alert('No output items.', color='danger')]

            # type correction
            if isinstance(selected_input_values, str):
                selected_input_values = [selected_input_values]
            if isinstance(selected_output_values, str):
                selected_output_values = [selected_output_values]

            # nothing selected
            # selected_values = selected_input_values + selected_output_values
            # if len(selected_values) == 0:
            #     logger.debug('No items are selected.')
            #     return no_update, [dbc.Alert('No items are selected.', color='danger')]
            if len(selected_input_values) == 0:
                logger.debug('No input items are selected.')
                return no_update, [dbc.Alert('No input items are selected.', color='danger')]
            if len(selected_output_values) == 0:
                logger.debug('No output items are selected.')
                return no_update, [dbc.Alert('No output items are selected.', color='danger')]

            # ----- main -----
            used_df = self.make_used_df(history, df, main_df, selected_input_values, selected_output_values)
            assert len(used_df) > 0
            assert len(used_df.columns) > 0

            fig = self.create_plot(used_df)

            return fig, []

    def setup_callback(self):
        self.setup_update_plot_checklist_callback()
        self.setup_update_plot_graph_callback()

    @staticmethod
    def make_used_df(history, df, main_df, selected_input_values, selected_output_values):
        # NotImplementedError でもいいが、汎用的なので

        columns = [
            col for col in history.prm_names + history.obj_names + history.cns_names
            if col in selected_input_values + selected_output_values
        ]

        use_df = main_df[columns]

        return use_df

    @staticmethod
    def create_plot(used_df):
        raise NotImplementedError


class ParallelPlot(SelectablePlot):

    plot_title = 'parallel plot'

    @staticmethod
    def create_plot(used_df):
        return parallel_plot(used_df)


class ContourPlot(SelectablePlot):

    plot_title = 'contour plot'
    OutputItemsClass = dcc.RadioItems

    @staticmethod
    def create_plot(used_df):
        return contour_creator(used_df)


class SensPlot(SelectablePlot):

    plot_title = 'sensitivity plot'
