from __future__ import annotations

# dash components
from pyfemtet.opt.visualization.history_viewer._wrapped_components import dcc, dbc, html

# dash callback
from dash import Output, Input, callback_context
from dash.exceptions import PreventUpdate

from pyfemtet.logger import get_module_logger

from pyfemtet.opt.history import MAIN_FILTER
from pyfemtet.opt.visualization.history_viewer._base_application import AbstractPage
from pyfemtet.opt.visualization.history_viewer._complex_components.detail_graphs import (
    ParallelPlot,
    ContourPlot,
    ImportancePlot,
    SlicePlot,
    HistoryPlot,
)


class DetailPage(AbstractPage):
    location: dcc.Location
    alerts: html.Div
    parallel_plot: ParallelPlot
    contour_plot: ContourPlot
    importance_plot: ImportancePlot
    slice_plot: SlicePlot
    history_plot: HistoryPlot

    def setup_component(self):

        self.location = dcc.Location(id='new-detail-location', refresh=True)

        # alerts
        self.alerts = html.Div(id='new-detail-alerts')

        # graphs
        self.parallel_plot = ParallelPlot(location=self.location)
        self.add_subpage(self.parallel_plot)
        self.contour_plot = ContourPlot(location=self.location)
        self.add_subpage(self.contour_plot)
        self.importance_plot = ImportancePlot(location=self.location)
        self.add_subpage(self.importance_plot)
        self.slice_plot = SlicePlot(location=self.location)
        self.add_subpage(self.slice_plot)
        self.history_plot = HistoryPlot(location=self.location)
        self.add_subpage(self.history_plot)

    def setup_layout(self):

        # title
        title = html.H1('Detail Plot Graphs')

        # layout
        self.layout = dbc.Container([
            dbc.Row([self.location]),
            dbc.Row([title]),
            dbc.Row([html.Hr()]),
            dbc.Row([self.alerts]),
            dbc.Row([self.importance_plot.layout]),
            dbc.Row([self.history_plot.layout]),
            dbc.Row([self.slice_plot.layout]),
            dbc.Row([self.contour_plot.layout]),
            dbc.Row([self.parallel_plot.layout]),
        ])

    def setup_callback(self):
        super().setup_callback()  # setup callback of subpages

        app = self.application.app

        # ===== update alert =====
        @app.callback(
            Output(self.alerts.id, 'children'),
            Input(self.location.id, 'pathname'),  # on page load
        )
        def update_alerts_new_detail(_):

            logger = get_module_logger(
                'opt.update_alerts_new_detail',
                debug=False,
            )

            # ----- preconditions -----

            if callback_context.triggered_id is None:
                logger.debug('PreventUpdate. No trigger.')
                raise PreventUpdate

            if self.application is None:
                logger.debug('PreventUpdate. No application.')
                raise PreventUpdate

            if self.application.history is None:
                logger.debug('PreventUpdate. No history.')
                return [dbc.Alert('No history.', color='danger')]

            # df = self.application.get_df()
            main_df = self.application.get_df(MAIN_FILTER)
            if len(main_df) == 0:
                logger.debug('PreventUpdate. No df.')
                return [dbc.Alert('No data.', color='danger')]

            return []
