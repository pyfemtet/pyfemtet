import optuna

from dash import Output, Input

from pyfemtet.opt.visualization.history_viewer._wrapped_components import html
from pyfemtet.opt.visualization.history_viewer._wrapped_components import dcc
from pyfemtet.opt.visualization.history_viewer._base_application import AbstractPage
from pyfemtet.opt.visualization.history_viewer._complex_components.pm_graph import *

from pyfemtet._i18n import Msg

__all__ = [
    'PredictionModelPage',
    'OptunaVisualizerPage',
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


def is_iterable(component):
    return hasattr(component, '__len__')


class PredictionModelPage(AbstractPage):
    rsm_graph: PredictionModelGraph

    def __init__(self, title, rel_url, application):
        super().__init__(title, rel_url, application)

    def setup_component(self):
        self.rsm_graph: PredictionModelGraph = PredictionModelGraph()
        self.add_subpage(self.rsm_graph)

    def setup_layout(self):
        self.layout = self.rsm_graph.layout


class OptunaVisualizerPage(AbstractPage):
    location: dcc.Location
    _layout: html.Div

    def __init__(self, title, rel_url, application):
        super().__init__(title, rel_url, application)

    def setup_component(self):
        self.location = dcc.Location(id='optuna-page-location', refresh=True)
        self._layout = html.Div(children=[Msg.DETAIL_PAGE_TEXT_BEFORE_LOADING])
        self.layout = [self.location, self._layout]

    def _setup_layout(self):

        study = self.application.history._create_optuna_study_for_visualization()
        # prm_names = self.application.history.prm_names
        obj_names = self.application.history.obj_names

        layout = list()

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

        layout.append(html.H2(Msg.DETAIL_PAGE_IMPORTANCE_HEADER))
        layout.append(html.H4(Msg.DETAIL_PAGE_IMPORTANCE_DESCRIPTION))
        for i, obj_name in enumerate(obj_names):
            fig = optuna.visualization.plot_param_importances(
                study,
                target=lambda t: t.values[i],
                target_name=obj_name
            )
            import plotly.graph_objects as go
            fig: go.Figure
            fig.update_layout(title=obj_name)
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

            if len(self.application.get_df()) == 0:
                return Msg.ERR_NO_FEM_RESULT

            return self._setup_layout()

    def setup_layout(self):
        pass
