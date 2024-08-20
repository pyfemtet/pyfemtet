import plotly.graph_objects as go

from dash import dcc
import dash_bootstrap_components as dbc

from pyfemtet.opt.dash_ui.base_application import BaseComplexComponent


class FlexboxGraph(BaseComplexComponent):
    # noinspection PyUnresolvedReferences
    """Plotly figure object that follows the flexbox size order.

            Attributes:
                loading_graph (dcc.Loading): Contains a graph object.
                graph (dcc.Graph): Actual graph object.

                layout (dbc.Row): Contains a loading_graph object.
            ::

                +--------------+
                |+------------+|
                ||+----------+||
                |||          |<--- graph (dcc.Graph)
                |||          ||<-- loading_graph (dcc.Loading)
                ||+----------+||
                |+------------+|<- layout = dbc.Row(dbc.Col())
                +--------------+


            Examples:
                >>> graph = FlexboxGraph(application)  # doctest: +SKIP
                >>> container_style = {'height': '90vh'}  # doctest: +SKIP
                >>> container_style.update(FlexboxGraph.CONTAINER_STYLE)  # doctest: +SKIP
                >>> container = dbc.Container(
                ...     children=[graph.layout, ...],
                ...     style=container_style,  # make children to fill flexbox and shrink itself if zoomed in
                ...     fluid=True,
                ... )  # doctest: +SKIP
                ...

            """

    CONTAINER_STYLE = {'display': 'flex', 'flex-direction': 'column', 'overflow': 'hidden', }

    def __init__(self, application, with_loading=True, graph_style=None):
        self.with_loading = with_loading
        self.graph_style = graph_style or {}
        super().__init__(application)

    # noinspection PyAttributeOutsideInit
    def setup_components(self):
        # declare
        self.graph: dcc.Graph = ...
        self.loading_graph: dcc.Loading = ...

        # init graph
        graph_style = {'width': "100%", 'height': "100%", }  # fill flexbox
        graph_style.update(self.graph_style)
        self.graph = dcc.Graph(
            figure=go.Figure(),
            style=graph_style,
        )

        # init loading
        if self.with_loading:
            self.loading_graph = dcc.Loading(
                self.graph,
                style={'width': "100%", 'height': "100%", },  # fill flexbox
                parent_style={'width': "100%", 'height': "100%", },  # fill flexbox
                overlay_style={'width': "100%", 'height': "100%", },  # fill flexbox
            )
        else:
            self.loading_graph = None

    def setup_callbacks(self):
        super().setup_callbacks()

    def setup_layout(self):

        if self.with_loading:
            content = self.loading_graph
        else:
            content = self.graph

        self.layout = dbc.Row(
            children=[
                dbc.Col(
                    children=[content],
                    style={'display': 'flex', 'flex-direction': 'column', 'flex': '1 1 auto',
                           'overflow': 'hidden', },  # make children to fill flexbox and shrink itself if zoomed in
                )
            ],
            style={'display': 'flex', 'flex-direction': 'column', 'flex': '1 1 auto', 'overflow': 'hidden', },
            # make children to fill flexbox and shrink itself if zoomed in
        )


if __name__ == '__main__':
    from pyfemtet.opt.dash_ui.base_application import BaseMultiPageApplication

    g_application = BaseMultiPageApplication()


    class Page(BaseComplexComponent):

        def setup_components(self):
            pass

        def setup_callbacks(self):
            pass

        def setup_layout(self):
            g_graph_1 = FlexboxGraph(self.application, graph_style={'height': '30vh'})
            g_graph_2 = FlexboxGraph(self.application)
            container_style = {'height': '90hv'}
            container_style.update(FlexboxGraph.CONTAINER_STYLE)
            self.layout = dbc.Container(
                children=[g_graph_1.layout, g_graph_2.layout],
                style=container_style,
                fluid=True,
            )


    page = Page(g_application)

    g_application.add_page(page, 'home', '/')

    g_application.run(debug=True)
