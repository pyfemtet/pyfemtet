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
import plotly.graph_objs as go

# the others
import os
import base64
import json
import numpy as np

from pyfemtet.opt.visualization.complex_components import main_figure_creator
from pyfemtet.opt.visualization.base import PyFemtetApplicationBase, AbstractPage, logger
from pyfemtet.message import Msg


FLEXBOX_STYLE_ALLOW_VERTICAL_FILL = {
    'display': 'flex',
    'flex-direction': 'column',
    'flex-grow': '1',
}


# noinspection PyAttributeOutsideInit
class MainGraph(AbstractPage):
    """"""
    """
    +=================+
    |tab1|tab2|       | <- CardHeader
    |    +------------+
    | +-------------+ |
    | | Loading     | | <- CardBody
    | | +---------+ | |
    | | |  graph <-------- ToolTip
    | | +---------+ | |
    | +-------------+ |
    +=================+

        Data(data-selection)

    """

    def __init__(self):
        self.setup_figure_creator()
        super().__init__()

    def setup_figure_creator(self):
        # setup figure creators
        # list[0] is the default tab
        self.figure_creators = [
            dict(
                tab_id='tab-objective-plot',
                label=Msg.TAB_LABEL_OBJECTIVES,
                creator=main_figure_creator.get_default_figure,
            ),
            dict(
                tab_id='tab-hypervolume-plot',
                label='Hypervolume',
                creator=main_figure_creator.get_hypervolume_plot,
            ),
        ]

    def setup_component(self):
        self.dummy = html.Div(id='main-graph-dummy')
        self.location = dcc.Location(id='main-graph-location', refresh=True)

        # setup header
        self.tab_list = [dbc.Tab(label=d['label'], tab_id=d['tab_id']) for d in self.figure_creators]
        self.tabs = dbc.Tabs(self.tab_list, id='main-graph-tabs')
        self.card_header = dbc.CardHeader(self.tabs)

        # setup body
        self.tooltip = dcc.Tooltip(id='main-graph-tooltip')

        # set kwargs of Graph to reconstruct Graph in ProcessMonitor.
        self.graph_kwargs = dict(
            id='main-graph',
            # animate=True,  # THIS CAUSE THE UPDATED DATA / NOT-UPDATED FIGURE STATE.
            clear_on_unhover=True,
            style={
                # 'flex-grow': '1',  # Uncomment if the plotly's specification if fixed, and we can use dcc.Graph with FlexBox.
                'height': '60vh',
            },
            figure=go.Figure()
        )

        self.graph: dcc.Graph = dcc.Graph(
            **self.graph_kwargs
        )  # Graph make an element by js, so Flexbox CSS cannot apply to the graph (The element can be bigger, but cannot be smaller.)

        self.loading = dcc.Loading(
            children=self.graph,
            id='loading-main-graph',
        )  # Loading make an element that doesn't contain a style attribute, so Flexbox CSS cannot apply to the graph

        self.card_body = dbc.CardBody(
            children=html.Div([self.loading, self.tooltip]),  # If list is passed to CardBody's children, create element that doesn't contain a style attribute, so Flexbox CSS cannot apply to graph
            id='main-graph-card-body',
            # style=FLEXBOX_STYLE_ALLOW_VERTICAL_FILL,
        )

        # setup selection data
        self.selection_data_property = 'data-selection'  # must be starts with "data-"
        self.selection_data = html.Data(id='selection-data', **{self.selection_data_property: {}})

        # set data length
        self.data_length_prop = 'data-df-length'  # must start with "data-"
        self.data_length = html.Data(id='df-length-data', **{self.data_length_prop: None})

    def setup_layout(self):
        # setup component
        self.graph_card = dbc.Card(
            [
                self.dummy,
                self.location,
                self.data_length,
                self.card_header,
                self.card_body,
                self.selection_data,
            ],
            # style=FLEXBOX_STYLE_ALLOW_VERTICAL_FILL,
        )

        self.layout = self.graph_card

    def setup_callback(self):
        # setup callback of subpages
        super().setup_callback()

        app = self.application.app

        # ===== Update Graph =====
        @app.callback(
            Output(self.graph.id, 'figure'),
            Output(self.data_length.id, self.data_length_prop),  # To determine whether Process Monitor should update the graph, the main graph remembers the current amount of data.
            Input(self.tabs.id, 'active_tab'),
            Input(self.dummy, 'children'),
            prevent_initial_call=False,)
        def redraw_main_graph(active_tab_id, _):
            figure, length = self.get_fig_by_tab_id(active_tab_id, with_length=True)
            return figure, length

        # ===== Save Selection =====
        @app.callback(
            Output(self.selection_data.id, self.selection_data_property),
            Input(self.graph.id, 'selectedData'))
        def save_selection(data):
            return data

        # ===== Show Image and Parameter on Hover =====
        @app.callback(
            Output(self.tooltip.id, "show"),
            Output(self.tooltip.id, "bbox"),
            Output(self.tooltip.id, "children"),
            Input(self.graph.id, "hoverData"),)
        def show_hover(hoverData):
            if hoverData is None:
                return False, no_update, no_update

            if self.application.history is None:
                raise PreventUpdate

            # get hover location
            pt = hoverData["points"][0]
            bbox = pt["bbox"]

            # get row of the history from customdata defined in main_figure
            trial = pt['customdata'][0]

            df = self.data_accessor()
            row = df[df['trial'] == trial]

            # create component
            title_component = html.H3(f"trial{trial}", style={"color": "darkblue"})
            img_component = self.create_image_content_if_femtet(trial)
            tbl_component_prm = self.create_formatted_parameter(row)
            tbl_component_obj = self.create_formatted_objective(row)

            # create layout
            description = html.Div([
                title_component,
                tbl_component_prm,
            ])
            tooltip_layout = html.Div([
                html.Div(img_component, style={'display': 'inline-block', 'margin-right': '10px', 'vertical-align': 'top'}),
                html.Div(description, style={'display': 'inline-block', 'margin-right': '10px'}),
                html.Div(tbl_component_obj, style={'display': 'inline-block', 'margin-right': '10px'}),
            ])

            return True, bbox, tooltip_layout

    def create_formatted_parameter(self, row) -> Component:
        metadata = self.application.history.metadata
        pd.options.display.float_format = '{:.4e}'.format
        parameters = row.iloc[:, np.where(np.array(metadata) == 'prm')[0]]
        names = parameters.columns
        values = [f'{value:.3e}' for value in parameters.values.ravel()]
        data = pd.DataFrame(dict(
            parameter=names, value=values
        ))
        table = dash_table.DataTable(
            columns=[{'name': col, 'id': col} for col in data.columns],
            data=data.to_dict('records')
        )
        return table

    def create_formatted_objective(self, row) -> Component:
        metadata = self.application.history.metadata
        pd.options.display.float_format = '{:.4e}'.format
        objectives = row.iloc[:, np.where(np.array(metadata) == 'obj')[0]]
        names = objectives.columns
        values = [f'{value:.3e}' for value in objectives.values.ravel()]
        data = pd.DataFrame(dict(
            objective=names, value=values
        ))
        table = dash_table.DataTable(
            columns=[{'name': col, 'id': col} for col in data.columns],
            data=data.to_dict('records')
        )
        return table

    def create_image_content_if_femtet(self, trial) -> Component:
        img_url = None
        metadata = self.application.history.metadata
        if metadata[0] != '':
            # get img path
            d = json.loads(metadata[0])
            femprj_path = d['femprj_path']
            model_name = d['model_name']
            femprj_result_dir = femprj_path.replace('.femprj', '.Results')
            img_path = os.path.join(femprj_result_dir, f'{model_name}_trial{trial}.jpg')
            if os.path.exists(img_path):
                # create encoded image
                with open(img_path, 'rb') as f:
                    content = f.read()
                encoded_image = base64.b64encode(content).decode('utf-8')
                img_url = 'data:image/jpeg;base64, ' + encoded_image
        return html.Img(src=img_url, style={"width": "200px"}) if img_url is not None else html.Div()

    def get_fig_by_tab_id(self, tab_id, with_length=False):
        # If the history is not loaded, do nothing
        if self.application.history is None:
            raise PreventUpdate

        # else, get creator by tab_id
        if tab_id == 'default':
            creator = self.figure_creators[0]['creator']
        else:
            creators = [d['creator'] for d in self.figure_creators if d['tab_id'] == tab_id]
            if len(creators) == 0:
                raise PreventUpdate
            creator = creators[0]

        # create figure
        df = self.data_accessor()
        fig = creator(self.application.history, df)
        if with_length:
            return fig, len(df)
        else:
            return fig

    def data_accessor(self) -> pd.DataFrame:
        from pyfemtet.opt.visualization.process_monitor.application import ProcessMonitorApplication
        if isinstance(self.application, ProcessMonitorApplication):
            df = self.application.local_data
        else:
            df = self.application.history.get_df()
        return df
