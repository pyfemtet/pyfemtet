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

from pyfemtet.opt.visualization.complex_components import rsm_graph_creator
from pyfemtet.opt.visualization.base import PyFemtetApplicationBase, AbstractPage, logger


FLEXBOX_STYLE_ALLOW_VERTICAL_FILL = {
    'display': 'flex',
    'flex-direction': 'column',
    'flex-grow': '1',
}


# noinspection PyAttributeOutsideInit
class RSMGraph(AbstractPage):
    """"""
    """
    +=================+
    |3d|scatterplot|  | <- CardHeader
    |  +--------------+
    | +-------------+ |
    | | Loading     | | <- CardBody
    | | +---------+ | |
    | | |  graph <-------- ToolTip
    | | +---------+ | |
    | +-------------+ |
    |                 |
    | +-------------+ |
    | |p1, p2, obj <------ Dropdown (axis selection)
    | |             | |
    | | p3 --o----- | |
    | | p4 -----o-- <----- Slider(s) (choose the value of the others)
    | |             | |
    | +-------------+ |
    |                 |
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
                label='data',
                creator=rsm_graph_creator.rsm_3d_creator,
            ),
        ]

    def setup_component(self):
        self.location = dcc.Location(id='rsm-graph-location', refresh=True)

        # setup header
        self.tab_list = [dbc.Tab(label=d['label'], tab_id=d['tab_id']) for d in self.figure_creators]
        self.tabs = dbc.Tabs(self.tab_list)
        self.card_header = dbc.CardHeader(self.tabs)

        # setup body
        # self.tooltip = dcc.Tooltip()

        # set kwargs of Graph to reconstruct Graph in ProcessMonitor.
        self.graph_kwargs = dict(
            # animate=True,  # THIS CAUSE THE UPDATED DATA / NOT-UPDATED FIGURE STATE.
            clear_on_unhover=True,
            style={
                'height': '60vh',
            },
            figure=go.Figure()
        )

        self.graph: dcc.Graph = dcc.Graph(
            **self.graph_kwargs
        )

        self.loading = dcc.Loading(
            children=self.graph,
        )

        self.card_body = dbc.CardBody(
            children=html.Div([self.loading]),  # children=html.Div([self.loading, self.tooltip]),
        )

        # setup selection data
        # self.selection_data_property = 'data-selection'  # must be starts with "data-"
        # self.selection_data = html.Data(id='selection-data', **{self.selection_data_property: {}})

        # set data length (to notify data updated to application)
        self.data_length_prop = 'data-df-length'  # must start with "data-"
        self.data_length = html.Data(id='df-length-data', **{self.data_length_prop: None})

        # Dropdowns
        self.prm1_items = []
        self.prm2_items = []
        self.obj_items = []
        self.axis1_prm_dropdown = dbc.DropdownMenu(self.prm1_items)
        self.axis2_prm_dropdown = dbc.DropdownMenu(self.prm2_items)
        self.axis3_obj_dropdown = dbc.DropdownMenu(self.obj_items)
        self.axis_controllers = [
            self.axis1_prm_dropdown,
            self.axis2_prm_dropdown,
            self.axis3_obj_dropdown,
        ]

        # sliders
        self.slider_container = html.Div()

    def setup_layout(self):
        # setup component
        self.graph_card = dbc.Card(
            [
                self.location,
                self.data_length,
                self.card_header,
                self.card_body,
                # self.selection_data,
            ],
            # style=FLEXBOX_STYLE_ALLOW_VERTICAL_FILL,
        )

        self.layout = dbc.Container(
            children=[
                dbc.Row(self.graph_card),
                dbc.Row(self.axis_controllers),
                dbc.Row(self.slider_container),
            ]
        )

    def setup_callback(self):
        # setup callback of subpages
        super().setup_callback()

        app = self.application.app

        # # ===== Update Graph =====
        # @app.callback(
        #     Output(self.graph.id, 'figure'),
        #     Output(self.data_length.id, self.data_length_prop),  # To determine whether Process Monitor should update the graph, the main graph remembers the current amount of data.
        #     Input(self.tabs.id, 'active_tab'),
        #     Input(self.dummy, 'children'),
        #     prevent_initial_call=False,)
        # def redraw_main_graph(active_tab_id, _):
        #     logger.debug('====================')
        #     logger.debug(f'redraw_main_graph called by {callback_context.triggered_id}')
        #     figure, length = self.get_fig_by_tab_id(active_tab_id, with_length=True)
        #     return figure, length

        # # ===== Save Selection =====
        # @app.callback(
        #     Output(self.selection_data.id, self.selection_data_property),
        #     Input(self.graph.id, 'selectedData'))
        # def save_selection(data):
        #     return data

        # # ===== Show Image and Parameter on Hover =====
        # @app.callback(
        #     Output(self.tooltip.id, "show"),
        #     Output(self.tooltip.id, "bbox"),
        #     Output(self.tooltip.id, "children"),
        #     Input(self.graph.id, "hoverData"),)
        # def show_hover(hoverData):
        #     if hoverData is None:
        #         return False, no_update, no_update
        #
        #     if self.application.history is None:
        #         raise PreventUpdate
        #
        #     # get hover location
        #     pt = hoverData["points"][0]
        #     bbox = pt["bbox"]
        #
        #     # get row of the history from customdata defined in main_figure
        #     trial = pt['customdata'][0]
        #
        #     df = self.data_accessor()
        #     row = df[df['trial'] == trial]
        #
        #     # create component
        #     title_component = html.H3(f"trial{trial}", style={"color": "darkblue"})
        #     img_component = self.create_image_content_if_femtet(trial)
        #     tbl_component = self.create_formatted_parameter(row)
        #
        #     # create layout
        #     description = html.Div([
        #         title_component,
        #         tbl_component,
        #     ])
        #     tooltip_layout = html.Div([
        #         html.Div(img_component, style={'display': 'inline-block', 'margin-right': '10px', 'vertical-align': 'top'}),
        #         html.Div(description, style={'display': 'inline-block', 'margin-right': '10px'})
        #     ])
        #
        #     return True, bbox, tooltip_layout

        # ===== setup dropdown from history =====
        for i_, obj_name_ in enumerate(self.application.history.obj_names):
            self.obj_items.append(
                dbc.DropdownMenuItem(
                    children=obj_name_,
                    id=f'obj-dropdown-item-{i_}',
                )
            )
        for i_, prm_name_ in enumerate(self.application.history.prm_names):
            self.prm1_items.append(
                dbc.DropdownMenuItem(
                    children=prm_name_,
                    id=f'prm1-dropdown-item-{i_}',
                )
            )
            self.prm2_items.append(
                dbc.DropdownMenuItem(
                    children=prm_name_,
                    id=f'prm2-dropdown-item-{i_}',
                )
            )

        # ===== update axis1-prm-dropdown =====
        @app.callback(
            Output(self.axis1_prm_dropdown, 'label', allow_duplicate=True),
            Input(self.location, self.location.Prop.pathname),
            [Input(item, 'n_clicks') for item in self.prm1_items],
            State(self.axis2_prm_dropdown, 'label'),
            prevent_initial_call=True,
        )
        def update_prm1_dropdown_menu_label(*args):
            prm2_label = args[-1]

            # 一応
            if callback_context.triggered_id is None:
                raise PreventUpdate

            # load history
            if self.application.history is None:
                return 'History is not selected.'
            prm_names = self.application.history.prm_names

            # 1st parameter on loaded
            if callback_context.triggered_id == self.location.id:
                return prm_names[0]

            # clicked
            for i, item in enumerate(self.prm1_items):
                if item.id == callback_context.triggered_id:
                    if prm_names[i] != prm2_label:
                        return prm_names[i]
                    else:
                        logger.error('Cannot select same parameter')
                        raise PreventUpdate

            # something wrong
            # logger.debug('something wrong in `update_dropdown_menu_label`')
            raise PreventUpdate

        # ===== update axis2-prm-dropdown =====
        @app.callback(
            Output(self.axis2_prm_dropdown, 'label', allow_duplicate=True),
            Output(self.axis2_prm_dropdown, 'hidden', allow_duplicate=True),
            Input(self.location, self.location.Prop.pathname),
            [Input(item, 'n_clicks') for item in self.prm2_items],
            State(self.axis1_prm_dropdown, 'label'),
            prevent_initial_call=True,
        )
        def update_prm2_dropdown_menu_label(*args):
            prm1_label = args[-1]

            # 一応
            if callback_context.triggered_id is None:
                raise PreventUpdate

            # load history
            if self.application.history is None:
                return 'History is not selected.', no_update
            prm_names = self.application.history.prm_names

            # disable axis2 if only 1 parameter optimization
            if len(prm_names) == 1:
                return no_update, True

            # 2nd parameter on loaded
            if callback_context.triggered_id == self.location.id:
                return prm_names[1], False

            # clicked
            for i, item in enumerate(self.prm2_items):
                if item.id == callback_context.triggered_id:
                    if prm_names[i] != prm1_label:
                        return prm_names[i], False
                    else:
                        logger.error('Cannot select same parameter')
                        raise PreventUpdate

            # something wrong
            # logger.debug('something wrong in `update_dropdown_menu_label`')
            raise PreventUpdate

        # ===== update axis3-obj-dropdown =====
        @app.callback(
            Output(self.axis3_obj_dropdown, 'label', allow_duplicate=True),
            Input(self.location, self.location.Prop.pathname),
            [Input(item, 'n_clicks') for item in self.obj_items],
            prevent_initial_call=True,
        )
        def update_obj_dropdown_menu_label(*args):
            # 一応
            if callback_context.triggered_id is None:
                raise PreventUpdate

            if self.application.history is None:
                return 'History is not selected.'

            obj_names = self.application.history.obj_names

            # 1st objective on loaded
            if callback_context.triggered_id == self.location.id:
                return obj_names[0]

            # clicked
            for i, item in enumerate(self.obj_items):
                if item.id == callback_context.triggered_id:
                    return obj_names[i]

            # something wrong
            # logger.debug('something wrong in `update_dropdown_menu_label`')
            raise PreventUpdate

        # ===== setup sliders =====
        @app.callback(
            Output(self.slider_container, 'children'),
            Input(self.axis1_prm_dropdown, 'label'),
            Input(self.axis2_prm_dropdown, 'label'),
            prevent_initial_call=True,
        )
        def update_sliders(label1, label2):
            # Just in case
            if callback_context.triggered_id is None:
                raise PreventUpdate

            # load history
            if self.application.history is None:
                return 'History is not selected.', no_update
            prm_names: list = list(self.application.history.prm_names)  # shallow copy

            # get remaining parameters
            prm_names.remove(label1) if label1 in prm_names else None
            prm_names.remove(label2) if label2 in prm_names else None

            out = []
            for prm_name in prm_names:
                # get ub and lb
                lb_column = prm_name + '_lower_bound'
                ub_column = prm_name + '_upper_bound'
                # get minimum lb and maximum ub
                df = self.data_accessor()
                lb = df[lb_column].min()
                ub = df[ub_column].max()
                # if lb or ub is not specified, use value instead
                lb = df[prm_name].min() if np.isnan(lb) else lb
                ub = df[prm_name].max() if np.isnan(ub) else ub
                # create slider
                out.append(
                    dbc.Stack(
                        children=[
                            html.Div(f'{prm_name}: '),
                            dcc.Slider(
                                lb,
                                ub,
                                marks=None,
                                value=(lb + ub) / 2,
                                id=f'{prm_name}-slider',
                                tooltip={"placement": "bottom", "always_visible": True},
                            )
                        ]
                    )
                )

            return out



    def create_formatted_parameter(self, row) -> Component:
        metadata = self.application.history.metadata
        pd.options.display.float_format = '{:.4e}'.format
        parameters = row.iloc[:, np.where(np.array(metadata) == 'prm')[0]]
        names = parameters.columns
        values = [f'{value:.3e}' for value in parameters.values.ravel()]
        data = pd.DataFrame(dict(
            name=names, value=values
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
            df = self.application.history.local_data
        return df
