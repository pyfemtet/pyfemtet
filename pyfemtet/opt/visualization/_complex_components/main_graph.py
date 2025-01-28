# type hint
from dash.development.base_component import Component

# callback
from dash import Output, Input, State, no_update, callback_context, ALL
from dash.exceptions import PreventUpdate

# components
from dash import dash_table
from pyfemtet.opt.visualization._wrapped_components import html, dcc, dbc

# graph
import pandas as pd
# import plotly.express as px
import plotly.graph_objs as go

# the others
import os
import base64
import json
import numpy as np

from pyfemtet.opt.visualization._complex_components import main_figure_creator
from pyfemtet.opt.visualization._base import AbstractPage, logger
from pyfemtet._message import Msg


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
    +-----------------+
    |                 | <- CardFooter (depends on tab)
    +=================+

        Data(data-selection)

    """

    TAB_ID_OBJECTIVE_PLOT = 'tab-objectives-plot'
    TAB_ID_HYPERVOLUME_PLOT = 'tab-hypervolume-plot'

    def __init__(self):
        self.setup_figure_creator()
        super().__init__()

    def setup_figure_creator(self):
        # setup figure creators
        # list[0] is the default tab
        self.figure_creators = [
            dict(
                tab_id='tab-objectives-scatterplot',
                label=Msg.TAB_LABEL_OBJECTIVE_SCATTERPLOT,
                creator=main_figure_creator.get_default_figure,
            ),
            dict(
                tab_id=self.TAB_ID_OBJECTIVE_PLOT,
                label=Msg.TAB_LABEL_OBJECTIVE_PLOT,
                creator=main_figure_creator.get_objective_plot,
            ),
            dict(
                tab_id=self.TAB_ID_HYPERVOLUME_PLOT,
                label='Hypervolume',
                creator=main_figure_creator.get_hypervolume_plot,
            ),
        ]

    def setup_component(self):
        self.dummy = html.Div(id='main-graph-dummy')
        self.location = dcc.Location(id='main-graph-location', refresh=True)

        # setup header
        tab_list = []
        for d in self.figure_creators:
            is_objective_plot = d['tab_id'] == self.TAB_ID_OBJECTIVE_PLOT  # Objective plot tab is only shown in obj_names > 3. Set this via callback.
            is_hypervolume_plot = d['tab_id'] == self.TAB_ID_HYPERVOLUME_PLOT  # same above

            if is_objective_plot or is_hypervolume_plot:
                style = {'display': 'none'}
            else:
                style = None

            tab_list.append(
                dbc.Tab(
                    label=d['label'],
                    tab_id=d['tab_id'],
                    tab_style=style,
                )
            )
        self.tab_list = tab_list
        self.tabs = dbc.Tabs(self.tab_list, id='main-graph-tabs')
        self.card_header = dbc.CardHeader(self.tabs)

        # setup body
        self.tooltip = dcc.Tooltip(id='main-graph-tooltip')

        # setup footer components for objective plot
        self.axis1_dropdown = dbc.DropdownMenu()
        self.axis2_dropdown = dbc.DropdownMenu()
        self.axis3_dropdown = dbc.DropdownMenu()
        self.switch_3d = dbc.Checklist(
            options=[
                dict(
                    label="3D",
                    disabled=False,
                    value=False,
                )
            ],
            switch=True,
            value=[],
        )
        self.objective_plot_controller: html.Div = html.Div(
            [
                self.switch_3d,
                dbc.Stack(
                    children=['X axis', self.axis1_dropdown],
                    direction='horizontal', gap=2),
                dbc.Stack(
                    children=['Y axis', self.axis2_dropdown],
                    direction='horizontal', gap=2),
                dbc.Stack(
                    children=['Z axis', self.axis3_dropdown],
                    direction='horizontal', gap=2),
            ]
        )

        # setup footer
        self.card_footer: dbc.CardFooter = dbc.CardFooter(
            self.objective_plot_controller
        )

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
                self.card_footer,
                self.selection_data,
            ],
            # style=FLEXBOX_STYLE_ALLOW_VERTICAL_FILL,
        )

        self.layout = self.graph_card

    def setup_callback(self):
        # setup callback of subpages
        super().setup_callback()

        app = self.application.app

        # ===== Change visibility of plot tab =====
        objective_plot_tab = [t for t in self.tab_list if t.tab_id == self.TAB_ID_OBJECTIVE_PLOT][0]
        hypervolume_plot_tab = [t for t in self.tab_list if t.tab_id == self.TAB_ID_HYPERVOLUME_PLOT][0]
        @app.callback(
            output=(
                Output(objective_plot_tab, 'tab_style'),
                Output(hypervolume_plot_tab, 'tab_style'),
            ),
            inputs=dict(
                _=(
                    Input(self.tabs, 'active_tab'),
                    Input(self.location, 'pathname'),
                ),
                current_styles=dict(
                    obj=State(objective_plot_tab, 'tab_style'),
                    hv=State(hypervolume_plot_tab, 'tab_style'),
                ),
            ),
            prevent_initial_call=True,
        )
        def set_disabled(_, current_styles):
            obj_style: dict = no_update
            hv_style: dict = no_update

            # load history
            if self.application.history is None:
                raise PreventUpdate
            obj_names = self.application.history.obj_names

            # show objective plot with 3 or more objectives
            if len(obj_names) < 3:
                if 'display' not in current_styles['obj'].keys():
                    obj_style = current_styles['obj']
                    obj_style.update({'display': 'none'})
            else:
                if 'display' in current_styles['obj'].keys():
                    obj_style = current_styles['obj']
                    obj_style.pop('display')

            # show hypervolume plot with 2 or more objectives
            if len(obj_names) < 2:
                if 'display' not in current_styles['hv'].keys():
                    hv_style = current_styles['hv']
                    hv_style.update({'display': 'none'})
            else:
                if 'display' in current_styles['hv'].keys():
                    hv_style = current_styles['hv']
                    hv_style.pop('display')

            return obj_style, hv_style

        # ===== Change visibility of dropdown menus =====
        @app.callback(
            Output(self.objective_plot_controller, 'hidden'),
            Input(self.tabs, 'active_tab'),
            prevent_initial_call=True,
        )
        def invisible_controller(active_tab):
            return active_tab != self.TAB_ID_OBJECTIVE_PLOT

        # ===== Change accessibility of axis3 dropdown =====
        @app.callback(
            Output(self.axis3_dropdown, 'disabled'),
            Input(self.switch_3d, 'value'),
            Input(self.location, 'pathname'),
            prevent_initial_call=True,
        )
        def disable_axis_3_dropdown_menu(is_3d, _):
            return not is_3d

        # ===== Initialize Dropdown menus =====
        @app.callback(
            output=dict(
                items_list=[
                    Output(self.axis1_dropdown, 'children'),
                    Output(self.axis2_dropdown, 'children'),
                    Output(self.axis3_dropdown, 'children'),
                ],
                labels=[
                    Output(self.axis1_dropdown, 'label'),
                    Output(self.axis2_dropdown, 'label'),
                    Output(self.axis3_dropdown, 'label'),
                ]
            ),
            inputs=[Input(self.location, 'pathname')],
            prevent_initial_call=True,
        )
        def init_dropdown_menus(*_, **__):
            # just in case
            if callback_context.triggered_id is None:
                raise PreventUpdate

            # load history
            if self.application.history is None:
                logger.error(Msg.ERR_NO_HISTORY_SELECTED)
                raise PreventUpdate

            # assert 3 or more objectives
            obj_names = self.application.history.obj_names
            if len(obj_names) < 3:
                raise PreventUpdate

            # add dropdown item to dropdown
            axis1_dropdown_items = []
            axis2_dropdown_items = []
            axis3_dropdown_items = []

            for i, obj_name in enumerate(obj_names):
                axis1_dropdown_items.append(
                    dbc.DropdownMenuItem(
                        children=obj_name,
                        id={'type': 'objective-axis1-dropdown-menu-item', 'index': obj_name},
                    )
                )

                axis2_dropdown_items.append(
                    dbc.DropdownMenuItem(
                        children=obj_name,
                        id={'type': 'objective-axis2-dropdown-menu-item', 'index': obj_name},
                    )
                )

                axis3_dropdown_items.append(
                    dbc.DropdownMenuItem(
                        children=obj_name,
                        id={'type': 'objective-axis3-dropdown-menu-item', 'index': obj_name},
                    )
                )

            items_list = [axis1_dropdown_items, axis2_dropdown_items, axis3_dropdown_items]

            ret = dict(
                items_list=items_list,
                labels=obj_names[:3],
            )

            return ret

        # ===== Update Dropdown menus =====
        @app.callback(
            output=dict(
                label_1=Output(self.axis1_dropdown, 'label', allow_duplicate=True),  # label of dropdown
                label_2=Output(self.axis2_dropdown, 'label', allow_duplicate=True),
                label_3=Output(self.axis3_dropdown, 'label', allow_duplicate=True),
            ),
            inputs=dict(
                _=(  # when the dropdown item is clicked
                    Input({'type': 'objective-axis1-dropdown-menu-item', 'index': ALL}, 'n_clicks'),
                    Input({'type': 'objective-axis2-dropdown-menu-item', 'index': ALL}, 'n_clicks'),
                    Input({'type': 'objective-axis3-dropdown-menu-item', 'index': ALL}, 'n_clicks'),
                ),
                current_labels=dict(  # for exclusive selection
                    label_1=State(self.axis1_dropdown, 'label'),
                    label_2=State(self.axis2_dropdown, 'label'),
                    label_3=State(self.axis3_dropdown, 'label'),
                ),
            ),
            prevent_initial_call=True,
        )
        def update_dropdowns(_, current_labels):
            # just in case
            if callback_context.triggered_id is None:
                raise PreventUpdate

            # load history
            if self.application.history is None:
                logger.error(Msg.ERR_NO_HISTORY_SELECTED)
                raise PreventUpdate
            obj_names = self.application.history.obj_names

            if len(obj_names) < 3:
                raise PreventUpdate

            # default return values
            ret = dict(
                label_1=no_update,
                label_2=no_update,
                label_3=no_update,
            )

            # ===== update dropdown label =====
            new_label = callback_context.triggered_id['index']

            # example of triggerd_id: {'index': 'max_displacement', 'type': 'objective-axis1-dropdown-menu-item'}
            if callback_context.triggered_id['type'] == 'objective-axis1-dropdown-menu-item':
                ret["label_1"] = new_label
                if current_labels["label_2"] == new_label:
                    ret["label_2"] = current_labels["label_1"]
                if current_labels["label_3"] == new_label:
                    ret["label_3"] = current_labels["label_1"]
            if callback_context.triggered_id['type'] == 'objective-axis2-dropdown-menu-item':
                ret["label_2"] = new_label
                if current_labels["label_3"] == new_label:
                    ret["label_3"] = current_labels["label_2"]
                if current_labels["label_1"] == new_label:
                    ret["label_1"] = current_labels["label_2"]
            if callback_context.triggered_id['type'] == 'objective-axis3-dropdown-menu-item':
                ret["label_3"] = new_label
                if current_labels["label_1"] == new_label:
                    ret["label_1"] = current_labels["label_3"]
                if current_labels["label_2"] == new_label:
                    ret["label_2"] = current_labels["label_3"]

            return ret

        # ===== Update Graph (by tab selection or dropdown changed) =====
        @app.callback(
            Output(self.graph.id, 'figure'),
            Output(self.data_length.id, self.data_length_prop),  # To determine whether Process Monitor should update the graph, the main graph remembers the current amount of data.
            inputs=dict(
                active_tab_id=Input(self.tabs.id, 'active_tab'),
                _=Input(self.dummy, 'children'),
                is_3d=Input(self.switch_3d, 'value'),
                obj_names=(
                    Input(self.axis1_dropdown, 'label'),
                    Input(self.axis2_dropdown, 'label'),
                    Input(self.axis3_dropdown, 'label'),
                ),
            ),
            prevent_initial_call=False,
        )
        def redraw_main_graph(active_tab_id, _, is_3d, obj_names):
            kwargs = {}
            if active_tab_id == self.TAB_ID_OBJECTIVE_PLOT:
                if not is_3d:
                    obj_names = obj_names[:-1]
                kwargs = dict(
                    obj_names=obj_names
                )
            figure, length = self.get_fig_by_tab_id(active_tab_id, with_length=True, kwargs=kwargs)
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
            Output(self.tooltip.id, "direction"),
            Input(self.graph.id, "hoverData"),
            State(self.graph.id, "figure"),
        )
        def show_hover(hoverData, figure):
            if hoverData is None:
                return False, no_update, no_update, no_update

            if self.application.history is None:
                raise PreventUpdate

            # pt = {'curveNumber': 1, 'pointNumber': 0, 'pointIndex': 0, 'x': 1, 'y': 2369132.929996869, 'bbox': {'x0': 341.5, 'x1': 350.5, 'y0': 235, 'y1': 244}, 'customdata': [1]}
            # bbox = {'x0': 341.5, 'x1': 350.5, 'y0': 235, 'y1': 244}  # point の大きさ？
            # figure = {'data': [{'customdata': [[1], [2], [3], [4], [5]], 'marker': {'color': '#6c757d', 'size': 6}, 'mode': 'markers', 'name': 'すべての解', 'x': [1, 2, 3, 4, 5], 'y': [2369132.929996866, 5797829.746579617, 1977631.590419346, 2083867.2403273676, 1539203.6452652698], 'type': 'scatter', 'hoverinfo': 'none'}, {'customdata': [[1], [3], [5]], 'legendgroup': 'optimal', 'line': {'color': '#6c757d', 'width': 1}, 'marker': {'color': '#007bff', 'size': 9}, 'mode': 'markers+lines', 'name': '最適解の推移', 'x': [1, 3, 5], 'y': [2369132.929996866, 1977631.590419346, 1539203.6452652698], 'type': 'scatter', 'hoverinfo': 'none'}, {'legendgroup': 'optimal', 'line': {'color': '#6c757d', 'dash': 'dash', 'width': 0.5}, 'mode': 'lines', 'showlegend': False, 'x': [5, 5], 'y': [1539203.6452652698, 1539203.6452652698], 'type': 'scatter', 'hoverinfo': 'none'}], 'layout': {'template': {'data': {'histogram2dcontour': [{'type': 'histogram2dcontour', 'colorbar': {'outlinewidth': 0, 'ticks': ''}, 'colorscale': [[0, '#0d0887'], [0.1111111111111111, '#46039f'], [0.2222222222222222, '#7201a8'], [0.3333333333333333, '#9c179e'], [0.4444444444444444, '#bd3786'], [0.5555555555555556, '#d8576b'], [0.6666666666666666, '#ed7953'], [0.7777777777777778, '#fb9f3a'], [0.8888888888888888, '#fdca26'], [1, '#f0f921']]}], 'choropleth': [{'type': 'choropleth', 'colorbar': {'outlinewidth': 0, 'ticks': ''}}], 'histogram2d': [{'type': 'histogram2d', 'colorbar': {'outlinewidth': 0, 'ticks': ''}, 'colorscale': [[0, '#0d0887'], [0.1111111111111111, '#46039f'], [0.2222222222222222, '#7201a8'], [0.3333333333333333, '#9c179e'], [0.4444444444444444, '#bd3786'], [0.5555555555555556, '#d8576b'], [0.6666666666666666, '#ed7953'], [0.7777777777777778, '#fb9f3a'], [0.8888888888888888, '#fdca26'], [1, '#f0f921']]}], 'heatmap': [{'type': 'heatmap', 'colorbar': {'outlinewidth': 0, 'ticks': ''}, 'colorscale': [[0, '#0d0887'], [0.1111111111111111, '#46039f'], [0.2222222222222222, '#7201a8'], [0.3333333333333333, '#9c179e'], [0.4444444444444444, '#bd3786'], [0.5555555555555556, '#d8576b'], [0.6666666666666666, '#ed7953'], [0.7777777777777778, '#fb9f3a'], [0.8888888888888888, '#fdca26'], [1, '#f0f921']]}], 'heatmapgl': [{'type': 'heatmapgl', 'colorbar': {'outlinewidth': 0, 'ticks': ''}, 'colorscale': [[0, '#0d0887'], [0.1111111111111111, '#46039f'], [0.2222222222222222, '#7201a8'], [0.3333333333333333, '#9c179e'], [0.4444444444444444, '#bd3786'], [0.5555555555555556, '#d8576b'], [0.6666666666666666, '#ed7953'], [0.7777777777777778, '#fb9f3a'], [0.8888888888888888, '#fdca26'], [1, '#f0f921']]}], 'contourcarpet': [{'type': 'contourcarpet', 'colorbar': {'outlinewidth': 0, 'ticks': ''}}], 'contour': [{'type': 'contour', 'colorbar': {'outlinewidth': 0, 'ticks': ''}, 'colorscale': [[0, '#0d0887'], [0.1111111111111111, '#46039f'], [0.2222222222222222, '#7201a8'], [0.3333333333333333, '#9c179e'], [0.4444444444444444, '#bd3786'], [0.5555555555555556, '#d8576b'], [0.6666666666666666, '#ed7953'], [0.7777777777777778, '#fb9f3a'], [0.8888888888888888, '#fdca26'], [1, '#f0f921']]}], 'surface': [{'type': 'surface', 'colorbar': {'outlinewidth': 0, 'ticks': ''}, 'colorscale': [[0, '#0d0887'], [0.1111111111111111, '#46039f'], [0.2222222222222222, '#7201a8'], [0.3333333333333333, '#9c179e'], [0.4444444444444444, '#bd3786'], [0.5555555555555556, '#d8576b'], [0.6666666666666666, '#ed7953'], [0.7777777777777778, '#fb9f3a'], [0.8888888888888888, '#fdca26'], [1, '#f0f921']]}], 'mesh3d': [{'type': 'mesh3d', 'colorbar': {'outlinewidth': 0, 'ticks': ''}}], 'scatter': [{'fillpattern': {'fillmode': 'overlay', 'size': 10, 'solidity': 0.2}, 'type': 'scatter'}], 'parcoords': [{'type': 'parcoords', 'line': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}}], 'scatterpolargl': [{'type': 'scatterpolargl', 'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}}], 'bar': [{'error_x': {'color': '#2a3f5f'}, 'error_y': {'color': '#2a3f5f'}, 'marker': {'line': {'color': '#E5ECF6', 'width': 0.5}, 'pattern': {'fillmode': 'overlay', 'size': 10, 'solidity': 0.2}}, 'type': 'bar'}], 'scattergeo': [{'type': 'scattergeo', 'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}}], 'scatterpolar': [{'type': 'scatterpolar', 'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}}], 'histogram': [{'marker': {'pattern': {'fillmode': 'overlay', 'size': 10, 'solidity': 0.2}}, 'type': 'histogram'}], 'scattergl': [{'type': 'scattergl', 'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}}], 'scatter3d': [{'type': 'scatter3d', 'line': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}, 'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}}], 'scattermapbox': [{'type': 'scattermapbox', 'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}}], 'scatterternary': [{'type': 'scatterternary', 'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}}], 'scattercarpet': [{'type': 'scattercarpet', 'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}}], 'carpet': [{'aaxis': {'endlinecolor': '#2a3f5f', 'gridcolor': 'white', 'linecolor': 'white', 'minorgridcolor': 'white', 'startlinecolor': '#2a3f5f'}, 'baxis': {'endlinecolor': '#2a3f5f', 'gridcolor': 'white', 'linecolor': 'white', 'minorgridcolor': 'white', 'startlinecolor': '#2a3f5f'}, 'type': 'carpet'}], 'table': [{'cells': {'fill': {'color': '#EBF0F8'}, 'line': {'color': 'white'}}, 'header': {'fill': {'color': '#C8D4E3'}, 'line': {'color': 'white'}}, 'type': 'table'}], 'barpolar': [{'marker': {'line': {'color': '#E5ECF6', 'width': 0.5}, 'pattern': {'fillmode': 'overlay', 'size': 10, 'solidity': 0.2}}, 'type': 'barpolar'}], 'pie': [{'automargin': True, 'type': 'pie'}]}, 'layout': {'autotypenumbers': 'strict', 'colorway': ['#636efa', '#EF553B', '#00cc96', '#ab63fa', '#FFA15A', '#19d3f3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'], 'font': {'color': '#2a3f5f'}, 'hovermode': 'closest', 'hoverlabel': {'align': 'left'}, 'paper_bgcolor': 'white', 'plot_bgcolor': '#E5ECF6', 'polar': {'bgcolor': '#E5ECF6', 'angularaxis': {'gridcolor': 'white', 'linecolor': 'white', 'ticks': ''}, 'radialaxis': {'gridcolor': 'white', 'linecolor': 'white', 'ticks': ''}}, 'ternary': {'bgcolor': '#E5ECF6', 'aaxis': {'gridcolor': 'white', 'linecolor': 'white', 'ticks': ''}, 'baxis': {'gridcolor': 'white', 'linecolor': 'white', 'ticks': ''}, 'caxis': {'gridcolor': 'white', 'linecolor': 'white', 'ticks': ''}}, 'coloraxis': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}, 'colorscale': {'sequential': [[0, '#0d0887'], [0.1111111111111111, '#46039f'], [0.2222222222222222, '#7201a8'], [0.3333333333333333, '#9c179e'], [0.4444444444444444, '#bd3786'], [0.5555555555555556, '#d8576b'], [0.6666666666666666, '#ed7953'], [0.7777777777777778, '#fb9f3a'], [0.8888888888888888, '#fdca26'], [1, '#f0f921']], 'sequentialminus': [[0, '#0d0887'], [0.1111111111111111, '#46039f'], [0.2222222222222222, '#7201a8'], [0.3333333333333333, '#9c179e'], [0.4444444444444444, '#bd3786'], [0.5555555555555556, '#d8576b'], [0.6666666666666666, '#ed7953'], [0.7777777777777778, '#fb9f3a'], [0.8888888888888888, '#fdca26'], [1, '#f0f921']], 'diverging': [[0, '#8e0152'], [0.1, '#c51b7d'], [0.2, '#de77ae'], [0.3, '#f1b6da'], [0.4, '#fde0ef'], [0.5, '#f7f7f7'], [0.6, '#e6f5d0'], [0.7, '#b8e186'], [0.8, '#7fbc41'], [0.9, '#4d9221'], [1, '#276419']]}, 'xaxis': {'gridcolor': 'white', 'linecolor': 'white', 'ticks': '', 'title': {'standoff': 15}, 'zerolinecolor': 'white', 'automargin': True, 'zerolinewidth': 2}, 'yaxis': {'gridcolor': 'white', 'linecolor': 'white', 'ticks': '', 'title': {'standoff': 15}, 'zerolinecolor': 'white', 'automargin': True, 'zerolinewidth': 2}, 'scene': {'xaxis': {'backgroundcolor': '#E5ECF6', 'gridcolor': 'white', 'linecolor': 'white', 'showbackground': True, 'ticks': '', 'zerolinecolor': 'white', 'gridwidth': 2}, 'yaxis': {'backgroundcolor': '#E5ECF6', 'gridcolor': 'white', 'linecolor': 'white', 'showbackground': True, 'ticks': '', 'zerolinecolor': 'white', 'gridwidth': 2}, 'zaxis': {'backgroundcolor': '#E5ECF6', 'gridcolor': 'white', 'linecolor': 'white', 'showbackground': True, 'ticks': '', 'zerolinecolor': 'white', 'gridwidth': 2}}, 'shapedefaults': {'line': {'color': '#2a3f5f'}}, 'annotationdefaults': {'arrowcolor': '#2a3f5f', 'arrowhead': 0, 'arrowwidth': 1}, 'geo': {'bgcolor': 'white', 'landcolor': '#E5ECF6', 'subunitcolor': 'white', 'showland': True, 'showlakes': True, 'lakecolor': 'white'}, 'title': {'x': 0.05}, 'mapbox': {'style': 'light'}}}, 'title': {'text': '目的関数プロット'}, 'xaxis': {'title': {'text': '成功した解析数'}, 'type': 'linear', 'range': [0.7173255954539959, 5.282674404546004], 'autorange': False}, 'yaxis': {'title': {'text': 'Mises Stress'}, 'type': 'linear', 'range': [1194338.1744483153, 6109662.126322151], 'autorange': False}, 'transition': {'duration': 1000}, 'clickmode': 'event+select'}}

            # get point location
            pt = hoverData["points"][0]

            # get the bounding box of target
            bbox = pt['bbox']

            # get relative location of point
            xrange = figure['layout']['xaxis']['range']
            # yrange = figure['layout']['yaxis']['range']

            is_left = pt['x'] < np.mean(xrange)

            # デフォルトでは Hover が Point に重なり、
            # Hover した瞬間に Un-hover する場合があるので
            # offset を追加
            if is_left:
                direction = 'right'
                bbox['x0'] = bbox['x0'] + 40
                bbox['x1'] = bbox['x1'] + 40

            else:
                direction = 'left'
                bbox['x0'] = bbox['x0'] + 15
                bbox['x1'] = bbox['x1'] + 15

            # ついでに縦方向も適当に調整
            bbox['y0'] = bbox['y0'] + 80
            bbox['y1'] = bbox['y1'] + 80


            # get row of the history from customdata defined in main_figure
            if 'customdata' not in pt.keys():
                raise PreventUpdate

            if len(pt['customdata']) == 0:
                raise PreventUpdate

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
            tooltip_layout = html.Div(
                children=[
                    html.Div(img_component, style={'display': 'inline-block', 'margin-right': '10px'}),
                    html.Div(description, style={'display': 'inline-block', 'margin-right': '10px'}),
                    html.Div(tbl_component_obj, style={'display': 'inline-block', 'margin-right': '10px'}),
                ],
            )

            return True, bbox, tooltip_layout, direction

    def create_formatted_parameter(self, row) -> Component:
        meta_columns = self.application.history.meta_columns
        pd.options.display.float_format = '{:.4e}'.format
        parameters = row.iloc[:, np.where(np.array(meta_columns) == 'prm')[0]]
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
        meta_columns = self.application.history.meta_columns
        pd.options.display.float_format = '{:.4e}'.format
        objectives = row.iloc[:, np.where(np.array(meta_columns) == 'obj')[0]]
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
        meta_columns = self.application.history.meta_columns
        if meta_columns[0] != '':
            extra_data = json.loads(meta_columns[0])
            if 'femprj_path' in extra_data.keys():
                # get img path
                femprj_path = extra_data['femprj_path']
                model_name = extra_data['model_name']
                femprj_result_dir = femprj_path.replace('.femprj', '.Results')
                img_path = os.path.join(femprj_result_dir, f'{model_name}_trial{trial}.jpg')
                if os.path.exists(img_path):
                    # create encoded image
                    with open(img_path, 'rb') as f:
                        content = f.read()
                    encoded_image = base64.b64encode(content).decode('utf-8')
                    img_url = 'data:image/jpeg;base64, ' + encoded_image
        return html.Img(src=img_url, style={"width": "200px"}) if img_url is not None else html.Div()

    def get_fig_by_tab_id(self, tab_id, with_length=False, kwargs: dict = None):
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
        kwargs = kwargs or {}
        fig = creator(self.application.history, df, **kwargs)
        if with_length:
            return fig, len(df)
        else:
            return fig

    def data_accessor(self) -> pd.DataFrame:
        from pyfemtet.opt.visualization._process_monitor.application import ProcessMonitorApplication
        if isinstance(self.application, ProcessMonitorApplication):
            df = self.application.local_data
        else:
            df = self.application.history.get_df(valid_only=True)
        return df
