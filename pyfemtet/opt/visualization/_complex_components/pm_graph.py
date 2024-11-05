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
from enum import Enum, auto
import os
import base64
import json
import numpy as np

from pyfemtet.opt.visualization._complex_components.pm_graph_creator import PredictionModelCreator
from pyfemtet.opt.visualization._base import AbstractPage, logger
from pyfemtet._message import Msg


FLEXBOX_STYLE_ALLOW_VERTICAL_FILL = {
    'display': 'flex',
    'flex-direction': 'column',
    'flex-grow': '1',
}


# noinspection PyAttributeOutsideInit
class PredictionModelGraph(AbstractPage):
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

    class CommandState(Enum):
        ready = auto()
        recalc = auto()
        redraw = auto()

    def __init__(self):
        self.rsm_creator: PredictionModelCreator = PredictionModelCreator()
        super().__init__()

    def setup_component(self):
        self.location = dcc.Location(id='rsm-graph-location', refresh=True)

        # setup header
        self.tab_list = [dbc.Tab(label=Msg.TAB_LABEL_PREDICTION_MODEL)]
        self.tabs = dbc.Tabs(self.tab_list)

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

        # setup selection data
        # self.selection_data_property = 'data-selection'  # must be starts with "data-"
        # self.selection_data = html.Data(id='selection-data', **{self.selection_data_property: {}})

        # update rsm button
        self.command_manager_prop = 'data-command-manager'
        self.command_manager = html.Data(**{self.command_manager_prop: self.CommandState.ready.value})
        self.fit_rsm_button_spinner = dbc.Spinner(size='sm', spinner_style={'display': 'none'})
        self.fit_rsm_button = dbc.Button([self.fit_rsm_button_spinner, Msg.LABEL_OF_CREATE_PREDICTION_MODEL_BUTTON], color='success')
        self.redraw_graph_button_spinner = dbc.Spinner(size='sm', spinner_style={'display': 'none'})
        self.redraw_graph_button = dbc.Button([self.redraw_graph_button_spinner, Msg.LABEL_OF_REDRAW_PREDICTION_MODEL_GRAPH_BUTTON])

        # # set data length (to notify data updated to application)
        # self.data_length_prop = 'data-df-length'  # must start with "data-"
        # self.data_length = html.Data(id='df-length-data', **{self.data_length_prop: None})

        # Dropdowns
        self.axis1_prm_dropdown = dbc.DropdownMenu()
        self.axis2_prm_dropdown = dbc.DropdownMenu()
        self.axis3_obj_dropdown = dbc.DropdownMenu()
        self.axis_controllers = [
            self.axis1_prm_dropdown,
            self.axis2_prm_dropdown,
            self.axis3_obj_dropdown,
        ]

        # sliders
        self.slider_stack_data_prop = 'data-prm-slider-values'
        self.slider_stack_data = html.Data(**{self.slider_stack_data_prop: {}})
        self.slider_container = html.Div()

        # 2d or 3d
        self.switch_3d = dbc.Checklist(
            options=[
                dict(
                    label=Msg.LABEL_SWITCH_PREDICTION_MODEL_3D,
                    disabled=False,
                    value=False,
                )
            ],
            switch=True,
            value=[],
        )

    def setup_layout(self):
        self.card_header = dbc.CardHeader(self.tabs)

        self.card_body = dbc.CardBody(
            children=html.Div([self.loading]),  # children=html.Div([self.loading, self.tooltip]),
        )

        dropdown_rows = [
            dbc.Row([dbc.Col(html.Span(Msg.LABEL_OF_AXIS1_SELECTION), align='center', style={'text-align': 'end'}, width=2), dbc.Col(self.axis_controllers[0])]),
            dbc.Row([dbc.Col(html.Span(Msg.LABEL_OF_AXIS2_SELECTION), align='center', style={'text-align': 'end'}, width=2), dbc.Col(self.axis_controllers[1])], id='prm-axis-2-dropdown'),
            dbc.Row([dbc.Col(html.Span(Msg.LABEL_OF_AXIS3_SELECTION), align='center', style={'text-align': 'end'}, width=2), dbc.Col(self.axis_controllers[2])]),
        ]

        self.card_footer = dbc.CardFooter(
            children=[
                dbc.Stack(
                    children=[
                        self.fit_rsm_button,
                        self.redraw_graph_button,
                        self.command_manager
                    ],
                    direction='horizontal', gap=2),
                self.switch_3d,
                *dropdown_rows,
                self.slider_container,
                self.slider_stack_data,
            ],
        )

        self.graph_card = dbc.Card(
            [
                self.location,
                # self.data_length,
                self.card_header,
                self.card_body,
                self.card_footer,
            ],
            # style=FLEXBOX_STYLE_ALLOW_VERTICAL_FILL,
        )

        self.layout = dbc.Container(
            children=[
                dbc.Row(self.graph_card),
            ]
        )

    def setup_callback(self):
        # setup callback of subpages
        super().setup_callback()

        app = self.application.app

        # ===== control button disabled and calculation =====
        """
        [fit button clicked] ────┬──> [disabled] ─┬─> [calc] ─╮
        [redraw button clicked] ─┘                └───────────┴─> [redraw] ─> [enabled]
        """

        # ===== disable fit buttons when clicked =====
        @app.callback(
            Output(self.fit_rsm_button_spinner, 'spinner_style', allow_duplicate=True),
            Output(self.fit_rsm_button, 'disabled', allow_duplicate=True),
            Output(self.redraw_graph_button_spinner, 'spinner_style', allow_duplicate=True),
            Output(self.redraw_graph_button, 'disabled', allow_duplicate=True),
            Output(self.command_manager, self.command_manager_prop, allow_duplicate=True),
            Output(self.switch_3d, 'options', allow_duplicate=True),
            Input(self.fit_rsm_button, 'n_clicks'),
            Input(self.redraw_graph_button, 'n_clicks'),
            State(self.fit_rsm_button_spinner, 'spinner_style'),
            State(self.redraw_graph_button_spinner, 'spinner_style'),
            State(self.switch_3d, 'options'),
            prevent_initial_call=True,
        )
        def disable_fit_button(_1, _2, state1, state2, switch_options):
            # spinner visibility
            if 'display' in state1.keys(): state1.pop('display')
            if 'display' in state2.keys(): state2.pop('display')

            # recalc or redraw
            if callback_context.triggered_id == self.fit_rsm_button.id:
                command = self.CommandState.recalc.value
            else:
                command = self.CommandState.redraw.value

            # disable switch
            option = switch_options[0]
            option.update({'disabled': True})
            switch_options[0] = option

            return state1, True, state2, True, command, switch_options

        # ===== recreate RSM =====
        @app.callback(
            Output(self.command_manager, self.command_manager_prop, allow_duplicate=True),
            Output(self.graph, 'figure', allow_duplicate=True),  # for show spinner during calculation
            Input(self.command_manager, self.command_manager_prop),
            prevent_initial_call=True,
        )
        def recalculate_rsm(command):
            # just in case
            if callback_context.triggered_id is None:
                raise PreventUpdate

            # check command
            if command != self.CommandState.recalc.value:
                raise PreventUpdate

            # load history
            if self.application.history is None:
                return self.CommandState.redraw.value, no_update  # error handling in the next `redraw_graph()` callback

            # check history
            if len(self.data_accessor()) == 0:
                return self.CommandState.redraw.value, no_update  # error handling in the next `redraw_graph()` callback

            # create model
            self.rsm_creator.fit(
                self.application.history,
                self.data_accessor(),
            )

            return self.CommandState.redraw.value, no_update

        # ===== Update Graph =====
        @app.callback(
            Output(self.graph, 'figure'),
            Output(self.command_manager, self.command_manager_prop),
            # Output(self.data_length.id, self.data_length_prop),  # To determine whether Process Monitor should update the graph, the main graph remembers the current amount of data.
            Input(self.command_manager, self.command_manager_prop),
            State(self.tabs, 'active_tab'),
            State(self.axis1_prm_dropdown, 'label'),
            State(self.axis2_prm_dropdown, 'label'),
            State(self.axis3_obj_dropdown, 'label'),
            State(self.slider_container, 'children'),  # for callback chain
            State({'type': 'prm-slider', 'index': ALL}, 'value'),
            State(self.switch_3d, 'value'),
            prevent_initial_call=True,
        )
        def redraw_graph(command, active_tab_id, axis1_label, axis2_label, axis3_label, _2, prm_values, is_3d):
            # just in case
            if callback_context.triggered_id is None:
                raise PreventUpdate

            # check command
            if command != self.CommandState.redraw.value:
                raise PreventUpdate

            # load history
            if self.application.history is None:
                logger.error(Msg.ERR_NO_HISTORY_SELECTED)
                return no_update, self.CommandState.ready.value  # to re-enable buttons, fire callback chain
            prm_names = self.application.history.prm_names

            # check history
            if len(self.data_accessor()) == 0:
                logger.error(Msg.ERR_NO_FEM_RESULT)
                return no_update, self.CommandState.ready.value  # to re-enable buttons, fire callback chain

            # check fit
            if not hasattr(self.rsm_creator, 'history'):
                logger.error(Msg.ERR_NO_PREDICTION_MODEL)
                return no_update, self.CommandState.ready.value  # to re-enable buttons, fire callback chain

            if not is_3d:
                axis2_label = None

            # get indices to remove
            idx1 = prm_names.index(axis1_label) if axis1_label in prm_names else None
            idx2 = prm_names.index(axis2_label) if axis2_label in prm_names else None

            # replace values to remove to None
            if idx1 is not None:
                prm_values[idx1] = None
            if idx2 is not None:
                prm_values[idx2] = None

            # remove all None from array: prm_values is now remaining_x
            while None in prm_values:
                prm_values.remove(None)

            # create figure
            fig = self.rsm_creator.create_figure(
                axis1_label,
                axis3_label,
                remaining_x=prm_values,
                prm_name_2=axis2_label,
            )

            return fig, self.CommandState.ready.value

        # ===== re-enable buttons when the graph is updated,  =====
        @app.callback(
            Output(self.fit_rsm_button, 'disabled', allow_duplicate=True),
            Output(self.fit_rsm_button_spinner, 'spinner_style', allow_duplicate=True),
            Output(self.redraw_graph_button, 'disabled', allow_duplicate=True),
            Output(self.redraw_graph_button_spinner, 'spinner_style', allow_duplicate=True),
            Output(self.switch_3d, 'options', allow_duplicate=True),
            Input(self.command_manager, self.command_manager_prop),
            State(self.fit_rsm_button_spinner, 'spinner_style'),
            State(self.redraw_graph_button_spinner, 'spinner_style'),
            State(self.switch_3d, 'options'),
            prevent_initial_call=True,
        )
        def enable_buttons(command, state1, state2, switch_options):
            if command != self.CommandState.ready.value:
                raise PreventUpdate
            state1.update({'display': 'none'})
            state2.update({'display': 'none'})

            # enable switch
            option = switch_options[0]
            option.update({'disabled': False})
            switch_options[0] = option

            return False, state1, False, state2, switch_options

        # ===== setup dropdown and sliders from history =====
        @app.callback(
            Output(self.axis1_prm_dropdown, 'children'),
            Output(self.axis2_prm_dropdown, 'children'),
            Output(self.axis3_obj_dropdown, 'children'),
            Output(self.slider_container, 'children'),
            Output(self.slider_stack_data, self.slider_stack_data_prop, allow_duplicate=True),
            Input(self.location, self.location.Prop.pathname),
            prevent_initial_call=True,
        )
        def setup_dropdown_and_sliders(*_):
            # just in case
            if callback_context.triggered_id is None:
                raise PreventUpdate

            # load history
            if self.application.history is None:
                logger.error(Msg.ERR_NO_HISTORY_SELECTED)
                raise PreventUpdate

            # add dropdown item to dropdown
            axis1_dropdown_items, axis2_dropdown_items = [], []
            for i, prm_name in enumerate(self.application.history.prm_names):
                dm_item_1 = dbc.DropdownMenuItem(
                    children=prm_name,
                    id={'type': 'axis1-dropdown-menu-item', 'index': prm_name},
                )
                axis1_dropdown_items.append(dm_item_1)

                dm_item_2 = dbc.DropdownMenuItem(
                    children=prm_name,
                    id={'type': 'axis2-dropdown-menu-item', 'index': prm_name},
                )
                axis2_dropdown_items.append(dm_item_2)
            axis3_dropdown_items = []
            for i, obj_name in enumerate(self.application.history.obj_names):
                dm_item = dbc.DropdownMenuItem(
                    children=obj_name,
                    id={'type': 'axis3-dropdown-menu-item', 'index': obj_name},
                )
                axis3_dropdown_items.append(dm_item)

            # add sliders
            sliders = []
            slider_values = {}
            for prm_name in self.application.history.prm_names:
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
                value = (lb + ub) / 2
                slider_values.update({prm_name: value})
                stack = dbc.Stack(
                    id={'type': 'prm-slider-stack', 'index': prm_name},
                    style={'display': 'inline'},
                    children=[
                        html.Div(f'{prm_name}: '),
                        dcc.Slider(
                            lb,
                            ub,
                            marks=None,
                            value=value,
                            id={'type': 'prm-slider', 'index': prm_name},
                            tooltip={"placement": "bottom", "always_visible": True},
                        )
                    ]
                )
                sliders.append(stack)

            return axis1_dropdown_items, axis2_dropdown_items, axis3_dropdown_items, sliders, slider_values

        # ===== control dropdown and slider visibility =====
        @app.callback(
            Output(self.axis1_prm_dropdown, 'label'),  # label of dropdown
            Output(self.axis2_prm_dropdown, 'label'),
            Output(self.axis3_obj_dropdown, 'label'),
            Output({'type': 'prm-slider-stack', 'index': ALL}, 'style'),  # visibility of slider
            Output('prm-axis-2-dropdown', 'hidden'),
            Input({'type': 'axis1-dropdown-menu-item', 'index': ALL}, 'n_clicks'),  # when the dropdown item is clicked
            Input({'type': 'axis2-dropdown-menu-item', 'index': ALL}, 'n_clicks'),
            Input({'type': 'axis3-dropdown-menu-item', 'index': ALL}, 'n_clicks'),
            Input(self.axis1_prm_dropdown, 'children'),  # for callback chain timing
            Input(self.switch_3d, 'value'),
            State(self.axis1_prm_dropdown, 'label'),
            State(self.axis2_prm_dropdown, 'label'),
            State(self.axis3_obj_dropdown, 'label'),
            State({'type': 'prm-slider-stack', 'index': ALL}, 'style'),  # visibility of slider
            prevent_initial_call=True,
        )
        def update_controller(*args):
            # argument processing
            current_ax1_label = args[5]
            current_ax2_label = args[6]
            current_styles: list[dict] = args[8]
            is_3d = args[4]

            # just in case
            if callback_context.triggered_id is None:
                raise PreventUpdate

            # load history
            if self.application.history is None:
                logger.error(Msg.ERR_NO_HISTORY_SELECTED)
                raise PreventUpdate
            prm_names = self.application.history.prm_names
            obj_names = self.application.history.obj_names

            # default return values
            ret = {
                (ax1_label_key := 1): no_update,
                (ax2_label_key := 2): no_update,
                (ax3_label_key := 3): no_update,
                (slider_style_list_key := 4): [(style.update({'display': 'inline'}), style)[1] for style in current_styles],
                (ax2_hidden := 5): False,
            }

            # ===== hide dropdown of axis 2 if prm number is 1 =====
            if len(prm_names) < 2:
                ret[ax2_hidden] = True

            # ===== hide dropdown of axis 2 if not is_3d =====
            if not is_3d:
                ret[ax2_hidden] = True

            # ===== update dropdown label =====

            # by callback chain on loaded after setup_dropdown_and_sliders()
            if callback_context.triggered_id == self.axis1_prm_dropdown.id:
                ret[ax1_label_key] = prm_names[0]
                ret[ax2_label_key] = prm_names[1] if len(prm_names) >= 2 else ''
                ret[ax3_label_key] = obj_names[0]

            # by dropdown clicked
            elif isinstance(callback_context.triggered_id, dict):
                # example of triggerd_id: {'index': 'd', 'type': 'axis1-dropdown-menu-item'}
                new_label = callback_context.triggered_id['index']

                # ax1
                if callback_context.triggered_id['type'] == 'axis1-dropdown-menu-item':
                    ret[ax1_label_key] = new_label
                    if new_label == current_ax2_label:
                        ret[ax2_label_key] = current_ax1_label


                # ax2
                elif callback_context.triggered_id['type'] == 'axis2-dropdown-menu-item':
                    ret[ax2_label_key] = new_label
                    if new_label == current_ax1_label:
                        ret[ax1_label_key] = current_ax2_label


                # ax3
                elif callback_context.triggered_id['type'] == 'axis3-dropdown-menu-item':
                    ret[ax3_label_key] = new_label

            # ===== update visibility of sliders =====

            # invisible the slider correspond to the dropdown-1
            label_key, current_label = ax1_label_key, current_ax1_label
            # get label of output
            label = ret[label_key] if ret[label_key] != no_update else current_label
            # update display style of slider
            idx = prm_names.index(label) if label in prm_names else None
            if idx is not None:
                current_styles[idx].update({'display': 'none'})
                ret[slider_style_list_key][idx] = current_styles[idx]

            # invisible the slider correspond to the dropdown-2
            label_key, current_label = ax2_label_key, current_ax2_label
            # get label of output
            label = ret[label_key] if ret[label_key] != no_update else current_label
            # update display style of slider
            idx = prm_names.index(label) if label in prm_names else None
            if idx is not None:
                # if 2d, should not disable the slider correspond to dropdown-2.
                if is_3d:
                    current_styles[idx].update({'display': 'none'})
                    ret[slider_style_list_key][idx] = current_styles[idx]

            return tuple(ret.values())

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

    # def get_fig_by_tab_id(self, tab_id, with_length=False):
    #     # If the history is not loaded, do nothing
    #     if self.application.history is None:
    #         raise PreventUpdate
    #
    #     # else, get creator by tab_id
    #     if tab_id == 'default':
    #         creator = self.figure_creators[0]['creator']
    #     else:
    #         creators = [d['creator'] for d in self.figure_creators if d['tab_id'] == tab_id]
    #         if len(creators) == 0:
    #             raise PreventUpdate
    #         creator = creators[0]
    #
    #     # create figure
    #     df = self.data_accessor()
    #     fig = creator(self.application.history, df)
    #     if with_length:
    #         return fig, len(df)
    #     else:
    #         return fig

    def data_accessor(self) -> pd.DataFrame:
        from pyfemtet.opt.visualization._process_monitor.application import ProcessMonitorApplication
        if isinstance(self.application, ProcessMonitorApplication):
            df = self.application.local_data
        else:
            df = self.application.history.get_df()
        return df
