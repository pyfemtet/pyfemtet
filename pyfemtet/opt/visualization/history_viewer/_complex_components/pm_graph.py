# callback
from dash import Output, Input, State, no_update, callback_context, ALL
from dash.exceptions import PreventUpdate

# components
from pyfemtet.opt.visualization.history_viewer._wrapped_components import html
from pyfemtet.opt.visualization.history_viewer._wrapped_components import dcc, dbc
from pyfemtet.opt.visualization.history_viewer._complex_components.alert_region import *

# graph
import plotly.graph_objs as go

# the others
from enum import Enum, auto

from pyfemtet.opt.prediction._model import *
from pyfemtet.opt.prediction._helper import *
from pyfemtet.opt.visualization.plotter.pm_graph_creator import plot2d, plot3d
from pyfemtet.opt.visualization.history_viewer._base_application import AbstractPage, logger
from pyfemtet.opt.visualization.history_viewer._helper import has_full_bound, control_visibility_by_style
from pyfemtet._i18n import Msg, _
from pyfemtet._util.df_util import *


__all__ = [
    'PredictionModelGraph'
]


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
        self.pyfemtet_model: PyFemtetModel = PyFemtetModel()
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
        self.fit_rsm_button = dbc.Button([self.fit_rsm_button_spinner, Msg.LABEL_OF_CREATE_PREDICTION_MODEL_BUTTON],
                                         color='success')
        self.redraw_graph_button_spinner = dbc.Spinner(size='sm', spinner_style={'display': 'none'})
        self.redraw_graph_button = dbc.Button([self.redraw_graph_button_spinner,
                                               Msg.LABEL_OF_REDRAW_PREDICTION_MODEL_GRAPH_BUTTON])

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

        # consider sub sampling or not
        self.SUB_SAMPLING_CHECKED = 'sub_sampling_checked'
        self.switch_about_sub_sampling = dbc.Checklist(
            options=[
                dict(
                    label=_('Consider sub sampling when the model re-calc', 'モデル再計算時にサブサンプリングのみを考慮'),
                    value=self.SUB_SAMPLING_CHECKED,  # チェックされたら value (list) の要素に "sub_sampling_checked" が入る
                )
            ],
            switch=True,
            value=[],  # 初期状態ではチェックされている値なし
            style=control_visibility_by_style(
                visible=False, current_style={}
            ),
        )

        # alert region subpage
        self.alert_region: AlertRegion = AlertRegion()
        self.add_subpage(self.alert_region)

    def setup_layout(self):
        self.card_header = dbc.CardHeader(self.tabs)

        self.card_body = dbc.CardBody(
            children=html.Div([
                self.loading,
                self.alert_region.layout,
            ]),  # children=html.Div([self.loading, self.tooltip]),
        )

        dropdown_rows = [
            dbc.Row([
                dbc.Col(html.Span(Msg.LABEL_OF_AXIS1_SELECTION),
                        align='center',
                        style={'text-align': 'end'}, width=2
                        ),
                dbc.Col(self.axis_controllers[0])
            ]),
            dbc.Row(
                [
                    dbc.Col(
                        html.Span(Msg.LABEL_OF_AXIS2_SELECTION),
                        align='center',
                        style={'text-align': 'end'},
                        width=2
                    ),
                    dbc.Col(self.axis_controllers[1])
                ],
                id='prm-axis-2-dropdown'
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Span(Msg.LABEL_OF_AXIS3_SELECTION),
                        align='center',
                        style={'text-align': 'end'},
                        width=2
                    ),
                    dbc.Col(self.axis_controllers[2])
                ]
            ),
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
                dbc.Row((
                    dbc.Col(self.switch_3d),
                    dbc.Col(self.switch_about_sub_sampling)
                )),
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
            if 'display' in state1.keys():
                state1.pop('display')
            if 'display' in state2.keys():
                state2.pop('display')

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
            State(self.switch_about_sub_sampling, 'value'),
            prevent_initial_call=True,
        )
        def recalculate_rsm(command, is_sub_sampling):
            # just in case
            if callback_context.triggered_id is None:
                raise PreventUpdate

            # check command
            if command != self.CommandState.recalc.value:
                raise PreventUpdate

            # load history
            if self.application.history is None:
                return self.CommandState.redraw.value, no_update  # error handling in the next `redraw_graph()` callback

            # check df
            if len(self.application.get_df()) == 0:
                return self.CommandState.redraw.value, no_update  # error handling in the next `redraw_graph()` callback

            # switch the consideration of sub_sampling
            df = self.application.get_df()
            equality_filters = {'sub_sampling': float('nan')}

            if self.SUB_SAMPLING_CHECKED in is_sub_sampling:
                df = get_partial_df(
                    df,
                    equality_filters=equality_filters,
                    method='all-exclude',  # nan 以外 = sampling のみ
                )
            else:
                df = get_partial_df(
                    df,
                    equality_filters=equality_filters,
                    method='all',  # nan のみ全て
                )

            # create model
            model = SingleTaskGPModel()
            self.pyfemtet_model.update_model(model)
            self.pyfemtet_model.fit(
                self.application.history,
                df,
                **{}
            )

            return self.CommandState.redraw.value, no_update

        # ===== Update Graph =====
        @app.callback(
            Output(self.graph, 'figure'),
            Output(self.command_manager, self.command_manager_prop),
            Output(self.alert_region.alert_region, 'children', allow_duplicate=True),
            # To determine whether Process Monitor should update
            # the graph, the main graph remembers the current
            # amount of data.
            # Output(self.data_length.id, self.data_length_prop),
            Input(self.command_manager, self.command_manager_prop),
            State(self.tabs, 'active_tab'),
            State(self.axis1_prm_dropdown, 'label'),
            State(self.axis2_prm_dropdown, 'label'),
            State(self.axis3_obj_dropdown, 'label'),
            State(self.slider_container, 'children'),  # for callback chain
            State({'type': 'prm-slider', 'index': ALL}, 'id'),
            State({'type': 'prm-slider', 'index': ALL}, 'value'),
            State({'type': 'prm-slider', 'index': ALL}, 'marks'),
            State(self.switch_3d, 'value'),
            prevent_initial_call=True,
        )
        def redraw_graph(
                command,
                _1,
                axis1_label,
                axis2_label,
                axis3_label,
                _2,
                prm_ids,
                prm_values,
                prm_slider_marks,
                is_3d
        ):
            # just in case
            if callback_context.triggered_id is None:
                raise PreventUpdate

            # check command
            if command != self.CommandState.redraw.value:
                raise PreventUpdate

            # load history
            if self.application.history is None:
                logger.error(Msg.ERR_NO_HISTORY_SELECTED)
                # to re-enable buttons, fire callback chain
                return (
                    no_update,
                    self.CommandState.ready.value,
                    self.alert_region.create_alerts(Msg.ERR_NO_HISTORY_SELECTED),
                )
            # prm_names = self.application.history.prm_names

            # check history
            if len(self.application.get_df()) == 0:
                logger.error(Msg.ERR_NO_FEM_RESULT)
                # to re-enable buttons, fire callback chain
                return (
                    no_update,
                    self.CommandState.ready.value,
                    self.alert_region.create_alerts(Msg.ERR_NO_FEM_RESULT, color='danger'),
                )

            # check fit
            if not hasattr(self.pyfemtet_model, 'current_model'):
                logger.error(Msg.ERR_NO_PREDICTION_MODEL)
                # to re-enable buttons, fire callback chain
                return (
                    no_update,
                    self.CommandState.ready.value,
                    self.alert_region.create_alerts(Msg.ERR_NO_PREDICTION_MODEL, color='danger'),
                )

            # check fixed and no-boundary parameter,
            # cancel to write graph.
            if (
                    not has_full_bound(self.application.history, axis1_label)
                    or (is_3d and not has_full_bound(self.application.history, axis2_label))
            ):
                # to re-enable buttons, fire callback chain
                return (
                    go.Figure(),
                    self.CommandState.ready.value,
                    self.alert_region.create_alerts(
                        _(
                            en_message='Cannot draw the graph because '
                                       'the bounds of selected parameter '
                                       'are not given.',
                            jp_message='選択された変数は上下限が与えられていないため、'
                                       'グラフを描画できません。',
                        ),
                        color='danger',
                    ),
                )

            # create params
            params = dict()
            for prm_id, slider_value, slider_marks in zip(prm_ids, prm_values, prm_slider_marks):

                prm_name = prm_id['index']

                if self.application.history.is_numerical_parameter(prm_name):
                    value = slider_value

                # categorical parameters are encoded as an integer, so restore them here.
                elif self.application.history.is_categorical_parameter(prm_name):
                    value = slider_marks[str(slider_value)]

                else:
                    raise NotImplementedError

                params.update({prm_name: value})

            if is_3d:
                fig = plot3d(
                    history=self.application.history,
                    prm_name1=axis1_label,
                    prm_name2=axis2_label,
                    params=params,
                    obj_name=axis3_label,
                    df=self.application.get_df(),
                    pyfemtet_model=self.pyfemtet_model,
                    n=20,
                )

            else:
                fig = plot2d(
                    history=self.application.history,
                    prm_name1=axis1_label,
                    params=params,
                    obj_name=axis3_label,
                    df=self.application.get_df(),
                    pyfemtet_model=self.pyfemtet_model,
                    n=200,
                )

            return fig, self.CommandState.ready.value, []

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
            Output(self.switch_about_sub_sampling, 'style'),
            Input(self.location, self.location.Prop.pathname),
            State(self.switch_about_sub_sampling, 'style'),
            prevent_initial_call=True,
        )
        def setup_dropdown_and_sliders(_, current_style):
            # just in case
            if callback_context.triggered_id is None:
                raise PreventUpdate

            # load history
            if self.application.history is None:
                logger.error(Msg.ERR_NO_HISTORY_SELECTED)
                raise PreventUpdate

            # add dropdown item to dropdown 1, 2 (input)
            axis1_dropdown_items, axis2_dropdown_items = [], []
            for i, prm_name in enumerate(self.application.history.prm_names):

                # dropdown1 (x)
                dm_item_1 = dbc.DropdownMenuItem(
                    children=prm_name,
                    id={'type': 'axis1-dropdown-menu-item', 'index': prm_name},
                )
                axis1_dropdown_items.append(dm_item_1)

                # dropdown2 (y)
                dm_item_2 = dbc.DropdownMenuItem(
                    children=prm_name,
                    id={'type': 'axis2-dropdown-menu-item', 'index': prm_name},
                )
                axis2_dropdown_items.append(dm_item_2)

            # add dropdown item to dropdown 3 (output)
            axis3_dropdown_items = []
            for i, obj_name in enumerate(self.application.history.obj_names):
                dm_item = dbc.DropdownMenuItem(
                    children=obj_name,
                    id={'type': 'axis3-dropdown-menu-item', 'index': obj_name},
                )
                axis3_dropdown_items.append(dm_item)

            df = self.application.get_df()

            # add sliders
            sliders = []
            slider_values = {}
            for prm_name in self.application.history.prm_names:

                if self.application.history.is_numerical_parameter(prm_name):
                    lb, ub = get_bounds_containing_entire_bounds(
                        self.application.get_df(),
                        prm_name,
                    )

                    if lb is None:
                        lb = df[prm_name].dropna().min()
                    if ub is None:
                        ub = df[prm_name].dropna().max()

                    # create slider
                    value = (lb + ub) / 2
                    slider = dcc.Slider(
                        lb,
                        ub,
                        marks=None,
                        value=value,
                        id={'type': 'prm-slider', 'index': prm_name},
                        tooltip={"placement": "bottom", "always_visible": True},
                    )

                elif self.application.history.is_categorical_parameter(prm_name):
                    choices: set = get_choices_containing_entire_bounds(
                        self.application.get_df(),
                        prm_name,
                    )

                    choices: tuple = tuple(choices)

                    value = choices[0]
                    slider = dcc.Slider(
                        0,
                        len(choices),
                        step=None,
                        marks={i: choice for i, choice in enumerate(choices)},
                        value=0,
                        id={'type': 'prm-slider', 'index': prm_name},
                        tooltip={"placement": "bottom", "always_visible": True},
                        included=False,
                    )

                else:
                    raise NotImplementedError

                slider_values.update({prm_name: value})
                stack = dbc.Stack(
                    id={'type': 'prm-slider-stack', 'index': prm_name},
                    style={'display': 'inline'},
                    children=[
                        html.Div(f'{prm_name}: '),
                        slider
                    ]
                )
                sliders.append(stack)

            # ひとつでも sub sampling があれば visible=True
            visible = (~df['sub_sampling'].isna()).any()
            switch_style = control_visibility_by_style(
                visible, current_style
            )

            return (
                axis1_dropdown_items,
                axis2_dropdown_items,
                axis3_dropdown_items,
                sliders,
                slider_values,
                switch_style,
            )

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
                (slider_style_list_key := 4): [(style.update({'display': 'inline'}), style)[1]
                                               for style in current_styles],
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

                # avoid unable choice
                index = None
                for i in range(len(prm_names)):
                    if has_full_bound(self.application.history, prm_names[i]):
                        ret[ax1_label_key] = prm_names[i]
                        index = i
                        break

                ret[ax2_label_key] = ''
                if index is not None:
                    for j in range(index + 1, len(prm_names)):
                        # ret[ax2_label_key] = prm_names[1] if len(prm_names) >= 2 else ''
                        if has_full_bound(self.application.history, prm_names[j]):
                            ret[ax2_label_key] = prm_names[j]
                            break

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
