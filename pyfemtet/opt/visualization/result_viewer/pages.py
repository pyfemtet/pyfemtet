import json
import os
import tempfile
import base64

import numpy as np
import pandas as pd
import shutil

from dash import Output, Input, State, callback_context, no_update
from dash.exceptions import PreventUpdate

from pyfemtet.opt.visualization.wrapped_components import dcc, dbc, html
from pyfemtet.opt.visualization.base import AbstractPage  # , logger
from pyfemtet.opt.visualization.complex_components.main_graph import MainGraph  # , FLEXBOX_STYLE_ALLOW_VERTICAL_FILL
from pyfemtet.opt.visualization.complex_components.control_femtet import FemtetControl, FemtetState
from pyfemtet.opt.visualization.complex_components.alert_region import AlertRegion
from pyfemtet.opt.visualization.complex_components.pm_graph import PredictionModelGraph

from pyfemtet.opt._femopt_core import History

from pyfemtet.message import Msg


class HomePage(AbstractPage):

    def __init__(self, title, rel_url='/'):
        super().__init__(title, rel_url)

    # noinspection PyAttributeOutsideInit
    def setup_component(self):

        self.location = dcc.Location(id='result-viewer-location')

        # control femtet subpage
        self.femtet_control: FemtetControl = FemtetControl()
        self.add_subpage(self.femtet_control)

        # main graph subpage
        self.main_graph: MainGraph = MainGraph()
        self.add_subpage(self.main_graph)

        # alert region subpage
        self.alert_region: AlertRegion = AlertRegion()
        self.add_subpage(self.alert_region)

        # open pdt (or transfer variable to femtet)
        self.open_pdt_button = dbc.Button(
            Msg.LABEL_OPEN_PDT_BUTTON,
            id='open-pdt-button',
            color='primary',
            className="position-relative",  # need to show badge
        )

        # update parameter
        self.update_parameter_button = dbc.Button(
            Msg.LABEL_RECONSTRUCT_MODEL_BUTTON,
            id='update-parameter-button',
            color='secondary',
        )

        # file picker
        self.file_picker_button = dbc.Button(
            Msg.LABEL_FILE_PICKER,
            id='file-picker-button',
            color='primary',
        )
        self.file_picker = dcc.Upload(
            id='history-file-picker',
            children=[self.file_picker_button],
            multiple=False,
            style={'display': 'inline'},
        )

        # tutorial subpage (after setup self components)
        self.tutorial: Tutorial = Tutorial(self, self.main_graph, self.femtet_control)
        self.add_subpage(self.tutorial)

    # noinspection PyAttributeOutsideInit
    def setup_layout(self):
        """"""
        """
            =======================
            |  | ---------------- |
            |  | |              | |
            |  | |  Main Graph  | |
            |  | |              | |
            |  | ---------------- |
            |  | [] ...       [] <---- Buttons
            |  | ---------------- |
            |  | | Alert Region | |
            | ^| ---------------- |
            ==|====================
              |
              SideBar
        """

        # Uncomment if the plotly's specification if fixed, and we can use dcc.Graph with FlexBox.
        # style = {'height': '95vh'}
        # style.update(FLEXBOX_STYLE_ALLOW_VERTICAL_FILL)
        self.layout = dbc.Container(
            children=[
                dbc.Row([self.location, self.main_graph.layout, self.tutorial.graph_popover, self.tutorial.control_visibility_input_dummy]),  # visible / invisible components
                dbc.Row(self.femtet_control.layout),  # invisible components
                dbc.Row(
                    children=[
                        dbc.Col(html.Div([self.tutorial.tutorial_mode_switch], className='d-flex justify-content-center')),
                        dbc.Col(html.Div([self.file_picker, self.tutorial.load_sample_csv_div], className='d-flex justify-content-center',),),
                        dbc.Col(html.Div([self.femtet_control.connect_femtet_button_spinner, self.tutorial.connect_femtet_popover], className='d-flex justify-content-center')),
                        dbc.Col(html.Div([self.open_pdt_button, self.update_parameter_button, self.tutorial.open_pdt_popover], className='d-flex justify-content-center')),
                    ],
                    justify='evenly',
                ),
                dbc.Row(self.alert_region.layout),  # visible / invisible components
            ],
            # style=style,
            fluid=True,
        )

    def setup_callback(self):
        # setup callback of subpages
        super().setup_callback()

        app = self.application.app

        # ===== read history csv file from file picker =====
        @app.callback(
            Output(self.file_picker_button.id, self.file_picker_button.Prop.children),
            Output(self.main_graph.tabs.id, self.main_graph.tabs.Prop.active_tab, allow_duplicate=False),
            # Output(self.femtet_control.connect_femtet_button.id, self.femtet_control.connect_femtet_button.Prop.n_clicks),  # automatically connect to femtet is now deprecated because use view-only without femtet license
            Input(self.file_picker.id, self.file_picker.Prop.contents),
            State(self.file_picker.id, self.file_picker.Prop.filename),
            State(self.main_graph.tabs.id, self.main_graph.tabs.Prop.active_tab),
            prevent_initial_call=False,
        )
        def set_history(encoded_file_content: str, file_name: str, active_tab_id: str):

            # if the history is specified before launch GUI (not implemented), respond it.
            if callback_context.triggered_id is None:
                if self.application.history is not None:
                    file_name = os.path.split(self.application.history.path)[1]
                    return f'current file: {file_name}', no_update

            if encoded_file_content is None:
                raise PreventUpdate

            if not file_name.lower().endswith('.csv'):
                raise PreventUpdate

            # create temporary file from file_content(full path is hidden by browser because of the security reason)
            content_type, content_string = encoded_file_content.split(',')
            file_content = base64.b64decode(content_string)
            with tempfile.TemporaryDirectory() as tmp_dir_path:
                csv_path = os.path.join(tmp_dir_path, file_name)
                with open(csv_path, 'wb') as f:
                    f.write(file_content)
                self.application.history = History(csv_path)

            return f'current file: {file_name}', active_tab_id

        # ===== open pdt =====
        @app.callback(
            Output(self.alert_region.alert_region.id, 'children', allow_duplicate=True),
            Input(self.open_pdt_button.id, self.open_pdt_button.Prop.n_clicks),
            State(self.main_graph.selection_data.id, self.main_graph.selection_data_property),
            State(self.alert_region.alert_region.id, 'children'),
            prevent_initial_call=True,
        )
        def open_pdt(_, selection_data, current_alerts):
            # open_pdt allows to open "解析結果単体" without opening any femprj.

            # check Femtet state
            connection_state = self.femtet_control.check_femtet_state()
            if connection_state == FemtetState.missing or connection_state == FemtetState.unconnected:
                msg = Msg.ERR_NO_CONNECTION_ESTABLISHED
                alerts = self.alert_region.create_alerts(msg, color='danger', current_alerts=current_alerts)
                return alerts

            # check selection
            if selection_data is None:
                msg = Msg.ERR_NO_SOLUTION_SELECTED
                alerts = self.alert_region.create_alerts(msg, color='danger', current_alerts=current_alerts)
                return alerts

            # get femprj in history csv
            kwargs = self.femtet_control.create_femtet_interface_args()[0]  # read metadata from history.
            femprj_path = kwargs['femprj_path']
            model_name = kwargs['model_name']

            # check metadata is valid
            if femprj_path is None:
                msg = Msg.ERR_FEMPRJ_IN_CSV_NOT_FOUND
                alerts = self.alert_region.create_alerts(msg, color='danger')
                return alerts

            if model_name is None:
                msg = Msg.ERR_MODEL_IN_CSV_NOT_FOUND
                alerts = self.alert_region.create_alerts(msg, color='danger')
                return alerts

            # check pdt_path in selection_data
            pt = selection_data['points'][0]
            trial = pt['customdata'][0]

            # get pdt path
            pdt_path = self.femtet_control.fem.create_pdt_path(femprj_path, model_name, trial)

            # check pdt exists
            if not os.path.exists(pdt_path):
                msg = ('.pdt file is not found. '
                       'Please check the .Results folder. '
                       'Note that .pdt file save mode depends on '
                       'the `save_pdt` argument of FemtetInterface in optimization script'
                       '(default to `all`).')
                alerts = self.alert_region.create_alerts(msg, color='danger')
                return alerts

            # OpenPDT(PDTFile As String, bOpenWithGUI As Boolean)
            Femtet = self.femtet_control.fem.Femtet
            succeed = Femtet.OpenPDT(pdt_path, True)

            if not succeed:
                msg = Msg.ERR_FAILED_TO_OPEN_PREFIX + pdt_path
                alerts = self.alert_region.create_alerts(msg, color='danger')
                return alerts

            return no_update

        # ===== reconstruct model with updating parameter =====
        @app.callback(
            Output(self.alert_region.alert_region.id, 'children', allow_duplicate=True),
            Input(self.update_parameter_button.id, self.update_parameter_button.Prop.n_clicks),
            State(self.main_graph.selection_data.id, self.main_graph.selection_data_property),
            State(self.alert_region.alert_region.id, 'children'),
            prevent_initial_call=True,
        )
        def update_parameter(_, selection_data, current_alerts):

            # check Femtet state
            connection_state = self.femtet_control.check_femtet_state()
            if connection_state != FemtetState.connected:
                msg = Msg.ERR_NO_CONNECTION_ESTABLISHED
                alerts = self.alert_region.create_alerts(msg, color='danger', current_alerts=current_alerts)
                return alerts

            # check selection
            if selection_data is None:
                msg = Msg.ERR_NO_SOLUTION_SELECTED
                alerts = self.alert_region.create_alerts(msg, color='danger', current_alerts=current_alerts)
                return alerts

            try:
                Femtet = self.femtet_control.fem.Femtet

                # check model to open is included in current project
                if (self.femtet_control.fem.femprj_path != Femtet.Project) \
                        or (self.femtet_control.fem.model_name not in Femtet.GetAnalysisModelNames_py()):
                    msg = Msg.ERR_NO_SUCH_MODEL_IN_FEMPRJ + f' model name: {self.femtet_control.fem.model_name}'
                    alerts = self.alert_region.create_alerts(msg, color='danger', current_alerts=current_alerts)
                    return alerts

                # certify opening model
                Femtet.LoadProjectAndAnalysisModel(
                    self.femtet_control.fem.femprj_path,
                    self.femtet_control.fem.model_name,
                    False
                )

                # copy analysis model or femprj.
                ...

            except Exception as e:
                msg = ('Unknown error has occurred in analysis model compatibility check. '
                       f'exception message is: {e}')
                alerts = self.alert_region.create_alerts(msg, color='danger', current_alerts=current_alerts)
                return alerts

            try:
                # get nth trial from selection data
                pt = selection_data['points'][0]
                trial = pt['customdata'][0]

                # get parameter and update model
                df = self.application.history.get_df()
                row = df[df['trial'] == trial]
                metadata = np.array(self.application.history.metadata)
                idx = np.where(metadata == 'prm')[0]

                names = np.array(row.columns)[idx]
                values = np.array(row.iloc[:, idx]).ravel()

                parameter = pd.DataFrame(
                    dict(
                        name=names,
                        value=values,
                    )
                )

                self.femtet_control.fem.update_model(parameter)

                Femtet.SaveProject(Femtet.Project, True)

                return no_update

            except Exception as e:
                msg = ('Unknown error has occurred in updating model. '
                       f'exception message is: {e}')
                alerts = self.alert_region.create_alerts(msg, color='danger', current_alerts=current_alerts)
                return alerts

        # ===== update alert (chained from launch femtet) =====
        @app.callback(
            Output(self.alert_region.alert_region.id, 'children', allow_duplicate=True),
            Input(self.femtet_control.femtet_state.id, self.femtet_control.femtet_state_prop),
            State(self.alert_region.alert_region.id, 'children'),
            prevent_initial_call=True,
        )
        def add_alert_by_connect_femtet(femtet_state_data, current_alerts):
            if callback_context.triggered_id is None:
                raise PreventUpdate

            msg = femtet_state_data['alert']
            if msg == '':
                raise PreventUpdate

            new_alerts = self.alert_region.create_alerts(msg, 'warning', current_alerts)

            return new_alerts

        # ===== update alert (chained from read history) =====
        @app.callback(
            Output(self.alert_region.alert_region.id, 'children', allow_duplicate=True),
            Input(self.file_picker_button.id, self.file_picker_button.Prop.children),
            State(self.alert_region.alert_region.id, 'children'),
            prevent_initial_call=True,
        )
        def add_alert_on_history_set(_, current_alerts):
            if callback_context.triggered_id is None:
                raise PreventUpdate

            if self.application.history is None:
                raise PreventUpdate

            if self.femtet_control.fem is None:
                raise PreventUpdate

            # check the corresponding between history and Femtet
            # ├ history-side
            kwargs = self.femtet_control.create_femtet_interface_args()[0]  # read metadata from history.
            femprj_path_history_on_history: str or None = kwargs['femprj_path']
            model_name_on_history: str or None = kwargs['model_name']
            # ├ Femtet-side
            Femtet = self.femtet_control.fem.Femtet
            femprj_path: str = Femtet.Project  # it can be '解析結果単体.femprj'
            model_name: str = Femtet.AnalysisModelName
            # └ check
            is_same_femprj = (femprj_path == femprj_path_history_on_history) if femprj_path_history_on_history is not None else True
            is_same_model = (model_name == model_name_on_history) if model_name is not None else True

            # alert
            new_alerts = current_alerts
            if femprj_path_history_on_history is None:
                msg = Msg.WARN_INCONSISTENT_FEMPRJ_PATH
                new_alerts = self.alert_region.create_alerts(msg, 'warning', new_alerts)
            else:
                if not is_same_femprj:
                    msg = Msg.WARN_INCONSISTENT_FEMPRJ_PATH
                    new_alerts = self.alert_region.create_alerts(msg, 'warning', new_alerts)

            if model_name_on_history is None:
                msg = Msg.WARN_INVALID_MODEL_NAME
                new_alerts = self.alert_region.create_alerts(msg, 'warning', new_alerts)
            else:
                if not is_same_model:
                    msg = Msg.WARN_INCONSISTENT_MODEL_NAME
                    new_alerts = self.alert_region.create_alerts(msg, 'warning', new_alerts)

            if new_alerts == current_alerts:
                raise PreventUpdate

            return new_alerts


class Tutorial(AbstractPage):

    # noinspection PyMissingConstructor
    def __init__(self, home_page, main_graph, femtet_control):
        self.home_page: HomePage = home_page
        self.main_graph: MainGraph = main_graph
        self.femtet_control: FemtetControl = femtet_control

    # noinspection PyAttributeOutsideInit
    def setup_component(self):

        self.control_visibility_input_dummy = html.Div(hidden=True)

        self.tutorial_state_prop = 'data-tutorial-state'
        self.tutorial_state = html.Data(
            id='tutorial-state',
            **{self.tutorial_state_prop: dict(
                is_tutorial=False,
                has_history=None,
                point_selected=None,
            )}
        )

        # switch tutorial mode (always visible)
        self.tutorial_mode_switch = dbc.Checklist(
            [Msg.LABEL_TUTORIAL_MODE_SWITCH],
            id='tutorial-mode-switch',
            switch=True,
            value=False
        )

        # load sample csv
        self.load_sample_csv_badge = self.create_badge('Click Me!', 'load-sample-csv-badge')
        self.load_sample_csv_button = dbc.Button(
            children=[Msg.LABEL_LOAD_SAMPLE_CSV, self.load_sample_csv_badge],
            id='load-sample-csv-button',
            className="position-relative",  # need to show badge
        )
        self.load_sample_csv_popover = dbc.Popover(
            children=[
                dbc.PopoverHeader(Msg.LOAD_CSV_POPOVER_HEADER),
                dbc.PopoverBody(Msg.LOAD_CSV_POPOVER_BODY),
            ],
            id='load-sample-csv-popover',
            target=self.load_sample_csv_button.id,
            is_open=False,
            placement='top',
        )
        self.load_sample_csv_div = html.Span(
            children=[self.load_sample_csv_button, self.load_sample_csv_popover],
            id='load-sample-csv-div',
            style={'display': 'none'},
        )

        # popover and badge of main graph
        self.graph_badge = self.create_badge('Choose a Point!', 'graph-badge')
        self.graph_popover = dbc.Popover(
            children=[
                dbc.PopoverHeader(Msg.MAIN_GRAPH_POPOVER_HEADER),
                dbc.PopoverBody(
                    children=[
                        Msg.MAIN_GRAPH_POPOVER_BODY,
                        self.graph_badge
                    ],
                    className="position-relative",  # need to show badge
                ),
            ],
            id='graph-popover',
            target=self.main_graph.tabs.id,
            is_open=False,
            placement='bottom',
            hide_arrow=True,
        )

        # popover and badge of open pdt
        self.open_pdt_badge = self.create_badge('Click Me!', 'open-pdt-badge')
        self.open_pdt_popover = dbc.Popover(
            children=[
                dbc.PopoverHeader(Msg.OPEN_PDT_POPOVER_HEADER),
                dbc.PopoverBody(Msg.OPEN_PDT_POPOVER_BODY),
            ],
            id='open-pdt-popover',
            target=self.home_page.open_pdt_button.id,
            is_open=False,
            placement='top',
        )

        # popover of connect-femtet
        self.connect_femtet_badge = self.create_badge('Click Me!', 'connect-femtet-badge')
        self.connect_femtet_popover = dbc.Popover(
            children=[
                dbc.PopoverBody(Msg.CONNECT_FEMTET_POPOVER_BODY),
            ],
            id='connect-femtet-popover',
            target=self.femtet_control.connect_femtet_button.id,
            is_open=False,
            placement='bottom',
        )

    def setup_layout(self):
        pass

    def setup_callback(self):
        app = self.application.app

        # ===== add tutorial badge to subpage's components =====
        @app.callback(
            Output(self.home_page.open_pdt_button, 'children'),
            Output(self.femtet_control.connect_femtet_button, 'children', allow_duplicate=True),
            Input(self.home_page.location, self.home_page.location.Prop.pathname),
            State(self.home_page.open_pdt_button, 'children'),
            State(self.femtet_control.connect_femtet_button, 'children'),
            prevent_initial_call=True,
        )
        def add_badge_to_button_init(_, current_children_pdt, current_children_femtet):
            # open pdt button
            children_pdt = current_children_pdt if isinstance(current_children_pdt, list) else [current_children_pdt]
            if self.open_pdt_badge not in children_pdt:
                children_pdt.append(self.open_pdt_badge)

            # connect femtet button
            children_femtet = current_children_femtet if isinstance(current_children_femtet, list) else [current_children_femtet]
            if self.connect_femtet_badge not in children_femtet:
                children_femtet.append(self.connect_femtet_badge)

            return children_pdt, children_femtet

        @app.callback(
            Output(self.load_sample_csv_badge, 'style'),  # switch visibility
            Output(self.load_sample_csv_popover, 'is_open'),  # switch visibility
            Output(self.load_sample_csv_div, 'style'),  # switch visibility
            Output(self.home_page.file_picker, 'style'),  # switch visibility
            Output(self.graph_badge, 'style'),  # switch visibility
            Output(self.graph_popover, 'is_open'),  # switch visibility
            Output(self.open_pdt_badge, 'style'),  # switch visibility
            Output(self.open_pdt_popover, 'is_open'),  # switch visibility
            Output(self.connect_femtet_popover, 'is_open'),  # switch visibility
            Output(self.connect_femtet_badge, 'style'),  # switch visibility
            Input(self.tutorial_mode_switch, 'value'),
            Input(self.load_sample_csv_button, 'n_clicks'),  # load button clicked
            Input(self.main_graph.selection_data, self.main_graph.selection_data_property),  # selection changed
            Input(self.femtet_control.femtet_state.id, self.femtet_control.femtet_state_prop),  # connection established
            Input(self.control_visibility_input_dummy, 'children'),
            State(self.load_sample_csv_badge, 'style'),  # switch visibility
            State(self.load_sample_csv_div, 'style'),  # switch visibility
            State(self.home_page.file_picker, 'style'),  # switch visibility
            State(self.graph_badge, 'style'),  # switch visibility
            State(self.open_pdt_badge, 'style'),  # switch visibility
            State(self.connect_femtet_badge, 'style'),  # switch visibility
            prevent_initial_call=True,
        )
        def control_visibility(
                is_tutorial,
                _1,  # load_sample clicked
                selection_data,  # data selected
                _2,  # result of connect_femtet clicked
                _3,  # dummy component loaded
                load_sample_csv_badge_current_style,
                load_sample_csv_div_current_style,
                file_picker_current_style,
                graph_badge_current_style,
                open_pdt_badge_current_style,
                connect_femtet_badge_current_style,
        ):
            ret = {
                (load_sample_csv_badge_style := 0): no_update,
                (load_sample_csv_popover_visible := 1): no_update,
                (load_sample_csv_div_style := 2): no_update,
                (file_picker_style := 3): no_update,
                (graph_badge_style := 4): no_update,
                (graph_popover_visible := 5): no_update,
                (open_pdt_badge_style := 6): no_update,
                (open_pdt_popover_visible := 7): no_update,
                (connect_femtet_popover_visible := 8): no_update,
                (connect_femtet_badge_style := 9): no_update,
            }

            # prevent unexpected update
            if callback_context.triggered_id is None:
                raise PreventUpdate

            # ===== initialize =====
            # non-tutorial component
            ret[file_picker_style] = self.control_visibility_by_style(True, file_picker_current_style)
            # tutorial component
            ret[load_sample_csv_badge_style] = self.control_visibility_by_style(False, load_sample_csv_badge_current_style)
            ret[load_sample_csv_popover_visible] = False
            ret[load_sample_csv_div_style] = self.control_visibility_by_style(False, load_sample_csv_div_current_style)
            ret[graph_badge_style] = self.control_visibility_by_style(False, graph_badge_current_style)
            ret[graph_popover_visible] = False
            ret[open_pdt_badge_style] = self.control_visibility_by_style(False, open_pdt_badge_current_style)
            ret[open_pdt_popover_visible] = False
            ret[connect_femtet_popover_visible] = False
            ret[connect_femtet_badge_style] = self.control_visibility_by_style(False, connect_femtet_badge_current_style)

            # if not tutorial, disable all anyway
            if not is_tutorial:
                return tuple(ret.values())

            # else, visible popover
            else:
                ret[file_picker_style] = self.control_visibility_by_style(False, file_picker_current_style)
                ret[load_sample_csv_div_style] = self.control_visibility_by_style(True, load_sample_csv_div_current_style)
                ret[load_sample_csv_popover_visible] = True
                ret[graph_popover_visible] = True
                ret[connect_femtet_popover_visible] = True
                ret[open_pdt_popover_visible] = True

            # if history is None, show badge to load csv
            if self.application.history is None:
                ret[load_sample_csv_badge_style] = self.control_visibility_by_style(
                    True,
                    load_sample_csv_badge_current_style
                )

            # if history is not None,
            else:
                # if a point is already selected,
                # show badge to open-pdt or connect-femtet
                if self.check_point_selected(selection_data):

                    # if femtet is connected, show badge to open-pdt
                    if self.femtet_control.check_femtet_state() is FemtetState.connected:
                        ret[open_pdt_badge_style] = self.control_visibility_by_style(
                            True,
                            open_pdt_badge_current_style,
                        )

                    # else, show femtet-connect badge
                    else:
                        ret[connect_femtet_badge_style] = self.control_visibility_by_style(
                            True,
                            connect_femtet_badge_current_style,
                        )

                # selection not yet, show badge to main-graph
                else:
                    ret[graph_badge_style] = self.control_visibility_by_style(
                        True,
                        graph_badge_current_style,
                    )

            return tuple(ret.values())

        # ===== load sample csv =====
        alert_region = self.home_page.alert_region.alert_region

        @app.callback(
            Output(self.main_graph.tabs, self.main_graph.tabs.Prop.active_tab, allow_duplicate=True),
            Output(self.control_visibility_input_dummy, 'children'),
            # Output(self.femtet_control.connect_femtet_button, 'n_clicks', allow_duplicate=True),  # automatically open femtet is now deprecated because enable to use analytical functions without using license.
            Output(alert_region, 'children', allow_duplicate=True),
            Input(self.load_sample_csv_button, 'n_clicks'),
            State(self.main_graph.tabs, self.main_graph.tabs.Prop.active_tab),
            State(alert_region, 'children'),
            prevent_initial_call=True,
        )
        def load_sample_csv(_, active_tab, current_alerts):
            """Load sample csv"""
            if callback_context.triggered_id is None:
                raise PreventUpdate

            # get sample file
            import pyfemtet
            package_root = os.path.dirname(pyfemtet.__file__)
            sample_dir = os.path.join(package_root, 'opt', 'samples', 'femprj_sample')  # FIXME: locale によってパスを変える
            path = os.path.join(sample_dir, 'wat_ex14_parametric_test_result.reccsv')

            if not os.path.exists(path):
                msg = Msg.ERR_SAMPLE_CSV_NOT_FOUND
                alerts = self.home_page.alert_region.create_alerts(msg, color='danger', current_alerts=current_alerts)
                return no_update, no_update, alerts
            destination_file = path.replace('wat_ex14_parametric_test_result.reccsv', 'tutorial.csv')
            shutil.copyfile(path, destination_file)
            self.application.history = History(destination_file)

            source_file = path.replace('_test_result.reccsv', '.femprj')
            if not os.path.exists(source_file):
                msg = Msg.ERR_SAMPLE_FEMPRJ_NOT_FOUND
                alerts = self.home_page.alert_region.create_alerts(msg, color='danger', current_alerts=current_alerts)
                return no_update, no_update, alerts
            destination_file = source_file.replace('wat_ex14_parametric', 'tutorial')
            shutil.copyfile(source_file, destination_file)

            source_folder = path.replace('_test_result.reccsv', '.Results')
            if not os.path.exists(source_file):
                msg = Msg.ERR_FEMPRJ_RESULT_NOT_FOUND
                alerts = self.home_page.alert_region.create_alerts(msg, color='danger', current_alerts=current_alerts)
                return no_update, no_update, alerts
            destination_folder = source_folder.replace('wat_ex14_parametric', 'tutorial')
            shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)

            self.application.history.metadata[0] = json.dumps(
                dict(
                    femprj_path=destination_file,
                    model_name='Ex14',
                )
            )

            return active_tab, 1, no_update

    @staticmethod
    def check_point_selected(data):
        if data is None:
            return False
        if len(data['points']) == 0:
            return False
        return True

    @staticmethod
    def create_badge(text, id):
        badge = dbc.Badge(
            children=text,
            color="danger",
            pill=True,
            text_color="white",
            style={'display': 'none'},
            className="position-absolute top-0 start-100 translate-middle",
            id=id,
        )
        return badge

    @staticmethod
    def control_visibility_by_style(visible: bool, current_style: dict):

        visibility = 'inline' if visible else 'none'
        part = {'display': visibility}

        if current_style is None:
            return part

        else:
            current_style.update(part)
            return current_style


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
