import os
import tempfile
import base64

import numpy as np
import pandas as pd

from dash import Output, Input, State, callback_context, no_update
from dash.exceptions import PreventUpdate

from pyfemtet.opt.visualization.wrapped_components import dcc, dbc, html
from pyfemtet.opt.visualization.base import AbstractPage, logger
from pyfemtet.opt.visualization.complex_components.main_graph import MainGraph  # , FLEXBOX_STYLE_ALLOW_VERTICAL_FILL
from pyfemtet.opt.visualization.complex_components.control_femtet import FemtetControl, FemtetState
from pyfemtet.opt.visualization.complex_components.alert_region import AlertRegion

from pyfemtet.opt._femopt_core import History




class HomePage(AbstractPage):

    def __init__(self, title, rel_url='/'):
        super().__init__(title, rel_url)

    def setup_component(self):
        # control femtet
        # noinspection PyAttributeOutsideInit
        self.femtet_control: FemtetControl = FemtetControl()
        self.add_subpage(self.femtet_control)

        # main graph
        # noinspection PyAttributeOutsideInit
        self.main_graph: MainGraph = MainGraph()
        self.add_subpage(self.main_graph)

        # alert region
        # noinspection PyAttributeOutsideInit
        self.alert_region: AlertRegion = AlertRegion()
        self.add_subpage(self.alert_region)

        # open pdt (or transfer variable to femtet)
        # noinspection PyAttributeOutsideInit
        self.open_pdt_button = dbc.Button(
            'Open Result in Femtet',
            id='open-pdt-button',
            color='primary',
        )

        # update parameter
        # noinspection PyAttributeOutsideInit
        self.update_parameter_button = dbc.Button(
            'Reconstruct Model',
            id='update-parameter-button',
            color='secondary',
        )

        # file picker
        # noinspection PyAttributeOutsideInit
        self.file_picker_button = dbc.Button('drag and drop or select files', id='file-picker-button', outline=True, color='primary')
        # noinspection PyAttributeOutsideInit
        self.file_picker = dcc.Upload(
            id='history-file-picker',
            children=[self.file_picker_button],
            multiple=False,
        )

    # noinspection PyAttributeOutsideInit
    def setup_layout(self):
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
                dbc.Row(self.main_graph.layout),  # visible / invisible components
                dbc.Row(self.femtet_control.layout),  # invisible components
                dbc.Row(
                    children=[
                        dbc.Col(html.Div(self.file_picker, className='d-flex justify-content-center')),
                        dbc.Col(html.Div(self.femtet_control.connect_femtet_button_spinner, className='d-flex justify-content-center')),
                        dbc.Col(html.Div([self.open_pdt_button, self.update_parameter_button], className='d-flex justify-content-center')),
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

        # ===== read history csv file from file picker ==s===
        @app.callback(
            Output(self.file_picker_button.id, self.file_picker_button.Prop.children),  # FIXME: Used to show current file and fire callback of alert. Separate them.
            Output(self.main_graph.tabs.id, self.main_graph.tabs.Prop.active_tab),
            Output(self.femtet_control.connect_femtet_button.id, self.femtet_control.connect_femtet_button.Prop.n_clicks),  # automatically connect to femtet if the metadata of csv is valid
            Input(self.file_picker.id, self.file_picker.Prop.contents),
            State(self.file_picker.id, self.file_picker.Prop.filename),
            State(self.main_graph.tabs.id, self.main_graph.tabs.Prop.active_tab),)
        def set_history(encoded_file_content: str, file_name: str, active_tab_id: str):

            if callback_context.triggered_id is None:
                if self.application.history is not None:
                    file_name = os.path.split(self.application.history.path)[1]
                    return f'current file: {file_name}', no_update, no_update

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

            return f'current file: {file_name}', active_tab_id, 1

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
                msg = ('Connection to Femtet is not established. '
                       'Launch Femtet and Open a project.')
                alerts = self.alert_region.create_alerts(msg, color='danger', current_alerts=current_alerts)
                return alerts

            # check selection
            if selection_data is None:
                msg = 'No result plot is selected.'
                alerts = self.alert_region.create_alerts(msg, color='danger', current_alerts=current_alerts)
                return alerts

            # get femprj in history csv
            kwargs = self.femtet_control.create_femtet_interface_args()[0]  # read metadata from history.
            femprj_path = kwargs['femprj_path']
            model_name = kwargs['model_name']

            # check metadata is valid
            if femprj_path is None:
                msg = 'The femprj file path in the history csv is not found or valid. '
                alerts = self.alert_region.create_alerts(msg, color='danger')
                return alerts

            if model_name is None:
                msg = 'The model name in the history csv is not found.'
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
                msg = f'Failed to open {pdt_path}.'
                alerts = self.alert_region.create_alerts(msg, color='danger')
                return alerts

            raise PreventUpdate

        # make a new AnalysisModel that is the <model_name>_trial<n>
        # and reconstruct model.
        # ===== update parameter =====
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
                msg = ('Connection to Femtet is not established. '
                       'Launch Femtet and Open a project.')
                alerts = self.alert_region.create_alerts(msg, color='danger', current_alerts=current_alerts)
                return alerts

            # check selection
            if selection_data is None:
                msg = 'No result plot is selected.'
                alerts = self.alert_region.create_alerts(msg, color='danger', current_alerts=current_alerts)
                return alerts

            try:
                Femtet = self.femtet_control.fem.Femtet


                # check model to open is included in current project
                if (self.femtet_control.fem.femprj_path != Femtet.Project) \
                        or (self.femtet_control.fem.model_name not in Femtet.GetAnalysisModelNames_py()):
                    msg = (f'{self.femtet_control.fem.model_name} is not in current project. '
                           f'Please check opened project. '
                           f'For example, not "analysis model only" but your .femprj file.')
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
                df = self.application.history.local_data
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
                msg = '.femprj file path of the history csv is invalid. Please certify matching between csv and opening .femprj file.'
                new_alerts = self.alert_region.create_alerts(msg, 'warning', new_alerts)
            else:
                if not is_same_femprj:
                    msg = '.femprj file path of the history csv and opened in Femtet is inconsistent. Please certify matching between csv and .femprj file.'
                    new_alerts = self.alert_region.create_alerts(msg, 'warning', new_alerts)

            if model_name_on_history is None:
                msg = 'Analysis model name of the history csv is invalid. Please certify matching between csv and opening analysis model.'
                new_alerts = self.alert_region.create_alerts(msg, 'warning', new_alerts)
            else:
                if not is_same_model:
                    msg = 'Analysis model name of the history csv and opened in Femtet is inconsistent. Please certify matching between csv and opening analysis model.'
                    new_alerts = self.alert_region.create_alerts(msg, 'warning', new_alerts)

            if new_alerts == current_alerts:
                raise PreventUpdate

            return new_alerts

