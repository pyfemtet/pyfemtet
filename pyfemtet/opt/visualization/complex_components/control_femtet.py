# type hint
from dash.development.base_component import Component

# callback
from dash import Output, Input, State, no_update, callback_context
from dash.exceptions import PreventUpdate

# components
from pyfemtet.opt.visualization.wrapped_components import html, dcc, dbc

# the others
import logging
import os
from enum import Enum
import json
# noinspection PyUnresolvedReferences
from pythoncom import com_error

from pyfemtet.opt.visualization.base import PyFemtetApplicationBase, AbstractPage, logger
from pyfemtet.opt.interface._femtet import FemtetInterface
from pyfemtet.message import Msg


class FemtetState(Enum):
    unconnected = -1
    missing = 0
    connected_but_empty = 1
    connected = 2


class FemtetControl(AbstractPage):

    def __init__(self):
        super().__init__()
        self.fem: FemtetInterface = None

    def setup_component(self):

        # noinspection PyAttributeOutsideInit
        self.dummy = html.Div(
            id='control-femtet-dummy',
            hidden=True,
        )

        # button
        # noinspection PyAttributeOutsideInit
        self.connect_femtet_button = dbc.Button(
            children=Msg.LABEL_CONNECT_FEMTET_BUTTON,
            id='connect-femtet-button',
            outline=True,
            color='primary',
            className="position-relative",  # need to show badge
        )

        # noinspection PyAttributeOutsideInit
        self.connect_femtet_button_spinner = dbc.Spinner(
            self.connect_femtet_button,
            color='primary',
        )

        # store femtet connection state
        # noinspection PyAttributeOutsideInit
        self.femtet_state_prop = 'data-femtet-state'  # must start with "data-"
        # noinspection PyAttributeOutsideInit
        self.femtet_state = html.Data(
            id='femtet-state',
            **{
                self.femtet_state_prop: {
                    'pid': 0,
                    'alert': '',
                }
            }
        )

    def setup_layout(self):
        self.layout = [self.dummy, self.femtet_state]

    def setup_callback(self):
        # setup callback of subpages
        super().setup_callback()

        app = self.application.app

        # ===== launch femtet and show spinner =====
        @app.callback(
            Output(self.femtet_state.id, self.femtet_state_prop),
            Output(self.connect_femtet_button.id, self.connect_femtet_button.Prop.children),  # for spinner
            Input(self.connect_femtet_button.id, self.connect_femtet_button.Prop.n_clicks),)
        def connect_femtet(_):

            # ignore when loaded
            if callback_context.triggered_id is None:
                raise PreventUpdate

            # create FemtetInterface arguments
            kwargs, warning_msg = self.create_femtet_interface_args()

            # try to launch femtet
            try:
                self.fem = FemtetInterface(**kwargs)

            except com_error:
                # model_name is not included in femprj
                kwargs.update(dict(model_name=None))
                self.fem = FemtetInterface(**kwargs)
                warning_msg = Msg.WARN_CSV_MODEL_NAME_IS_INVALID

            # if 'new' femtet, Interface try to terminate Femtet when the self.fem is reconstructed.
            self.fem.quit_when_destruct = False

            state_data = {
                'pid': self.fem.femtet_pid,
                'alert': warning_msg,
            }

            return state_data, no_update

    def create_femtet_interface_args(self) -> [dict, str]:
        """Returns the argument of FemtetInterface and warning message

        # femprj_path=None,
        # model_name=None,
        # connect_method='auto',  # dask worker では __init__ の中で 'new' にするので super() の引数にしない。（しても意味がない）
        # save_pdt='all',  # 'all' or None
        # strictly_pid_specify=True,  # dask worker では True にしたいので super() の引数にしない。
        # allow_without_project=False,  # main でのみ True を許容したいので super() の引数にしない。
        # open_result_with_gui=True,
        # parametric_output_indexes_use_as_objective=None,

        """

        kwargs = dict(
            connect_method='auto',
            femprj_path=None,
            model_name=None,
            allow_without_project=True,
        )

        # check holding history
        if self.application.history is None:
            return kwargs, Msg.ERR_HISTORY_CSV_NOT_READ

        # get metadata
        additional_metadata = self.application.history.metadata[0]

        # check metadata exists
        if additional_metadata == '':
            return kwargs, Msg.WARN_INVALID_METADATA

        # check the metadata is valid json
        try:
            d = json.loads(additional_metadata)
            femprj_path = os.path.abspath(d['femprj_path'])
        except (TypeError, json.decoder.JSONDecodeError, KeyError):
            return kwargs, Msg.WARN_INVALID_METADATA

        # check containing femprj
        if femprj_path is None:
            return kwargs, Msg.WARN_INVALID_METADATA

        # check femprj exists
        if not os.path.exists(femprj_path):
            return kwargs, Msg.WARN_FEMPRJ_IN_CSV_NOT_FOUND

        # at this point, femprj is valid at least.
        kwargs.update({'femprj_path': femprj_path})

        # check model name
        model_name = d['model_name'] if 'model_name' in d.keys() else None
        msg = '' if model_name is not None else Msg.WARN_MODEL_IN_CSV_NOT_FOUND_IN_FEMPRJ
        kwargs.update({'model_name': model_name})
        return kwargs, msg

    def check_femtet_state(self) -> FemtetState:

        # check fem is initialized
        if self.fem is None:
            return FemtetState.missing

        Femtet = self.fem.Femtet

        # check Femtet is constructed
        if Femtet is None:
            return FemtetState.missing

        # check Femtet is alive
        if not self.fem.femtet_is_alive():
            return FemtetState.missing

        # check a project is opened
        if Femtet.Project == '':
            return FemtetState.connected_but_empty

        return FemtetState.connected
