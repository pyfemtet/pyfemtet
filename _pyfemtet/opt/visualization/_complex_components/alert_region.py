# type hint
from typing import List
from dash.development.base_component import Component

# callback
from dash import Output, Input, State, no_update, callback_context
from dash.exceptions import PreventUpdate

# components
from pyfemtet.opt.visualization._wrapped_components import html, dcc, dbc

from pyfemtet.opt.visualization._base import AbstractPage, logger


class AlertRegion(AbstractPage):

    def setup_component(self):
        # alert
        # noinspection PyAttributeOutsideInit
        self.alert_region = dbc.CardBody(children=[], id='alerts-card-body')

        # clear alert
        # noinspection PyAttributeOutsideInit
        self.clear_alert_button = dbc.Button(
            children='Clear messages',
            id='clear-messages-button',
            color='secondary',
            outline=True,
            className="position-relative",
        )

    def setup_layout(self):
        self.layout = dbc.Card(
            [
                dbc.CardHeader(
                    children=self.clear_alert_button,
                    className='d-flex',                # align right
                    style={'justify-content': 'end'},  # align right
                ),
                self.alert_region,
            ]
        )

    def setup_callback(self):
        app = self.application.app

        # ===== clear alerts ==-==
        @app.callback(
            Output(self.alert_region.id, 'children', allow_duplicate=True),
            Input(self.clear_alert_button.id, self.clear_alert_button.Prop.n_clicks),
            prevent_initial_call=True,  # required if allow_duplicate=True
        )
        def clear_alerts(_):
            return []

    def create_alerts(self, msg, color='secondary', current_alerts=None) -> List[Component]:

        if current_alerts is None:
            current_alerts = []

        new_alert = dbc.Alert(
            msg,
            id=f'alert-{len(current_alerts) + 1}',
            dismissable=True,
            color=color,
        )

        new_alerts = [new_alert]
        new_alerts.extend(current_alerts)

        return new_alerts
