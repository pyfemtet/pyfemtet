# type hint
from dash.development.base_component import Component

# ui
import dash_bootstrap_components as dbc

# callback
from dash import Output, Input, State, no_update, callback_context
from dash.exceptions import PreventUpdate

# components
from pyfemtet.opt.dash_ui.base_application import BaseComplexComponent


class AlertRegion(BaseComplexComponent):
    # noinspection PyUnresolvedReferences,GrazieInspection
    """
        Examples:

            >>> alert_region = AlertRegion(application)  # doctest: +SKIP
            >>> # ===== update alert =====
            >>> @app.callback(
            ...     Output(alert_region.alerts, 'children', allow_duplicate=True),
            ...     Input(...),
            ...     State(alert_region.alerts, 'children'),
            ...     prevent_initial_call=True,
            ... )
            ... def add_alert(..., current_alerts):
            ...     msg = ...
            ...     color = ...  # 'primary', 'secondary', 'warning', ...
            ...     return alert_region.create_alerts(msg, 'warning', current_alerts)
            ...

        Attributes:

            clear_alerts_button (dbc.Button):
            alerts (dbc.CardBody):

            layout (dbc.Card):
        ::

            +------------+
            |         []<--- clear_alerts_button (dbc.Button)
            |------------|
            | [        ]<--- (dbc.Alert)
            |            |<--- alerts (dbc.CardBody)
            |            |
            +------------+ <-- layout = dbc.Card
    """

    # noinspection PyAttributeOutsideInit
    def setup_component(self):
        # alert
        self.alerts = dbc.CardBody(children=[])

        # clear alert
        self.clear_alerts_button = dbc.Button(
            children='Clear messages',
            color='secondary',
            outline=True,
            className="position-relative",
        )

    def setup_layout(self):
        self.layout = dbc.Card(
            [
                dbc.CardHeader(
                    children=self.clear_alerts_button,
                    className='d-flex',  # align right
                    style={'justify-content': 'end'},  # align right
                ),
                self.alerts,
            ]
        )

    def setup_callback(self):
        app = self.application.app

        # ===== clear alerts =====
        @app.callback(
            Output(self.alerts, 'children', allow_duplicate=True),
            Input(self.clear_alerts_button, 'n_clicks'),
            prevent_initial_call=True,
        )
        def clear_alerts(_): return []

    @staticmethod
    def create_alerts(msg, color='secondary', current_alerts=None) -> list[Component]:
        if current_alerts is None:
            current_alerts = []

        new_alert = dbc.Alert(
            msg,
            dismissable=True,
            color=color,
        )

        new_alerts = [new_alert]
        new_alerts.extend(current_alerts)

        return new_alerts
