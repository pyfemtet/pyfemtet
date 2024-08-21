# type hint
from typing import Callable, Any

# ui
import dash_bootstrap_components as dbc

# callback
from dash import Output, Input, State
from dash.exceptions import PreventUpdate

# components
from dash_application import BaseComplexComponent


class AlertRegion(BaseComplexComponent):
    # noinspection PyUnresolvedReferences,GrazieInspection
    """
        Examples:

            >>> def check_something(_, num):
            ...     if num < 10:
            ...         return 'num is smaller than 10.'
            ...
            >>> alert_region = AlertRegion(application)
            >>> alert_region.add_callback(
            ...     [Input(check_button, 'n_clicks')],
            ...     [State(user_input, 'children')],
            ...     check_something
            ... )
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
    def setup_components(self):
        self.alerts: dbc.CardBody
        self.clear_alerts_button: dbc.Button

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

    def setup_callbacks(self):
        # ===== clear alerts =====
        @self.application.app.callback(
            Output(self.alerts, 'children', allow_duplicate=True),
            Input(self.clear_alerts_button, 'n_clicks'),
            prevent_initial_call=True,
        )
        def clear_alerts(_): return []


    def add_callback(self, inputs: list[Input], states: list[State], message_generator: Callable[[Any], str or tuple[str, str] or None]) -> None:
        """

        Args:
            inputs: Trigger to start to run message_generator(). Should be args of message_generator().
            states: Should be args of message_generator().
            message_generator: return msg or (msg, color) or None. If it returns None, alerts aren't updated.

        Returns:
            None

        """

        # ===== add alert =====
        @self.application.app.callback(
            Output(alert_region.alerts, 'children', allow_duplicate=True),
            *inputs,
            *states,
            State(alert_region.alerts, 'children'),
            prevent_initial_call=True,
        )
        def add_alert(*args):
            current_alerts = args[-1]
            args = args[:-1]

            ret = message_generator(*args)

            # ret が None の場合、警告をアップデートしない
            if ret is None:
                raise PreventUpdate

            # ret が tuple の場合、警告文 + 色と見做す
            elif isinstance(ret, list) or isinstance(ret, tuple):
                print(ret)
                msg, color = ret

            # ret が str の場合、警告文と見做す
            else:
                msg, color = ret, None

            alerts = self.create_alerts(msg, color, current_alerts)
            return alerts


    @staticmethod
    def create_alerts(msg, color=None, current_alerts: list[dbc.Alert] = None) -> list[dbc.Alert]:
        color = color or 'secondary'
        current_alerts = current_alerts or []

        new_alert = dbc.Alert(
            msg,
            dismissable=True,
            color=color,
        )

        new_alerts = [new_alert]
        new_alerts.extend(current_alerts)

        return new_alerts


if __name__ == '__main__':
    from dash import html
    import dash_bootstrap_components as dbc
    from dash_application import MultiPageApplication, DebugPage

    g_application = MultiPageApplication()

    alert_region = AlertRegion(g_application)

    def check_number(n_clicks, num):
        if n_clicks < int(num):
            return f'n_clicks is smaller than {int(num)}.'
        elif n_clicks < 2*int(num):
            return f'n_clicks is smaller than {2*int(num)}.', 'danger'

    check_button = dbc.Button('click!')
    threshold = html.Div('3')

    alert_region.add_callback(
        [Input(check_button, 'n_clicks')],
        [State(threshold, 'children')],
        check_number
    )

    page = DebugPage(
        g_application,
        components=[check_button, threshold],
        complex_components=[alert_region],
    )

    g_application.add_page(page=page, rel_url='/', title='debug')
    g_application.run(debug=True)

