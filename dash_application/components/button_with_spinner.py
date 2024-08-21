# ui
from dash import html
import dash_bootstrap_components as dbc

# callback
from dash import Output, Input, State

# components
from dash_application import BaseComplexComponent


class ButtonWithSpinner(BaseComplexComponent):
    # noinspection PyUnresolvedReferences,GrazieInspection
    """
        Examples:

            >>> def do_something(args): ...  # doctest: +SKIP
            >>> button = ButtonWithSpinner(
            ...     application,
            ...     text = 'An awesome button!'
            ...     fun = do_something,
            ...     state = [State(...)],  # args
            ...     output = [Output(...)],  # return of do_something
            ... )
            ... # doctest: +SKIP

        Attributes:

            button (dbc.Button): button containing dbc.Spinner

            layout (dbc.Button): [button]
        ::

            [ ] <-- button == layout

            [C] <-- button.children == spinner
                    and button is disabled
                    during fun() running

    """
    """
        callback diagram:
            click
            |             ^
            +---------+   | # callback 1
            |         |   v
            data-n    disable buttons
            -button
            -clicks
            |      ^
            fun()  | # callback 2
            |      v
            enable buttons
    
    """

    DISABLED_SPINNER_STYLE = {'display': 'none'}
    BUTTON_CLICKED_PROP = 'data-n-button-clicked'

    def __init__(self, application, text, fun,
                 color: str = 'primary',
                 states: list[State] = None,
                 outputs: list[Output] = None,
                 ):
        self._text = text
        self._color = color
        self._fun = fun
        self._states: list[State] = states or []
        self._outputs: list[Output] = outputs or []
        super().__init__(application)

    # noinspection PyAttributeOutsideInit
    def setup_components(self):
        # declare
        self.button: dbc.Button
        self._spinner = dbc.Spinner
        self._button_clicked_data = html.Data

        # implement
        self._spinner = dbc.Spinner(size='sm', spinner_style=self.DISABLED_SPINNER_STYLE)
        self.button = dbc.Button([self._spinner, self._text], color=self._color)
        self._button_clicked_data = html.Data(**{self.BUTTON_CLICKED_PROP: 0})

    def setup_layout(self):
        self.layout = self.button
        self.hidden_layout = self._button_clicked_data

    def setup_callbacks(self):
        # ===== disable button when clicked immediately =====
        base_outputs = [
            Output(self.button, 'disabled', allow_duplicate=True),
            Output(self._spinner, 'spinner_style', allow_duplicate=True),
        ]
        base_states = [
            State(self._spinner, 'spinner_style')
        ]

        # create output for disable
        # add html.Data output to fire callback chain
        outputs = base_outputs.copy()  # weak copy
        outputs.append(
            Output(self._button_clicked_data, self.BUTTON_CLICKED_PROP)
        )
        # add html.Data state to fire callback chain
        states = base_states.copy()
        states.append(
            State(self._button_clicked_data, self.BUTTON_CLICKED_PROP)
        )

        @self.application.app.callback(
            *outputs,
            Input(self.button, 'n_clicks'),
            *states,
            prevent_initial_call=True,
        )
        def disable(n_clicks, spinner_style, n_fired):
            # remove {'display': 'none'} from current style
            if 'display' in spinner_style.keys():
                spinner_style.pop('display')
            return True, spinner_style, n_fired + 1

        # create outputs for enable / fun
        outputs = base_outputs.copy()
        outputs.extend(self._outputs)

        states = base_states.copy()
        states.extend(self._states)

        # ===== disable button when clicked immediately =====
        @self.application.app.callback(
            *outputs,
            Input(self._button_clicked_data, self.BUTTON_CLICKED_PROP),
            *states,
            prevent_initial_call=True,
        )
        def enable(n_fired, spinner_style, *args):
            spinner_style.update({'display': 'none'})
            ret = self._fun(*args)

            if len(self._outputs) > 0:
                # output が指定されいてる場合は値を返す
                if isinstance(ret, list) or isinstance(ret, tuple):
                    return False, spinner_style, *ret
                else:
                    return False, spinner_style, ret
            else:
                # そうでない場合は ret (is None) は使わない
                return False, spinner_style


if __name__ == '__main__':
    from time import sleep
    from dash_application import DebugPage, MultiPageApplication

    g_application = MultiPageApplication()

    # no args, no return
    def do_something_1():
        sleep(3)
    button_1 = ButtonWithSpinner(
        application=g_application,
        text='no args, no return',
        fun=do_something_1,
    )

    # with args, no return
    def do_something_2(sec: int):
        sleep(sec)
    g_args_data = html.Data(**{'data-wait': 1})
    button_2 = ButtonWithSpinner(
        application=g_application,
        text='with args, no return',
        fun=do_something_2,
        states=[State(g_args_data, 'data-wait')],
    )

    # no args, with return
    def do_something_3() -> str:
        sleep(3)
        return 'hi-3'
    g_returns = html.Div('nothing returns')
    button_3 = ButtonWithSpinner(
        application=g_application,
        text='no args, with return',
        fun=do_something_3,
        outputs=[Output(g_returns, 'children', allow_duplicate=True)],
    )

    # with args, with return
    def do_something_4(sec: int) -> str:
        sleep(sec)
        return 'hi-4'
    button_4 = ButtonWithSpinner(
        application=g_application,
        text='with args, with return',
        fun=do_something_4,
        states=[State(g_args_data, 'data-wait')],
        outputs=[Output(g_returns, 'children', allow_duplicate=True)],
    )

    g_home_page = DebugPage(
        g_application,
        hidden_components=[g_returns, g_args_data],
        complex_components=[
            button_1,
            button_2,
            button_3,
            button_4,
        ]
    )
    g_application.add_page(g_home_page, 'home', "/")

    g_application.run(launch_browser=True, debug=True)
