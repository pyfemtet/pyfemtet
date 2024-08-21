import os
import webbrowser
import threading
from abc import ABC, abstractmethod

import pandas as pd

from flask import Flask

from dash import Dash, html, Input, Output, State
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.development.base_component import Component

from dash_application import utils


class BaseApplication(object):
    """Base application class containing flask server and dash app.

    If not debug mode, the app runs on daemon thread.
    Release BaseApplication.lock_to_terminate to
    terminate dash app when the main process exit.

    Notes:

        Cannot terminate dash app by ctrl+c.

    """

    def __init__(self) -> None:
        """Set app with external stylesheets."""
        # declare
        self.app = ...
        self.lock_to_terminate = ...

        # init
        server = Flask(
            __name__,
            static_folder=os.path.join(os.path.dirname(__file__), 'assets'),
        )
        self.app = Dash(
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            # these meta_tags ensure content is scaled correctly on different devices
            # see: https://www.w3schools.com/css/css_rwd_viewport.asp for more
            meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1"}
            ],
            server=server,
        )

    def run(self, host: str = None, debug: bool = False, launch_browser: bool = False) -> None:
        """Run dash application that will terminate when self.lock.release(). """

        # certify host
        if host is None:
            host = 'localhost'

        # certify port
        if ':' in host:
            host, port = host.split(':')
            port = int(port)
        else:
            port = utils.DEFAULT_PORT

        # check port unused
        if not debug:
            port = utils.get_unused_port_number(port)

        # ブラウザを起動
        if launch_browser and not debug:
            if host == '0.0.0.0':
                webbrowser.open(f'http://localhost:{str(port)}')
            else:
                webbrowser.open(f'http://{host}:{str(port)}')

        # 実行
        if debug:
            self.app.run(debug=debug, host=host, port=port)

        else:
            # 終了待ちのためのロックをかける
            self.lock_to_terminate = threading.Lock()
            self.lock_to_terminate.acquire()

            # run
            app_thread = threading.Thread(
                target=self.app.run,
                kwargs=dict(debug=debug, host=host, port=port),
                daemon=True,  # trick: by this, this thread will terminate when the process is terminated.
            )
            app_thread.start()

            # 終了待ちロックの解除を待つ
            self.lock_to_terminate.acquire()
            self.lock_to_terminate.release()


class MultiPageApplication(BaseApplication):
    # noinspection GrazieInspection
    """

            Notes:

                Basic structure of ui is following.

            ::

                  +---------------- header of sidebar
                  |
                +-v-----------+
                |   | [     ]<----- hidden_layout
                |---| +-----+ |
                |   | |#####|<----- layout
                |   | +-----+ |
                +-------------+
                  ^      ^
                  |      +--------- content
                  +---------------- sidebar

        """


    def __init__(self, app_title: str = None, app_description: str = None):
        super().__init__()
        self.nav_links: dict[int, "dbc.NavLink"] = dict()  # order on sidebar -> link (contains rel_url)
        self.pages: dict[str: "BaseComplexComponent"] = dict()  # rel_url -> page (contains layout)
        self.app_title: str = app_title or 'Dash UI'
        self.app_description: str = app_description or 'Dash UI with a sidebar containing navigation links.'

    def add_page(self, page: "BaseComplexComponent", title: str, rel_url: str, order: int = None) -> None:
        """Add a page.

        Args:
            page (BaseComplexComponent):
            title (str): String in sidebar.
            rel_url (str): Must starts with '/'.
            order (int, optional): Order in sidebar.

        Returns:
            None

        """
        self.pages[rel_url] = page
        order = order or len(self.pages)
        self.nav_links[order] = dbc.NavLink(title, href=rel_url, active="exact")

    def run(self, host: str = None, debug: bool = False, launch_browser: bool = False) -> None:
        """Run setup_callbacks of components and run server.

        Args:
            host (str, optional): The format is "x.x.x.x:x".
            debug (bool, optional): Run debug mode  or not. Defaults to False.
            launch_browser (bool, optional): Launch browser automatically or not. Defaults to False.

        Returns:
            None

        """
        # setup sidebar
        self._create_sidebar()
        super().run(host, debug, launch_browser)

    def _create_sidebar(self) -> None:

        # sidebar に表示される順に並び替え
        ordered_nav_links = [v for k, v in sorted(self.nav_links.items(), key=lambda x: x[0])]

        # sidebar の header を作成
        sidebar_header = dbc.Row(
            [
                dbc.Col(html.H2(self.app_title, className="display-4")),
                dbc.Col(
                    [
                        html.Button(
                            # use the Bootstrap navbar-toggler classes to style
                            html.Span(className="navbar-toggler-icon"),
                            className="navbar-toggler",
                            # the navbar-toggler classes don't set color
                            style={
                                "color": "rgba(0,0,0,.5)",
                                "border-color": "rgba(0,0,0,.1)",
                            },
                            id="navbar-toggle",
                        ),
                        html.Button(
                            # use the Bootstrap navbar-toggler classes to style
                            html.Span(className="navbar-toggler-icon"),
                            className="navbar-toggler",
                            # the navbar-toggler classes don't set color
                            style={
                                "color": "rgba(0,0,0,.5)",
                                "border-color": "rgba(0,0,0,.1)",
                            },
                            id="sidebar-toggle",
                        ),
                    ],
                    # the column containing the toggle will be only as wide as the
                    # toggle, resulting in the toggle being right aligned
                    width="auto",
                    # vertically align the toggle in the center
                    align="center",
                ),
            ]
        )

        # sidebar を作成
        sidebar = html.Div(
            [
                sidebar_header,
                # we wrap the horizontal rule and short blurb in a div that can be
                # hidden on a small screen
                html.Div(
                    [
                        html.Hr(),
                        html.P(
                            self.app_description,
                            className="lead",
                        ),
                    ],
                    id="blurb",
                ),
                # use the Collapse component to animate hiding / revealing links
                dbc.Collapse(
                    dbc.Nav(
                        children=ordered_nav_links,
                        vertical=True,
                        pills=True,
                    ),
                    id="collapse",
                ),
            ],
            id="sidebar",
        )

        # content を作成
        content = html.Div(id="page-content")

        # 全体 layout を作成
        self.app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

        # content の切り替え callback
        @self.app.callback(Output("page-content", "children"), [Input("url", "pathname")])
        def render_page_content(pathname):
            if pathname in list(self.pages.keys()):
                return [
                    html.Div(self.pages[pathname].layout),
                    html.Div(self.pages[pathname].hidden_layout)
                ]

            # If the user tries to reach a different page, return a 404 message
            return html.Div(
                [
                    html.H1("404: Not found", className="text-danger"),
                    html.Hr(),
                    html.P(f"The pathname {pathname} was not recognised..."),
                ],
                className="p-3 bg-light rounded-3",
            )

        # sidebar の折り畳み callback
        @self.app.callback(
            Output("sidebar", "className"),
            [Input("sidebar-toggle", "n_clicks")],
            [State("sidebar", "className")],
        )
        def toggle_classname(n, classname):
            if n and classname == "":
                return "collapsed"
            return ""

        # sidebar の折り畳み callback その 2
        @self.app.callback(
            Output("collapse", "is_open"),
            [Input("navbar-toggle", "n_clicks")],
            [State("collapse", "is_open")],
        )
        def toggle_collapse(n, is_open):
            if n:
                return not is_open
            return is_open


class BaseComplexComponent(ABC):
    """Class for complex components."""

    def __init__(self, application: BaseApplication):
        # declare
        self.layout: Component or list[Component] = html.Div()
        self.hidden_layout: Component or list[Component] = html.Div()  # html.Data など、アプリケーションに含めたいが非表示にしたいもの
        self.application: BaseApplication = application

        # init
        self.setup_components()
        self.setup_callbacks()
        self.setup_layout()

    @abstractmethod
    def setup_components(self) -> None: ...

    @abstractmethod
    def setup_callbacks(self) -> None: ...

    @abstractmethod
    def setup_layout(self) -> None: ...


class DebugPage(BaseComplexComponent):

    def __init__(
            self,
            application,
            components: list[Component] = None,
            hidden_components: list[Component] = None,
            complex_components: list[BaseComplexComponent] = None,
    ):
        super().__init__(application)
        self.layout = components or []
        self.hidden_layout = hidden_components or []
        complex_components = complex_components or []
        self.layout.extend(c.layout for c in complex_components)
        self.hidden_layout.extend(c.hidden_layout for c in complex_components)

    def setup_components(self): pass
    def setup_callbacks(self): pass
    def setup_layout(self): pass


if __name__ == '__main__':

    g_application = MultiPageApplication()

    g_home_page = DebugPage(g_application)
    g_application.add_page(g_home_page, 'home', "/")

    g_sub_page = DebugPage(g_application)
    g_application.add_page(g_sub_page, 'sub', "/sub")

    g_sub_page2 = DebugPage(g_application)
    g_application.add_page(g_sub_page2, 'sub2', "/sub2")

    g_application.run(launch_browser=True, debug=False)
