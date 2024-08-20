import os
import webbrowser
from typing import Any, Dict, Optional
import threading
from time import sleep
from abc import ABC, abstractmethod

import pandas as pd

from flask import Flask

from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.development.base_component import Component

from pyfemtet.opt._femopt_core import History
from pyfemtet.opt.dash_ui import utils


class BaseApplication(object):

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


class BaseMultiPageApplication(BaseApplication):

    def __init__(self, app_title: str = None, app_description: str = None):
        super().__init__()
        self.nav_links: Dict[int, "dbc.NavLink"] = dict()  # order on sidebar -> link (contains rel_url)
        self.pages: Dict[str: "BaseComplexComponent"] = dict()  # rel_url -> page (contains layout)
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
        """Create sidebar

        Args:
            app_title (str):
            app_description (str):

        Returns:
            None

        """

        # sidebar に表示される順に並び替え
        ordered_nav_links = [v for k, v in sorted(self.nav_links.items(), key=lambda x: x[0])]

        # sidebar と contents から app 全体の layout を作成
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

        content = html.Div(id="page-content")
        self.app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

        @self.app.callback(Output("page-content", "children"), [Input("url", "pathname")])
        def render_page_content(pathname):
            if pathname in list(self.pages.keys()):
                return self.pages[pathname].layout

            # If the user tries to reach a different page, return a 404 message
            return html.Div(
                [
                    html.H1("404: Not found", className="text-danger"),
                    html.Hr(),
                    html.P(f"The pathname {pathname} was not recognised..."),
                ],
                className="p-3 bg-light rounded-3",
            )

        @self.app.callback(
            Output("sidebar", "className"),
            [Input("sidebar-toggle", "n_clicks")],
            [State("sidebar", "className")],
        )
        def toggle_classname(n, classname):
            if n and classname == "":
                return "collapsed"
            return ""

        @self.app.callback(
            Output("collapse", "is_open"),
            [Input("navbar-toggle", "n_clicks")],
            [State("collapse", "is_open")],
        )
        def toggle_collapse(n, is_open):
            if n:
                return not is_open
            return is_open


class BasePyFemtetApplication(BaseMultiPageApplication):

    def __init__(self, history: History):
        super().__init__()
        self.history = history
        self._df = history.get_df()

    def get_df(self) -> pd.DataFrame:
        """Get df from history.

        Please note that history.get_df() accesses Actor,
        but the dash callback cannot access to Actor,
        so the _df should be updated by self._sync().

        """
        return self._df

    def run(self, host=None, debug=False, launch_browser=False):
        # _sync も app と同様 **デーモンスレッドで** 並列実行
        sync_thread = threading.Thread(
            target=self._sync,
            args=(),
            kwargs={},
            daemon=True,
        )
        sync_thread.start()
        super().run(host, debug, launch_browser)

    def _sync(self):
        while True:
            # df はここでのみ上書きされ、dashboard から書き戻されることはない
            self._df = self.history.get_df()

            # status は...
            print('status の共有方法をまだ実装していません。')

            # lock が release されていれば、これ以上 sync が実行される必要はない
            if not self.lock_to_terminate.locked():
                break
            sleep(1)


class BaseComplexComponent(ABC):
    def __init__(self, application: BaseApplication):
        # declare
        self.layout: Component = ...
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


if __name__ == '__main__':
    import numpy as np
    from dash import dcc
    import plotly.express as px


    class SubPage(BaseComplexComponent):
        # noinspection PyAttributeOutsideInit
        def setup_components(self) -> None:
            df = pd.DataFrame(dict(a=[0, 1, 2], b=[0, 1, 4]))
            fig = px.scatter(df, x='a', y='b')
            self.graph = dcc.Graph(
                figure=fig,
                style={'width': "100%", 'height': "100%", },
            )
            self.location = dcc.Location(id='location' + str(np.random.randint(0, 100)))

        def setup_callbacks(self) -> None:
            # @self.application.app.callback(
            #     Output(..., ...),
            #     Output(..., ..., allow_duplicate=True),
            #     Input(..., ...),
            #     State(..., ...),
            #     prevent_initial_call=True,
            # )
            # def some_callback(*args, **kwargs) -> Any: ...

            @self.application.app.callback(
                # Output({'type': 'example-graph', 'index': 2}, 'figure'),
                Output(self.graph, 'figure'),
                # Input(self.location, 'pathname'),  # 遷移前と遷移後の両方で実行される
                # Input('url', 'pathname'),  # 一個前しか実行しない
                Input(self.graph, 'children'),  # ページがロードされたときに実行される
                # Input(self.location, 'children'),  # ページがロードされたときに実行される
                # prevent_initial_call=True,
            )
            def update_graph(*args, **kwargs):
                df = pd.DataFrame(dict(a=np.random.rand(5), b=np.random.rand(5)))
                fig = px.scatter(df, x='a', y='b')
                return fig

            pass

        def setup_layout(self) -> None:
            container = dbc.Container(
                children=[
                    self.location,
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[
                                    dcc.Loading(
                                        self.graph,
                                        style={'width': "100%", 'height': "100%", },
                                        parent_style={'width': "100%", 'height': "100%", },
                                        overlay_style={'width': "100%", 'height': "100%", },
                                    )
                                ],
                                style={'display': 'flex', 'flex-direction': 'column', 'flex': '1 1 auto',
                                       'overflow': 'hidden'},
                            )
                        ],
                        style={'display': 'flex', 'flex-direction': 'column', 'flex': '1 1 auto', 'overflow': 'hidden'},
                    ),
                ],
                style={'height': '90vh', 'display': 'flex', 'flex-direction': 'column', 'overflow': 'hidden'},
                fluid=True,
            )
            self.layout = container


    class SubPage2(SubPage):
        pass


    class SubPage3(SubPage):
        pass


    g_application = BaseMultiPageApplication()

    g_home_page = SubPage(g_application)
    g_application.add_page(g_home_page, 'home', "/")

    g_sub_page = SubPage2(g_application)
    g_application.add_page(g_sub_page, 'sub', "/sub")

    g_sub_page2 = SubPage3(g_application)
    g_application.add_page(g_sub_page2, 'sub2', "/sub2")

    g_application.run(launch_browser=True, debug=False)
