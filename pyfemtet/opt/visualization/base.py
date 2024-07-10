# type hint
from dash.development.base_component import Component

# application
from dash import Dash
import webbrowser

# callback
from dash import Output, Input  # , State, no_update, callback_context
# from dash.exceptions import PreventUpdate

# components
# from dash import html, dcc
import dash_bootstrap_components

from pyfemtet.opt._femopt_core import History
from pyfemtet.opt.visualization.wrapped_components import html, dcc, dbc

# the others
from abc import ABC, abstractmethod
import logging
import psutil
from pyfemtet.logger import get_logger


dash_logger = logging.getLogger('werkzeug')
dash_logger.setLevel(logging.ERROR)

logger = get_logger('viewer')
logger.setLevel(logging.ERROR)


class AbstractPage(ABC):
    """Define content."""
    """

        =================
        |~:8080/rel_url | <---- page.rel_url
        =================
page.   |    | -------- |
title -->home| |      | |
        |    | |    <-------- sub_page.layout
        |    | -------- |
        | ^  |          |<--- page.layout
        ==|==============
          |
         sidebar

    """
    def __init__(self, title='base-page', rel_url='/', application=None):
        self.layout: Component = None
        self.rel_url = rel_url
        self.title = title
        self.application: PyFemtetApplicationBase = application
        self.subpages = []
        self.setup_component()
        self.setup_layout()

    def add_subpage(self, subpage: 'AbstractPage'):
        subpage.setup_component()
        subpage.setup_layout()
        self.subpages.append(subpage)

    def set_application(self, app):
        self.application = app

    @abstractmethod
    def setup_component(self):
        # noinspection PyAttributeOutsideInit
        self._component = html.Div('This is a abstract page.')

    @abstractmethod
    def setup_layout(self):
        self.layout = self._component

    def setup_callback(self):
        # app = self.application.app

        for subpage in self.subpages:
            subpage.set_application(self.application)
            subpage.setup_callback()

        # @app.callback(...)
        # def do_something():
        #     return ...


def _unused_port_number(start=49152):
    # "LISTEN" 状態のポート番号をリスト化
    used_ports = [conn.laddr.port for conn in psutil.net_connections() if conn.status == 'LISTEN']
    port = start
    for port in range(start, 65535 + 1):
        # 未使用のポート番号ならreturn
        if port not in set(used_ports):
            break
    if port != start:
        logger.warning(f'Specified port "{start}" seems to be used. Port "{port}" is used instead.')
    return port


class SidebarApplicationBase:
    """"""
    """ Define entire layout and callback.
    +------+--------+
    | side | con-   |
    | bar  | tent   |
    +------+--------+
       │      └─ pages (dict(href: str = layout: Component))
       └──────── nav_links (dict(order: float) = NavLink)
    """

    # port
    DEFAULT_PORT = 49152

    # members for sidebar application
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    }
    CONTENT_STYLE = {
        "margin-left": "18rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    }

    def __init__(self, title=None, subtitle=None):
        # define app
        self.title = title if title is not None else 'App'
        self.subtitle = subtitle if title is not None else ''
        self.app = Dash(
            __name__,
            external_stylesheets=[dash_bootstrap_components.themes.BOOTSTRAP],
            title=title,
            update_title=None,
        )
        self.pages = dict()
        self.nav_links = dict()
        self.page_objects = []

    def add_page(self, page: AbstractPage, order: int = None):
        page.set_application(self)
        self.page_objects.append(page)
        self.pages[page.rel_url] = page.layout
        if order is None:
            order = len(self.pages)
        self.nav_links[order] = dbc.NavLink(page.title, href=page.rel_url, active="exact")

    def setup_callback(self):
        for page in self.page_objects:
            page.setup_callback()

    def _setup_layout(self):
        # setup sidebar
        # https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar/

        # sidebar に表示される順に並び替え
        ordered_items = sorted(self.nav_links.items(), key=lambda x: x[0])
        ordered_links = [value for key, value in ordered_items]

        # sidebar と contents から app 全体の layout を作成
        sidebar = html.Div(
            [
                html.H2(self.title, className='display-4'),
                html.Hr(),
                html.P(self.subtitle, className='lead'),
                dbc.Nav(ordered_links, vertical=True, pills=True),
            ],
            style=self.SIDEBAR_STYLE,
        )
        content = html.Div(id="page-content", style=self.CONTENT_STYLE)
        self.app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

        # sidebar によるページ遷移のための callback
        @self.app.callback(Output(content.id, "children"), [Input("url", "pathname")])
        def switch_page_content(pathname):
            if pathname in list(self.pages.keys()):
                return self.pages[pathname]

            else:
                return html.Div(
                    [
                        html.H1("404: Not found", className="text-danger"),
                        html.Hr(),
                        html.P(f"The pathname {pathname} was not recognised..."),
                    ],
                    className="p-3 bg-light rounded-3",
                )

    def run(self, host='localhost', port=None, debug=False):
        self._setup_layout()
        port = port or self.DEFAULT_PORT
        # port を検証
        port = _unused_port_number(port)
        # ブラウザを起動
        if host == '0.0.0.0':
            webbrowser.open(f'http://localhost:{str(port)}')
        else:
            webbrowser.open(f'http://{host}:{str(port)}')
        self.app.run(debug=debug, host=host, port=port)


class PyFemtetApplicationBase(SidebarApplicationBase):
    """"""
    """
        +------+--------+
        | side | con-   |
        | bar  | tent   |
        +--^---+--^-----+
           │      └─ pages (dict(href: str = layout: Component))
           └──────── nav_links (dict(order: float) = NavLink)

        Accessible members:
        - history: History
           └ local_df: pd.DataFrame
        - app: Dash

    """

    def __init__(
            self,
            title=None,
            subtitle=None,
            history: History = None,
    ):
        # register arguments
        self.history = history  # include actor
        super().__init__(title, subtitle)


def check_page_layout(page_cls: type):
    home_page = page_cls()  # required
    application = PyFemtetApplicationBase(title='test-app')
    application.add_page(home_page, 0)
    application.run(debug=True)


if __name__ == '__main__':

    # template
    g_home_page = AbstractPage('home-page')  # required
    g_page1 = AbstractPage('page-1', '/page-1')

    g_application = SidebarApplicationBase(title='test-app')
    g_application.add_page(g_home_page, 0)
    g_application.add_page(g_page1, 1)

    # g_application.setup_callback()

    g_application.run(debug=True)
