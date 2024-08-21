import threading

from dash import Dash
import dash_bootstrap_components as dbc
from dash.development.base_component import Component


class BaseApplication(object):
    app: Dash
    lock_to_terminate: threading.Lock  # If released, the daemon thread is ready to be terminated along with the server.
    def __init__(self) -> None: ...  # Set app with external stylesheets.
    def run(self, host: str = None, debug: bool = False, launch_browser: bool = False) -> None: ...


class MultiPageApplication(BaseApplication):
    nav_links: dict[int, "dbc.NavLink"]  # order on sidebar -> link (contains rel_url)
    pages: dict[str: "BaseComplexComponent"]  # rel_url -> page (contains layout)
    app_title: str
    app_description: str
    def add_page(self, page: BaseComplexComponent, title: str, rel_url: str, order: int = None) -> None: ...
    def run(self, host: str = None, debug: bool = False, launch_browser: bool = False) -> None: ...  # Run setup_callbacks of components and run server.
    def _create_sidebar(self) -> None: ...  # create sidebar from added pages


class BaseComplexComponent(object):
    layout: Component or list[Component]
    hidden_layout: Component or list[Component]
    application: BaseApplication
    def __init__(self, application: BaseApplication): ...
    def setup_components(self) -> None: ...
    def setup_callbacks(self) -> None: ...
    def setup_layout(self) -> None: ...


class DebugPage(BaseComplexComponent):
    def __init__(
            self,
            application: BaseApplication,
            components: list[Component] = None,
            hidden_components: list[Component] = None,
            complex_components: list[BaseComplexComponent] = None,
    ):
        pass
