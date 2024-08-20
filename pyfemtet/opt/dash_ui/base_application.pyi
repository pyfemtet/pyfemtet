from typing import Any, Dict, Optional
import threading

import pandas as pd

from dash import Dash
import dash_bootstrap_components as dbc
from dash.development.base_component import Component

from pyfemtet.opt._femopt_core import History


class BaseApplication(object):
    app: Dash
    lock_to_terminate: threading.Lock  # If released, the daemon thread is ready to be terminated along with the server.
    def __init__(self) -> None: ...  # Set app with external stylesheets.
    def run(self, host: str = None, debug: bool = False, launch_browser: bool = False) -> None: ...


class BaseMultiPageApplication(BaseApplication):
    nav_links: Dict[int, "dbc.NavLink"]  # order on sidebar -> link (contains rel_url)
    pages: Dict[str: "BaseComplexComponent"]  # rel_url -> page (contains layout)
    app_title: str
    app_description: str
    def add_page(self, page: BaseComplexComponent, title: str, rel_url: str, order: int = None) -> None: ...
    def run(self, host: str = None, debug: bool = False, launch_browser: bool = False) -> None: ...  # Run setup_callbacks of components and run server.
    def _create_sidebar(self) -> None: ...  # create sidebar from added pages


class BasePyFemtetApplication(BaseMultiPageApplication):
    history: History
    _df: pd.DataFrame
    def __init__(self, history: History): ...
    def get_df(self) -> pd.DataFrame: ...  # In Dash callbacks, use this.
    def run(self, host: str = None, debug: bool = False, launch_browser: bool = False) -> None: ...  # must call _sync()
    def _sync(self) -> None: ...  # sync Actor data with dash application


class BaseComplexComponent(object):
    layout: Component
    application: BaseApplication
    def __init__(self, application: BaseApplication): ...
    def setup_components(self) -> None: ...
    def setup_callbacks(self) -> None: ...
    def setup_layout(self) -> None: ...
