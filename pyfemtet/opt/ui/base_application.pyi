import pandas as pd

from dash_application import MultiPageApplication

from pyfemtet.opt._femopt_core import History


class BasePyFemtetApplication(MultiPageApplication):
    history: History
    _df: pd.DataFrame
    def __init__(self, history: History): ...
    def get_df(self) -> pd.DataFrame: ...  # In Dash callbacks, use this.
    def run(self, host: str = None, debug: bool = False, launch_browser: bool = False) -> None: ...  # must call _sync()
    def _sync(self) -> None: ...  # sync Actor data with dash application
