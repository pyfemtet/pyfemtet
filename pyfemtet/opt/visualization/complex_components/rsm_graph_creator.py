from typing import Callable, List

import pandas as pd

import plotly.graph_objs as go
import plotly.express as px

from pyfemtet.opt._femopt_core import History



def rsm_3d_creator(
        history: History,
        df: pd.DataFrame,
        prm_names: List[str],
        obj_name: str,
        rsm: Callable
):
    """
    Parameters:
        rsm (Callable): A function what returns objectives from parameters.
    """
    # x = df[prm_names...]
    # y = rsm(x)[obj_name...]
    # fig = px....(x, y, ...)

    fig = go.Figure()

    return fig
