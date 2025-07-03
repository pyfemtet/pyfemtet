import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from pyfemtet.logger import get_module_logger


__all__ = [
    'parallel_plot',
]


def parallel_plot(df: pd.DataFrame) -> go.Figure | str:

    logger = get_module_logger('opt.parallel_plot')

    if any([
        not ('float' in dtype.name or 'int' in dtype.name)
        for dtype in df.dtypes.values
    ]):
        logger.error('Not implemented: including categorical parameters.')
        # fig = px.parallel_categories()
        fig = 'Not implemented: including categorical parameters.'

    else:
        fig = px.parallel_coordinates(
            df,
            dimensions=df.columns,
            color=df.columns[-1],
            color_continuous_scale=px.colors.sequential.Turbo
        )

    return fig
