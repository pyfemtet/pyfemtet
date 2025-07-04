import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pyfemtet.logger import get_module_logger


__all__ = [
    'contour_creator',
]


def contour_creator(df: pd.DataFrame) -> go.Figure | str:

    logger = get_module_logger('opt.contour_creator')

    target_column = tuple(df.columns)[-1]
    explain_columns = [column for column in df.columns if column != target_column]

    subplots = make_subplots(
        rows=len(explain_columns),
        cols=len(explain_columns),
        shared_xaxes=True,
        shared_yaxes=True,
        row_titles=explain_columns,
        column_titles=explain_columns,
        start_cell='bottom-left',
    )

    is_first = True

    for r, r_key in enumerate(explain_columns):
        for c, c_key in enumerate(explain_columns):

            r_dtype = df.dtypes[r_key]
            c_dtype = df.dtypes[c_key]

            if not ('float' in r_dtype.name or 'int' in r_dtype.name):
                logger.error(f'dtype is {r_dtype}. Not implemented.')
                return 'Not implemented: including categorical parameters.'
            if not ('float' in c_dtype.name or 'int' in c_dtype.name):
                logger.error(f'dtype is {c_dtype}. Not implemented.')
                return 'Not implemented: including categorical parameters.'

            x = df[c_key]
            y = df[r_key]
            z = df[target_column]

            scatter = go.Scatter(
                x=x, y=y, mode='markers',
                marker=go.scatter.Marker(
                    symbol='circle',
                    color='black',
                    size=5,
                    line=dict(
                        color='white',
                        width=1,
                    )
                ),
                name='points (click to switch visibility)',
                legendgroup='points (click to switch visibility)',
                showlegend=is_first,
            )

            is_first = False

            if r == c:
                pass

            else:

                contour = go.Contour(
                    x=x, y=y, z=z,
                    connectgaps=True,
                    name=f'contour of\n{target_column}',
                    colorscale='Turbo',
                )

                subplots.add_trace(contour, row=r + 1, col=c + 1)

            subplots.add_trace(scatter, row=r + 1, col=c + 1)
            subplots.update_layout(
                legend=dict(
                    orientation='h',
                    xanchor='center',
                    x=0.5,
                    yanchor='bottom',
                    y=-0.2,
                    bgcolor='rgba(0, 0, 0, 0.15)',
                ),
                # margin=dict(b=50),
            )

    return subplots


if __name__ == '__main__':
    import numpy as np
    contour_creator(pd.DataFrame(dict(
        x1=np.random.rand(20),
        x2=np.random.rand(20),
        x3=np.random.rand(20),
        x4=np.random.rand(20),
        y=np.random.rand(20),
    ))).show()
