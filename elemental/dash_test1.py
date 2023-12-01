# 目的：外から与えられた値に対する scatter matrix の live-update
# 目標：scatter matrix の live-update


import dash
from dash import Dash, dcc, html, Input, Output, callback
import plotly
import plotly.graph_objects as go

import numpy as np
import pandas as pd

app = Dash(__name__)
app.layout = html.Div(
    html.Div(
        [
            dcc.Graph(id='live-update-graph'),
            dcc.Interval(
                id='interval-component',
                interval=3*1000, # in milliseconds
                n_intervals=0
                )
            ]
        )
    )


# Multiple components can update everytime interval gets fired.
@callback(
    Output('live-update-graph', 'figure'),
    Input('interval-component', 'n_intervals')
    )
def update_graph_live(n):
    
    # data
    _data = np.random.rand(5,5)
    data = pd.DataFrame(_data, columns='A B C D E'.split())

    # pairplot
    trace=go.Splom(
        dimensions=[dict(label=c, values=data[c]) for c in data.columns],
        diagonal_visible=False,
        showupperhalf=False,
        )
    
    # figure
    fig = go.Figure(
        data=trace,
        )
    
    fig.update_layout(
        title='random data set'
        )


    # # Create the graph with subplots
    # fig = plotly.tools.make_subplots(rows=2, cols=1)
    # # fig['layout']['margin'] = {
    # #     'l': 30, 'r': 10, 'b': 30, 't': 10
    # # }
    # # fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}
    # fig.append_trace({
    #     'x': data['A'],
    #     'y': data['B'],
    #     'name': 'Altitude',
    #     'mode': 'lines+markers',
    #     'type': 'scatter'
    # }, 1, 1)
    # fig.append_trace({
    #     'x': data['A'],
    #     'y': data['C'],
    #     'text': data['D'],
    #     'name': 'Longitude vs Latitude',
    #     'mode': 'lines+markers',
    #     'type': 'scatter'
    # }, 2, 1)

    return fig


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8080)
