import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import plotly.express as px
import threading
import time

import numpy as np
import pandas as pd

class Sample:

    def update_value(self):
        while True:
            self.df = pd.DataFrame(
                {
                    'x':np.random.rand(10),
                    'y':np.random.rand(10),
                    }
                )
            time.sleep(1)

sample = Sample()

# application
app = dash.Dash(__name__)

# htmle elements
graph1 = dcc.Graph(id='scatter-graph', animate=False)
graph2 = dcc.Graph(id='line-graph', animate=False)

interval_component = dcc.Interval(
    id='interval-component',
    interval=2*1000,
    n_intervals=0
    )


app.layout = html.Div([graph1, graph2, interval_component])


@app.callback([
    Output('scatter-graph', 'figure'),
    Output('line-graph', 'figure')
    ],
    Input('interval-component', 'n_intervals')
    )
def update_graph(n):
    # fig = go.Scatter()
    fig1 = go.Figure()

    # fig1.add_trace(px.scatter(sample.df, x='x', y='y')) # NG
    # fig1.add_trace(px.line(sample.df, x='x', y='y')) " NG
    # fig1.add_trace(go.Scatter(sample.df, x='x', y='y', mode='lines')) # NG
    fig1.add_trace(go.Scatter(x=sample.df['x'], y=sample.df['y'], mode='lines', name='hi'))
    fig1.add_trace(go.Scatter(x=sample.df['x'], y=sample.df['y'], mode='markers', name='hihi'))
    fig1.add_trace(go.Scatter(x=sample.df['x'], y=sample.df['y'], mode='lines+markers', name='hi3'))

    fig1.update_xaxes(range=[sample.df['x'].min(),sample.df['x'].max()])
    fig1.update_yaxes(range=[0,1], )
    
    fig1.update_layout(yaxis_range=[0,2])

    
    fig2 = px.line(sample.df, x='x', y='y')
    
    return fig1, fig2

threading.Thread(target=sample.update_value).start()

if __name__ == '__main__':
    app.run_server(debug=False, host='localhost', port=8050)
