import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import plotly.express as px

import numpy as np
import pandas as pd




    

# class Sample:

#     def update_value(self):
#         while True:
#             self.df = pd.DataFrame(
#                 {
#                     'x':np.random.rand(10),
#                     'y':np.random.rand(10),
#                     }
#                 )
#             time.sleep(1)

# sample = Sample()

# application
app = dash.Dash(__name__)

# htmle elements
graph1 = dcc.Graph(id='hypervolume-graph', animate=False)
graph2 = dcc.Graph(id='pairplot-graph', animate=False)

interval_component = dcc.Interval(
    id='interval-component',
    interval=2*1000,
    n_intervals=0
    )

app.layout = html.Div([graph1, graph2, interval_component])

class DashProcessMonitor:
    def __init__(self, FEMOpt):
        self.df = FEMOpt.history

@app.callback([
    Output('hypervolume-graph', 'figure'),
    ],
    Input('interval-component', 'n_intervals')
    )
def update_graph1(n):
    fig1 = go.Scatter(x=)
    return fig1

if __name__ == '__main__':
    threading.Thread(target=sample.update_value).start()
    app.run_server(debug=False, host='localhost', port=8050)
