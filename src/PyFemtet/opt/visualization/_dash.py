import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import plotly.express as px

import numpy as np
import pandas as pd


class DashProcessMonitor:

    def __init__(self, FEMOpt):
        # 引数の処理
        self.FEMOpt = FEMOpt

        # application の準備
        self.app = dash.Dash(__name__)
        graph1 = dcc.Graph(id='scatter-graph', animate=False)
        graph2 = dcc.Graph(id='pp', animate=False)
        interval_component = dcc.Interval(
            id='interval',
            interval=2*1000,
            n_intervals=0
            )
        self.app.layout = html.Div([graph1, graph2, interval_component])

        # # decolator を書き下す形で callback を設定
        # self.app.callback(
        #     dash.dependencies.Output('hv', 'figure'),
        #     dash.dependencies.Input('interval', 'n_intervals')
        #     )(self.update_hv)
        # self.app.callback(
        #     dash.dependencies.Output('pp', 'figure'),
        #     dash.dependencies.Input('interval', 'n_intervals')
        #     )(self.update_pp)
        @self.app.callback(
            Output('scatter-graph', 'figure'),
            Input('interval', 'n_intervals')
            )
        def update(_):
            return self.update_hv()

    
    def start(self):
        self.app.run_server(debug=True, host='localhost', port=8050)
        import webbrowser
        webbrowser.open("http://localhost:8050")

    def update_hv(self):
        df = self.FEMOpt.history
        x = df['n_trial']
        y = df['hypervolume']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))
        fig.update_layout(
            xaxis_range=[0, x.max()*1.1],
            yaxis_range=[0, y.max()*1.1],
            )
        return fig

        
    def update_pp(self, _):
        # https://plotly.com/python/splom/
        ...
        return ...
    
